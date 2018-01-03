import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
import logging

import pandas as pd
import PIL

class IcebergDataset(Dataset):
    """Iceberg dataset."""

    def __init__(self, dtpath, offset, length, transform=None, test=False):
        """
        dtpath: file path for input data
        offset: offset of the data set
        length: length of the data set
        """
        dataframe = pd.read_json(dtpath)
        self.data = dataframe.iloc[offset:offset+length]
        self.length = length
        self.transform = transform
        self.test = test

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sample = dict()
        band_1 = np.array(row['band_1']).reshape(75, 75).astype(np.float32)
        band_2 = np.array(row['band_2']).reshape(75, 75).astype(np.float32)
        band_3 = band_1 - band_2
        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())
        sample['img'] = np.stack((r, g, b), axis=2)
        sample['img'] = PIL.Image.fromarray(sample['img'], 'RGB')
        if (self.transform):
            sample['img'] = self.transform(sample['img'])
        
        if not self.test:
            sample['label'] = row['is_iceberg']
            
        sample['id'] = row['id']
        sample['angle'] = str(row['inc_angle'])
        return sample
        
def splitDataset(dtpath, points, transform=None):
    """
    dtpath: file path for input data
    points: list contains 2 points
    """
    df = pd.read_json(dtpath)
    length = df.shape[0]
    train_end = int(length * points[0])
    train_len = train_end
    ensem_end = int(length * points[1])
    ensem_len = ensem_end - train_end
    valid_end = length - 1
    valid_len = valid_end - ensem_end
    return IcebergDataset(dtpath, 0, train_len, transform), \
        IcebergDataset(dtpath, train_end, ensem_len, transform), \
        IcebergDataset(dtpath, ensem_end, valid_len, transform)
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
MODEL_STORE_KEY = 'model_state_dict'
OPTIM_STORE_KEY = 'optim_state_dict'
EPOCH_KEY = 'epoch'

class Iceberg:
    
    def __init__(self, dtpath, cppath, model, optim, nepoch, bsize):
        """
        cppath: file path for save/resume models
        dtpath: file path for input data
        model: deep learning model, pretrained params loaded
        optim: deep learning optimizer
        nepoch: number of epochs
        """
        # member from constructor params
        self.dtpath = dtpath
        self.cppath = cppath
        self.model = model
        self.optim = optim
        self.nepoch = nepoch
        self.bsize = bsize
        
        # member to initialize
        self.epoch = 0
        self.train_btime = AverageMeter()
        self.train_dtime = AverageMeter()
        self.train_loss = AverageMeter()
        self.train_top1 = AverageMeter()
        self.train_top2 = AverageMeter()
        
        self.valid_btime = AverageMeter()
        self.valid_loss = AverageMeter()
        self.valid_top1 = AverageMeter()
        self.valid_top2 = AverageMeter()
        
        # other initilization
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        directory = self.cppath[:-4]
        if not os.path.exists(directory):
            os.makedirs(directory)
        logpath = os.path.join(directory, 'stdout.log')
        
        fileHandler = logging.FileHandler(logpath)
        self.logger.addHandler(fileHandler)
        
        consoleHandler = logging.StreamHandler()
        self.logger.addHandler(consoleHandler)
        
        self.resume()
        
    def resume(self):
        if os.path.isfile(self.cppath):
            self.logger.info("=> loading checkpoint '{}'".format(self.cppath))
            cp = torch.load(self.cppath)
            self.model.load_state_dict(cp[MODEL_STORE_KEY])
            self.optim.load_state_dict(cp[OPTIM_STORE_KEY])
            self.epoch = cp[EPOCH_KEY]
            self.logger.info("=> loaded checkpoint '{}' (epoch {})".format(self.cppath, self.epoch))
            
    def store(self):
        state = {
            EPOCH_KEY: self.epoch,
            MODEL_STORE_KEY: self.model.state_dict(),
            OPTIM_STORE_KEY: self.optim.state_dict()
        }
        torch.save(state, self.cppath)
    
    def run(self, transform=None):
        """run the training and validate process"""
        train_dataset, _, valid_dataset = splitDataset(self.dtpath, [0.8, 0.9], transform)
        self.train_loader = DataLoader(train_dataset, batch_size=self.bsize, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None)
        self.valid_loader = DataLoader(valid_dataset, batch_size=self.bsize, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None)
        
        # To think: should I use some other loss function?
        self.criterion = nn.CrossEntropyLoss().cuda()
        
        stepoch = self.epoch
        for epoch in range(stepoch, self.nepoch):
            self.epoch = epoch
            
            # train for one epoch
            self.train()
            
            # evaluate on validation set
            p1 = self.validate()
            
            # save check point
            self.store()
    
    def infer(self, ttpath):
        
        # switch to eval mode
        self.model.eval()
        
        df = pd.read_json(ttpath)
        length = df.shape[0]
        test_dataset = IcebergDataset(ttpath, 0, length, transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]), True)
        test_loader = DataLoader(test_dataset, batch_size=self.bsize, shuffle=False,
            num_workers=0, pin_memory=True, sampler=None)
        
        for crop in range(10):
            directory = self.cppath[:-4]
            if not os.path.exists(directory):
                os.makedirs(directory)
            path = os.path.join(directory, "crop"+str(crop)+".csv")
            df = pd.DataFrame(columns=["id", "is_iceberg"])
            with open(path, 'w') as f:
                df.to_csv(f, index=False)
                
            for i_batch, sample_batched in enumerate(test_loader):
                var_images = torch.autograd.Variable(sample_batched['img'].cuda())
                logits = self.model(var_images)
                softmax = nn.Softmax(dim=1)
                props = softmax(logits)[:,1].data
                ids = sample_batched['id']
                df = pd.DataFrame({
                    'id': ids,
                    'is_iceberg': props.cpu()
                })
                with open(path, 'a') as f:
                    df.to_csv(f, index=False, header=False)
                
                self.logger.info('Crop: [{0}][{1}/{2}]'.format(
                       crop, i_batch, len(test_loader)))
    
    def train(self):
        
        # switch to train mode
        self.model.train()

        end = time.time()
        for i_batch, sample_batched in enumerate(self.train_loader):
            """Start batch train"""
            var_images = torch.autograd.Variable(sample_batched['img'].cuda())
            tensor_target = sample_batched['label'].cuda()
            var_target = torch.autograd.Variable(tensor_target)
            logits = self.model(var_images)
            loss = self.criterion(logits, var_target)
        
            prec1, prec2 = self.accuracy(logits.data, tensor_target, topk=(1, 2))
            batch_size = sample_batched['label'].size(0)
            self.train_loss.update(loss.data[0], batch_size)
            self.train_top1.update(prec1[0], batch_size)
            self.train_top2.update(prec2[0], batch_size)
        
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
            self.train_btime.update(time.time() - end)
            end = time.time()
        
            if i_batch % 1 == 0:
                self.logger.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       self.epoch, i_batch, len(self.train_loader),
                       batch_time=self.train_btime, loss=self.train_loss,
                       top1=self.train_top1, top2=self.train_top2))
    
    def validate(self):
        
        # switch to eval mode
        self.model.eval()
        
        end = time.time()
        for i_batch, sample_batched in enumerate(self.valid_loader):
            """Start batch validation"""
            var_images = torch.autograd.Variable(sample_batched['img'].cuda())
            tensor_target = sample_batched['label'].cuda()
            var_target = torch.autograd.Variable(tensor_target)
            logits = self.model(var_images)
            loss = self.criterion(logits, var_target)

            prec1, prec2 = self.accuracy(logits.data, tensor_target, topk=(1, 2))
            batch_size = sample_batched['label'].size(0)
            self.valid_loss.update(loss.data[0], batch_size)
            self.valid_top1.update(prec1[0], batch_size)
            self.valid_top2.update(prec2[0], batch_size)

            self.valid_btime.update(time.time() - end)
            end = time.time()

            if i_batch % 1 == 0:
                self.logger.info('Validate: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       i_batch, len(self.valid_loader), batch_time=self.valid_btime, 
                       loss=self.valid_loss, top1=self.valid_top1, top2=self.valid_top2))
    
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
