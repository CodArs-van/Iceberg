from models.resnet import resnet50
from iceberg import *
from torch.optim import SGD

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', required=True)
parser.add_argument('--mt', required=True)
parser.add_argument('--wd', required=True)
parser.add_argument('--bs', type=int, required=True)
results = parser.parse_args()

lr = results.lr
mom = results.mt
wd = results.wd
bsize = results.bs

model = resnet50(True, num_classes=2)
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), float(lr), momentum=float(mom), weight_decay=float(wd))
iceberg = Iceberg('./data/train.json', 
                  './data/resnet50_lr{}_mom{}_wd{}_bs{}_pretrained_model.pth'.format(lr, mom, wd, bsize), model, optim, 128, bsize)
iceberg.run(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
