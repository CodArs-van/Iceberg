from models.inceptionresnetv2 import inceptionresnetv2
from iceberg import *
from torch.optim import SGD

lr = 0.0005
mom = 0.9
wd = 1e-4
bsize = 128

model = inceptionresnetv2(1000)
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), lr, momentum=mom, weight_decay=wd)
iceberg = Iceberg('./data/train.json', 
                  './data/inceptionresnetv2_lr5e-4_mom9e-1_wd1e-4_bs{}_pretrained_model.pth'.format(bsize), model, optim, 10000, bsize)
iceberg.run(transforms.Compose([
            transforms.Resize(320),
            transforms.RandomCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
