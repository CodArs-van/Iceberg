from models.resnet import resnet18
from iceberg import *
from torch.optim import SGD

lr = '1e-3'
mom = '95e-2'
wd = '1e-4'
bsize = 64

model = resnet18(False, num_classes=2)
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), float(lr), momentum=float(mom), weight_decay=float(wd))
iceberg = Iceberg('./data/train.json', './data/resnet18_lr{}_mom{}_wd{}_bs{}_model.pth'.format(lr, mom, wd, bsize), model, optim, 128, bsize)
iceberg.run(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
