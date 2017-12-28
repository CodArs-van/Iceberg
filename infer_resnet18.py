from models.resnet import resnet18
from iceberg import *
from torch.optim import SGD

lr = 0.0005
mom = 0.9
wd = 1e-4

model = resnet18()
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), lr, momentum=mom, weight_decay=wd)
iceberg = Iceberg('./data/train.json', './data/resnet18_lr5e-4_mom9e-1_wd1e-4_bs256_model.pth', model, optim, 100000, 384)
iceberg.infer('./data/test.json')
