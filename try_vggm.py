from models.vggm import vggm
from iceberg import *
from torch.optim import SGD

lr = 0.0005
mom = 0.9
wd = 1e-4
bsize = 256

model = vggm(num_classes=1000, pretrained=None)
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), lr, momentum=mom, weight_decay=wd)
iceberg = Iceberg('./data/train.json', 
                  './data/vggm_lr5e-4_mom9e-1_wd1e-4_bs{}_model.pth'.format(bsize), model, optim, 10000, bsize)
iceberg.run(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(221),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
