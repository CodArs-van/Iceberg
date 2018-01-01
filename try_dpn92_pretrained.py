from models.dpn_model_factory import create_model
from iceberg import *
from torch.optim import SGD

lr = 0.0005
mom = 0.9
wd = 1e-4
bsize = 172
mname = 'dpn92'

model = create_model(mname, 1000, True)
model = torch.nn.DataParallel(model).cuda()
optim = SGD(model.parameters(), lr, momentum=mom, weight_decay=wd)
iceberg = Iceberg('./data/train.json', 
                  './data/{}_lr5e-4_mom9e-1_wd1e-4_bs{}_pretrained_model.pth'.format(mname, bsize), model, optim, 10000, bsize)
iceberg.run(transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]))
