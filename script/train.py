import os, glob, argparse
import pandas as pd

import torch
from torch import nn
from torch import optim
import torchvision
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts
from efficientnet_pytorch import EfficientNet

from .utils.dfdc_dataset import DeepfakeDataset
from .utils.augumentation import ImageTransform, ImageTransform_2, ImageTransform_3
from .utils.trainer import train_model
from .utils.radam import RAdam
from .utils.utils import seed_everything

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--experiencename')
parser.add_argument('-m', '--modelname')
parser.add_argument('-b', '--batchsize', type=int, default=4)
parser.add_argument('-bn', '--batchnum', type=int, default=10000)
parser.add_argument('-ims', '--imgsize', type=int, default=120)
parser.add_argument('-sch', '--scheduler', choices=['step', 'exp', 'cycle'], default='step')
parser.add_argument('-opt', '--optimizer', choices=['adam', 'radam', 'sgd'], default='adam')
parser.add_argument('-lr', '--learningrate', type=float, default=0.001)
parser.add_argument('-tr', '--imagetransform', type=int, default=1)
args = parser.parse_args()

# Config  ################################################################
faces_dir = '../data/faces_temp'
meta_dir = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

exp = args.experiencename
img_num = 15
batch_size = args.batchsize
batch_num = args.batchnum
img_size = args.imgsize
epoch = 1000
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
seed = 0

seed_everything(seed)

# Load Data  ##################################################################
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
metadata = pd.read_csv(os.path.join(meta_dir, 'meta2.csv'))

# ImageTransform  ##################################################################
transform = None
if args.imagetransform == 1:
    transform = ImageTransform(size=img_size, mean=mean, std=std)
elif args.imagetransform == 2:
    transform = ImageTransform_2(size=img_size, mean=mean, std=std)
elif args.imagetransform == 3:
    transform = ImageTransform_3(size=img_size, mean=mean, std=std)

# Dataset, DataLoader  ##################################################################
train_size = 0.9
metadata = metadata.sample(frac=1).reset_index(drop=True)
train_meta = metadata.iloc[:int(len(metadata)*train_size), :]
val_meta = metadata.iloc[int(len(metadata)*train_size):, :]

train_dataset = DeepfakeDataset(faces, train_meta, transform, 'train', sample_size=12000)
val_dataset = DeepfakeDataset(faces, val_meta, transform, 'val', sample_size=1200)

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

# Model  #########################################################################
net = None
if 'efficientnet' in args.modelname:
    net = EfficientNet.from_pretrained(args.modelname, num_classes=1)
elif 'resnet34' in args.modelname:
    net = torchvision.models.resnet34(pretrained=True)
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=1)
elif 'resnet50' in args.modelname:
    net = torchvision.models.resnet50(pretrained=True)
    net.fc = nn.Linear(in_features=net.fc.in_features, out_features=1)

# Optimizer
optimizer = None
if args.optimizer == 'adam':
    optimizer = optim.Adam(net.parameters(), lr=args.learningrate)
elif args.optimizer == 'radam':
    optimizer = RAdam(net.parameters(), lr=args.learningrate)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.learningrate, momentum=0.9, weight_decay=0.)

# Scheduler
scheduler = None
if 'step' in args.scheduler:
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
elif 'exp' in args.scheduler:
    scheduler = ExponentialLR(optimizer, gamma=0.95)
elif 'cycle' in args.scheduler:
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=args.learningrate*0.1)

# Train  #########################################################################
train_model(dataloaders, net, device, optimizer, scheduler, batch_num, num_epochs=epoch, exp=exp)

