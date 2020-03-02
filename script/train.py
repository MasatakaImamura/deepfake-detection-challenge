import os, glob, random, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Normalize
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

from utils.dfdc_dataset import DeepfakeDataset_per_img
from utils.data_augumentation import ImageTransform_2
from utils.pure_trainer import train_model

# Parser  ################################################################
parser = argparse.ArgumentParser()
parser.add_argument('-exp', '--experiencename')
parser.add_argument('-m', '--modelname')
parser.add_argument('-b', '--batchsize', type=int, default=4)
parser.add_argument('-bn', '--batchnum', type=int, default=10000)
parser.add_argument('-ims', '--imgsize', type=int, default=120)
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
epoch = 100
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Load Data  ##################################################################
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
metadata = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

# ImageTransform  ##################################################################
transform = ImageTransform_2(size=img_size, mean=mean, std=std)

# Dataset, DataLoader  ##################################################################
random.shuffle(faces)
train_size = 0.9

train_dataset = DeepfakeDataset_per_img(faces[:int(len(faces) * train_size)], metadata, transform, 'train')
val_dataset = DeepfakeDataset_per_img(faces[int(len(faces) * train_size):], metadata, transform, 'val')

dataloaders = {
    'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
    'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
}

# Model  #########################################################################
net = EfficientNet.from_pretrained(args.modelname, num_classes=1)

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Train  #########################################################################

train_model(dataloaders, net, device, optimizer, scheduler, batch_num, num_epochs=epoch, exp=exp)

