import matplotlib.pyplot as plt
import cv2, os, random, glob
import numpy as np
import time
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms

from utils.data_augumentation import ImageTransform, ImageTransform_2
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov, freeze_until
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_per_img, DeepfakeDataset_per_img_2
from utils.data_augumentation import GroupImageTransform, ImageTransform
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.Facenet_3d import Facenet_3d

from models.Efficientnet_3d import Efficientnet_3d, Efficientnet_2d
from efficientnet_pytorch import EfficientNet

import pandas as pd


sep = None
# OSの違いによるパス分割を定義
if os.name == 'nt':
    sep = '\\'
elif os.name == 'posix':
    sep = '/'

faces_dir = '../data/faces_temp'
meta_dir = '../data/meta'
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# Load Data  ##################################################################
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
metadata = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

# ImageTransform  ##################################################################
transform = ImageTransform_2(size=224, mean=mean, std=std)

# Dataset, DataLoader  ##################################################################
train_size = 0.9
metadata = metadata.sample(frac=1)
train_meta = metadata.iloc[:int(len(metadata)*train_size), :]
val_meta = metadata.iloc[int(len(metadata)*train_size):, :]

print(train_meta.shape)
print(val_meta.shape)
print(metadata.shape)

train_dataset = DeepfakeDataset_per_img_2(faces, train_meta, transform, 'train', sample_size=1000)

img, label = train_dataset.__getitem__(8)

print(img)
print(label)


del_mov_name = []

for i in tqdm(range(len(metadata))):
    row = metadata.iloc[i]
    mov_name = row['mov']
    target = [c for c in faces if mov_name in c]

    if len(target) == 0:
        del_mov_name.append(mov_name)

_metadata = metadata[~metadata['mov'].isin(del_mov_name)]

print(metadata.shape)
print(_metadata.shape)

_metadata.to_csv('../data/meta/meta2.csv', index=False)