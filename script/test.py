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
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_per_img, DeepfakeDataset_per_img_2, DeepfakeDataset_per_img_3
from utils.data_augumentation import GroupImageTransform, ImageTransform
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.Facenet_3d import Facenet_3d

from models.Efficientnet import Efficientnet_LSTM
from efficientnet_pytorch import EfficientNet

import pandas as pd


sep = None
# OSの違いによるパス分割を定義
if os.name == 'nt':
    sep = '\\'
elif os.name == 'posix':
    sep = '/'

# Config  ################################################################
faces_dir = '../data/faces_temp'
meta_dir = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

img_num = 15
batch_size = 4
batch_num = 10000
img_size = 120
epoch = 1000
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

# # Load Data  ##################################################################
# faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
# metadata = pd.read_csv(os.path.join(meta_dir, 'meta2.csv'))
#
# # ImageTransform  ##################################################################
# transform = ImageTransform(size=img_size, mean=mean, std=std)
#
# # Dataset, DataLoader  ##################################################################
# train_size = 0.9
# metadata = metadata.sample(frac=1).reset_index(drop=True)
# train_meta = metadata.iloc[:100, :]
#
# train_dataset = DeepfakeDataset_per_img_3(faces, train_meta, transform, 'train', sample_size=12)
#
#
# img, label = train_dataset.__getitem__(0)
#
# print(img.size())

net = torchvision.models.resnet34(pretrained=False)

net.relu = nn.ELU(inplace=True)
net.layer1[0].relu = nn.ELU(inplace=True)

print(net)

print('#' * 60)
for name, param in net.named_parameters():
    print(name)
