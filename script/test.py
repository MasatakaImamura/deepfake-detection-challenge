import matplotlib.pyplot as plt
import cv2, os, random, glob
import numpy as np
import time
from PIL import Image

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms

from models.model_init import model_init, convLSTM, convLSTM_resnet
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov, freeze_until
from utils.dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_2d, DeepfakeDataset_3d_realfake, DeepfakeDataset
from utils.data_augumentation import GroupImageTransform
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.Facenet_3d import Facenet_3d

from models.Efficientnet_3d import Efficientnet_3d
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
img_size = 120

# Load Data
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
meta = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

transform = GroupImageTransform(size=img_size)
dataset = DeepfakeDataset(faces, meta, transform, phase='train', img_size=img_size)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

img, label = next(iter(dataloader))

print(img.size())
print(label.size())
img = img.to(device)
label = label.to(device)

net = Efficientnet_3d(output_size=1)
net = net.to(device)

out = net(img)

criterion = nn.BCEWithLogitsLoss()

loss = criterion(out, label.unsqueeze(1).float())

print(loss)

