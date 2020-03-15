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

from models.Efficientnet import Efficientnet_3d
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

net = torchvision.models.resnet50(pretrained=True)
net.fc = nn.Linear(in_features=net.fc.in_features, out_features=1)
print(net)
