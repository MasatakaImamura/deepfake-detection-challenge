import matplotlib.pyplot as plt
import cv2
import numpy as np
import time

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init, convLSTM, convLSTM_resnet
from utils.Conv3D import Conv3dnet
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_continuous, face_img_generator, DeepfakeDataset_continuous_2
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from utils.mesonet import Mesonet

from utils.logger import create_logger, get_logger


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 16
epoch = 8
model_name = 'resnet50'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
img_num = 10
frame_window = 5
real_mov_num = None

# Set Seed
seed_everything(seed)

detector = MTCNN(image_size=img_size, margin=14, keep_all=True, factor=0.5, device=device).eval()

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                            real_mov_num=real_mov_num, train_size=0.9, seed=seed)

# Loss Function  ################################################################
criterion = nn.BCEWithLogitsLoss(reduction='sum')

# Preprocessing  ################################################################
# Dataset
train_dataset = DeepfakeDataset_continuous(
    train_mov_path, metadata, device, transform=ImageTransform(img_size),
    phase='train', img_size=img_size, img_num=img_num, frame_window=frame_window)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


train_dataset_2 = DeepfakeDataset_continuous_2(
    train_mov_path, metadata, device, detector, img_num=img_num, img_size=img_size, frame_window=frame_window)

train_dataloader_2 = DataLoader(train_dataset_2, batch_size=batch_size, shuffle=True)

print('DataLoader Already')


# Test


since = time.time()

iters = iter(train_dataloader)

img_list, label, mov_path = next(iters)

elapsedtime = time.time() - since

print(img_list.size())
print(elapsedtime)

since = time.time()

iters = iter(train_dataloader_2)

img_list, label, mov_path = next(iters)

elapsedtime = time.time() - since

print(img_list.size())
print(elapsedtime)
