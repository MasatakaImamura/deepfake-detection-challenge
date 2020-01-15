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
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_continuous, face_img_generator
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from utils.mesonet import Mesonet

from utils.logger import create_logger, get_logger


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 4
epoch = 8
model_name = 'resnet50'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()
img_num = 5
frame_window = 5
real_mov_num = None

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                            real_mov_num=real_mov_num, train_size=0.9, seed=seed)

imgs = get_img_from_mov(train_mov_path[0])[0]

print(imgs.shape)

mtcnn = MTCNN(keep_all=False, device=device, margin=0).eval()

faces = mtcnn(imgs[np.newaxis, :, :, :])

print(faces.shape)

