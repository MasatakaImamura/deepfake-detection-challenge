import matplotlib.pyplot as plt
import cv2, os, random
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
from torchvision.transforms import Normalize

from models.model_init import model_init, convLSTM, convLSTM_resnet
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov, freeze_until
from utils.dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_2d, DeepfakeDataset_3d_realfake
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.Facenet_3d import Facenet_3d

from efficientnet_pytorch import EfficientNet

from models.blazeface import BlazeFace


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 4
epoch = 8
model_name = 'resnet152'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_num = 14
frame_window = 20
real_mov_num = None
cascade_path = '../haarcascade/haarcascade_frontalface_alt2.xml'

# Set Seed
seed_everything(seed)

detector = MTCNN(image_size=img_size, margin=14, keep_all=False,
                 select_largest=False, factor=0.5, device=device, post_process=False).eval()

# Loss Function  ################################################################
criterion = nn.BCEWithLogitsLoss(reduction='sum')

# # Set Mov_file path  ################################################################
# metadata = get_metadata(data_dir)
# train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
#                                             real_mov_num=real_mov_num, train_size=0.9, seed=seed)
#
# imgs = get_img_from_mov(train_mov_path[0], num_img=5, frame_window=10)

# dataset = DeepfakeDataset_3d_realfake(data_dir, metadata, device, detector, img_num=14, img_size=224, frame_window=20)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


import torchvision.models as models

net = Facenet_3d()
# print(net)
#
# for name, params in net.named_parameters():
#     print(name)

freeze_until(net, "facenet.repeat_3.0.branch0.conv.weight")

print([params for params in net.parameters() if params.requires_grad])
