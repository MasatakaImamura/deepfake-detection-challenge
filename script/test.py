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
import torchvision

from models.model_init import model_init, convLSTM, convLSTM_resnet
from models.Conv3D import Efficientnet_3d, Facenet_3d
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov_2
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_3d, face_img_generator, DeepfakeDataset_3d_faster
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.xception import Xception


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 4
epoch = 8
model_name = 'resnet152'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
img_num = 10
frame_window = 5
real_mov_num = None

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


z = torch.randn(4, 14, 3, 224, 224)

model = Facenet_3d()
out = model(z)
print(out.size())



