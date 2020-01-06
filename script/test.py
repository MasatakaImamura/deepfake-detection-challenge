import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init, convLSTM, convLSTM_resnet
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_continuous
from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 4
epoch = 8
model_name = 'resnet50'
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
criterion = nn.CrossEntropyLoss()
img_num = 5
frame_window = 20
real_mov_num = None

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################

from PIL import Image, ImageDraw
import numpy as np

metadata = get_metadata(data_dir)
mov_path = get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=real_mov_num)

mov = get_img_from_mov(mov_path[0])
img = mov[80]
#
# img = Image.open("D:\\1.jpg")

face, points, boxes = detect_face_mtcnn(np.array(img), device)

print(face.shape)
print(points.shape)

print(points)
