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

from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov, freeze_until
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_per_img
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

# 動画全ファイルのパスを取得
# faces_dir = '../data/faces_temp'
# faces = glob.glob(os.path.join(faces_dir, '*.jpg'))


transform = ImageTransform(120)

target = '../data/faces_temp/aaagqkcdis.mp4_FAKE_frame_0.jpg'

img_pil = Image.open(target)

img_pil = np.array(img_pil)

img_cv = cv2.imread(target)
img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

img_cv_pil = Image.fromarray(img_cv)
img_cv_pil =transform(img_cv_pil, 'train')
img_cv_pil *= 255.


print(img_pil)
print('&&&&&&&&&&&&&&&&&&&&&&')
print(img_cv)
print('&&&&&&&&&&&&&&&&&&&&&&')
print(img_cv_pil.permute(1, 2, 0))

print(img_pil.shape)
print(img_cv.shape)
