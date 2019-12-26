import matplotlib.pyplot as plt
import cv2

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov
from utils.dfdc_dataset import DeepfakeDataset_idx0, DeepfakeDataset_continuous
from utils.trainer import train_model


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 1
epoch = 8
model_name = 'resnet50'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
img_num = 5
frame_window = 20


# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
mov_path = get_mov_path(metadata, data_dir, fake_per_real=1)

print(mov_path[:5])

imgs = get_img_from_mov(mov_path[0])

print(len(imgs))
plt.imshow(imgs[0])
plt.show()

for i in range(30):
    face = detect_face_mtcnn(imgs[i*5], device)
    plt.imshow(cv2.resize(face[0], (224, 224)))
    plt.show()
