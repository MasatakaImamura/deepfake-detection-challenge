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
from facenet_pytorch import InceptionResnetV1, MTCNN


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
real_mov_num = 50


# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
mov_path = get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=real_mov_num)

# Preprocessing  ################################################################
# Divide Train, Vaild Data
train_mov_path, val_mov_path = train_test_split(mov_path, test_size=0.1, random_state=seed)

# Dataset
train_dataset = DeepfakeDataset_continuous(
    train_mov_path, metadata, device, transform=ImageTransform(img_size),
    phase='train', img_num=img_num, frame_window=frame_window)

val_dataset = DeepfakeDataset_continuous(
    val_mov_path, metadata, device, transform=ImageTransform(img_size),
    phase='val', img_num=img_num, frame_window=frame_window)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

i = 0


model = InceptionResnetV1(pretrained='vggface2', num_classes=1, device=device, classify=True).train()

iterater = iter(train_dataloader)

img, label, _ = next(iterater)

out = model(img.squeeze(0))
print(out)
print(out.size())
print(torch.sigmoid(out.view(-1)).mean())
print('')

img, label, _ = next(iterater)

out = model(img.squeeze(0))
print(out)
print(out.size())
print(torch.sigmoid(out.view(-1)).mean())
