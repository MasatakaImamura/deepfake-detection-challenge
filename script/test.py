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
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_continuous, face_img_generator
# from utils.trainer import train_model
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


# metadata = get_metadata(data_dir)
#
# mov_path = get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=real_mov_num)
#
# # Divide Train, Vaild Data
# train_mov_path, val_mov_path = train_test_split(mov_path, test_size=0.1, random_state=seed)
#
# # Dataset
# train_dataset = DeepfakeDataset_continuous(
#     train_mov_path, metadata, device, transform=ImageTransform(img_size), phase='train')
#
# val_dataset = DeepfakeDataset_continuous(
#     val_mov_path, metadata, device, transform=ImageTransform(img_size), phase='val')
#
# # DataLoader
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# dataloader_dict = {
#     'train': train_dataloader,
#     'val': val_dataloader
# }
#
# print('DataLoader Already')
# iters = iter(train_dataloader)
# img_list, label, path = next(iters)
#
# print(img_list.size())

z = torch.randn((4, 50, 3, 224, 224))
# b, D, C, H, W -> b, C, D, H, W
z = z.permute(0, 2, 1, 3, 4)

print(z.size())
