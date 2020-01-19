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
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov_2
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_3d, face_img_generator, DeepfakeDataset_3d_faster
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from utils.mesonet import Meso4, MesoInception4

from utils.logger import create_logger, get_logger


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

detector = MTCNN(image_size=img_size, margin=14, keep_all=True, factor=0.5, device=device).eval()

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                            real_mov_num=real_mov_num, train_size=0.9, seed=seed)

# Loss Function  ################################################################
criterion = nn.BCEWithLogitsLoss(reduction='sum')

# Preprocessing  ################################################################

gen = DeepfakeDataset_3d_faster(train_mov_path, metadata, device, detector, img_num=20, img_size=224, frame_window=10)

img_list = get_img_from_mov_2(train_mov_path[0], img_num, frame_window)


imgss = []
from PIL import Image

# for i in range(3):
#     plt.imshow(img_list[i])
#     plt.show()
#
#     img = detector(Image.fromarray(img_list[i]))
#
#     # plt.imshow(img)
#     # plt.show()
#
#     # img = ImageTransform(img_size)(img, 'train')
#
#     img = img /2 + 0.5
#     print(img.size())
#
#     for i in range(img.size(0)):
#         plt.imshow(img[i, :, :, :].permute(1, 2, 0).numpy())
#         plt.show()
#
#     imgss.append(img)
#
# print(imgss)
# print(torch.stack(imgss).size())

img, label, path = gen.__getitem__(0)

print(img.size())
print(label)
print(path)

for i in range(5):
    print(img[i].size())
    print(torch.sum(img[i]))
    print(img[i])
    plt.imshow(img[i].permute(1, 2, 0).numpy())
    plt.show()
