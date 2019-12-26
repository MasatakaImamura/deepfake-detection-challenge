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


mov_path = mov_path[0]
transform = ImageTransform(resize=224)

# Label
label = metadata[metadata['mov'] == mov_path.split('/')[-1]]['label'].values[0]

# Movie to Image
# img_list = []
# for i in range(int(20)):
#     image = get_img_from_mov(mov_path)[int(i*10)]  # Only First Frame Face
#     # FaceCrop
#     image = detect_face_mtcnn(image, device)
#     # Transform
#     image = transform(image, 'train')
#     img_list.append(image)
#
# img_list = torch.stack(img_list)
#
# print(img_list.size())


for img, label, _ in train_dataloader:

    img = img.squeeze(0)

    model = model_init(model_name)

    out = model(img)
    preds = torch.sigmoid(out.view(-1)).mean().unsqueeze(0).to(device)

    print(label)
    print(preds)
    loss = criterion(preds, label)
    print(loss)

    break




# if len(img) == 0:
#
# print(img)
# print(label)
# print(label.size())
# print(label.item())
#
#
# model = model_init(model_name)
# z = torch.randn(20, 3, 224, 224)
#
# out = model(z)
# pred = torch.sigmoid(out.view(-1)).mean().unsqueeze(0).to(device)
# loss = criterion(pred, label)
# print(pred)
# print(loss)
# print(loss.size())
# print(loss.item())
# print(pred.item())
