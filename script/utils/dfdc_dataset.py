import os, cv2, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize
from PIL import Image

from .utils import get_img_from_mov, detect_face, detect_face_mtcnn

# OSの違いによるパス分割を定義
if os.name == 'nt':
    sep = '\\'
elif os.name == 'posix':
    sep = '/'


class DeepfakeDataset(Dataset):
    '''
    1動画ごとの画像をまとめて出力する
    output_size: (frames=15, channels=3, width, heights)

    '''

    def __init__(self, faces_img_path, metadata, transform, phase='train', img_size=224, img_num=15):
        self.faces_img_path = faces_img_path
        self.metadata = metadata
        self.transform = transform
        self.phase = phase
        self.img_size = img_size
        self.img_num = img_num

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        img_name = self.metadata['mov'].unique().tolist()[idx]
        # Extract Target Mov path only
        target_mov_path = [path for path in self.faces_img_path if img_name in path]

        # If target_mov_path is empty, return random noise and label=1
        if len(target_mov_path) == 0:
            img_list = torch.randn(15, 3, self.img_size, self.img_size)
            label = 1.0
            return img_list, label

        # Each Image(PIL) get into List
        img_list = []
        for t in target_mov_path:
            img = Image.open(t)
            img_list.append(img)

        # Shuffle Image List
        random.shuffle(img_list)

        # Get Label
        label = target_mov_path[0].split(sep)[1].split('_')[1]
        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Transform Images
        img_list = self.transform(img_list, self.phase)

        # Remained only Img Num
        if img_list.size(0) > self.img_num:
            img_list = img_list[:self.img_num, :, :, :]
        elif img_list.size(0) < self.img_num:
            img_list = torch.randn(self.img_num, 3, self.img_size, self.img_size)
            label = 1.0

        return img_list, label


class DeepfakeDataset_per_img(Dataset):
    '''
    1画像ごとに出力する
    output_size: (channels=3, width, heights)

    '''

    def __init__(self, faces_img_path, metadata, transform, phase='train'):
        self.faces_img_path = faces_img_path
        self.metadata = metadata
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.faces_img_path)

    def __getitem__(self, idx):

        img_path = self.faces_img_path[idx]

        img = Image.open(img_path)

        # Get Label
        label = img_path.split(sep)[1].split('_')[1]
        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Transform Images
        img = self.transform(img, self.phase)

        return img, label
