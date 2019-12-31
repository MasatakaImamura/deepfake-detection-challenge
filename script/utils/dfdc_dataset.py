import os
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_img_from_mov, detect_face, detect_face_mtcnn

if os.name == 'nt':
    sep = '\\'
elif os.name == 'posix':
    sep = '/'

# 動画から1枚だけの画像を出力する
# getting_idxで動画のフレームから任意の位置の画像をピックアップする
class DeepfakeDataset(Dataset):
    def __init__(self, file_list, metadata, device, transform=None, phase='train', getting_idx=0):
        self.file_list = file_list
        self.metadata = metadata
        self.transform = transform
        self.device = device
        self.phase = phase
        self.getting_idx = getting_idx

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]

        # Label
        label = self.metadata[self.metadata['mov'] == mov_path.split(sep)[-1]]['label'].values

        if label == 'FAKE':
            label = 1
        else:
            label = 0

        # Movie to Image
        try:
            image = get_img_from_mov(mov_path)[self.getting_idx]  # Specified Frame Only
            # FaceCrop
            image = detect_face_mtcnn(image, self.device)
            # Transform
            image = self.transform(image, self.phase)
        except:
            image = torch.randn((3, self.transform.size, self.transform.size), dtype=torch.float)
            label = 1

        return image, label, mov_path


# 1動画ごとの連続画像を生成
# img_numで画像の最大枚数  frame_windowで動画の間隔を指定
class DeepfakeDataset_continuous(Dataset):
    def __init__(self, file_list, metadata, device, transform=None, phase='train', img_num=20, frame_window=10):
        self.file_list = file_list
        self.metadata = metadata
        self.device = device
        self.transform = transform
        self.phase = phase
        self.img_num = img_num
        self.frame_window = frame_window

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]

        # Label
        label = self.metadata[self.metadata['mov'] == mov_path.split(sep)[-1]]['label'].values[0]

        if label == 'FAKE':
            label = 1
        else:
            label = 0

        # Movie to Image
        img_list = []
        for i in range(int(self.img_num)):
            try:
                image = get_img_from_mov(mov_path)[int(i*self.frame_window)]  # Only First Frame Face
                # FaceCrop
                image = detect_face_mtcnn(image, self.device)
                # Transform
                image = self.transform(image, self.phase)
                img_list.append(image)
            except:
                image = torch.randn(3, 224, 224)
                img_list.append(image)

        img_list = torch.stack(img_list)

        return img_list, label, mov_path
