import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_img_from_mov, detect_face


class DeepfakeDataset_idx0(Dataset):
    def __init__(self, file_list, metadata, transform=None, phase='train'):
        self.file_list = file_list
        self.metadata = metadata
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]

        # Label
        label = self.metadata[self.metadata['mov'] == mov_path.split('\\')[-1]]['label'].values

        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Movie to Image
        try:
            image = get_img_from_mov(mov_path)[0]  # Only First Frame Face
            # FaceCrop
            image = detect_face(image)[0]
            # Transform
            image = self.transform(image, self.phase)
        except:
            image = torch.ones((3, self.transform.size, self.transform.size), dtype=torch.float)
            label = 1.0

        return image, label, mov_path


# 1動画ごとの連続画像を生成
# img_numで画像の最大枚数  frame_windowで動画の間隔を指定
class DeepfakeDataset_continuous(Dataset):
    def __init__(self, file_list, metadata, transform=None, phase='train', img_num=20, frame_window=10):
        self.file_list = file_list
        self.metadata = metadata
        self.transform = transform
        self.phase = phase
        self.img_num = img_num
        self.frame_window = frame_window

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]

        # Label
        label = self.metadata[self.metadata['mov'] == mov_path.split('\\')[-1]]['label'].values

        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Movie to Image
        img_list = []
        for i in range(int(self.img_num)):
            try:
                image = get_img_from_mov(mov_path)[int(i*self.frame_window)]  # Only First Frame Face
                # FaceCrop
                image = detect_face(image)[0]
                # Transform
                image = self.transform(image, self.phase)
                img_list.append(image)
            except:
                pass

        if img_list == []:
            label = 0.5
        else:
            img_list = torch.stack(img_list)

        return img_list, label, mov_path