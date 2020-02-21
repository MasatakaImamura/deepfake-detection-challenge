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


# 動画から1枚だけの画像を出力する
# getting_idxで動画のフレームから任意の位置の画像をピックアップする
# Output Shape -> (channel=3, img_size, img_size)
class DeepfakeDataset_2d(Dataset):
    '''
    動画から1枚だけの画像を出力する
    getting_idxで動画のフレームから任意の位置の画像をピックアップする
    Output Shape -> (channel=3, img_size, img_size)
    '''

    def __init__(self, file_list, metadata, device, detector, img_size, getting_idx=0, img_num=14, frame_window=20):
        self.file_list = file_list
        self.metadata = metadata
        self.device = device
        self.detector = detector
        self.img_size = img_size
        self.getting_idx = getting_idx
        self.img_num = img_num
        self.frame_window = frame_window

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
            image = get_img_from_mov(mov_path, self.img_num, self.frame_window)[self.getting_idx]  # Specified Frame Only
            # FaceCrop
            face = self.detector(Image.fromarray(image))
            # 検出できなかった場合
            if face is None:
                face = torch.randn(3, self.img_size, self.img_size)
        except:
            face = torch.randn((3, self.img_size, self.img_size), dtype=torch.float)
            label = 0.5

        return face, label, mov_path


class DeepfakeDataset_3d(Dataset):
    '''
    detectorを指定する
    1動画ごとの連続画像を生成
    img_numで画像の最大枚数  frame_windowで動画の間隔を指定
    Output: (img_num, channels=3, img_size, img_size)
    '''

    def __init__(self, file_list, metadata, device, detector, img_num=20, img_size=224, frame_window=10):
        self.file_list = file_list
        self.metadata = metadata
        self.device = device
        self.detector = detector
        self.img_num = img_num
        self.img_size = img_size
        self.frame_window = frame_window

        # 前処理は別に行う
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        assert self.detector.post_process is False, 'Set Detector post_process=False'

        self.normalize = Normalize(mean, std)

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
        all_image = get_img_from_mov(mov_path, self.img_num, self.frame_window)
        for i in range(len(all_image)):
            img_list.append(Image.fromarray(all_image[i]))

        # まとめて顔抽出
        img_list = self.detector(img_list)

        # Preprocessing
        img_list = [c / 255.0 for c in img_list if c is not None]
        while True:
            if len(img_list) != self.img_num:
                img_list.append(torch.randn(3, self.img_size, self.img_size).abs())
            else:
                break

        img_list = [self.normalize(face) for face in img_list]

        # Noneを埋める
        img_list = [c for c in img_list if c is not None]
        while True:
            if len(img_list) != self.img_num:
                img_list.append(torch.randn(3, self.img_size, self.img_size))
            else:
                break
        # Stack
        img_list = torch.stack(img_list)

        return img_list, label, mov_path


class DeepfakeDataset_3d_realfake(Dataset):
    '''
    DeepfakeDataset_3dの拡張版
    real画像とfake画像を出力するように設定
    MTCNNのpostprocessは使わず、Normalizeを実行
    Output: (img_num, channels=3, img_size, img_size)
    '''

    def __init__(self, data_dir, metadata, device, detector, img_num=14, img_size=224, frame_window=20):
        self.data_dir = data_dir
        self.metadata = metadata
        self.device = device
        self.detector = detector
        self.img_num = img_num
        self.img_size = img_size
        self.frame_window = frame_window

        # 前処理は別に行う
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        assert self.detector.post_process is False, 'Set Detector post_process=False'

        self.normalize = Normalize(mean, std)

        # Real_mov_path
        self.real_mov_list = metadata[metadata['label'] == 'REAL']['mov'].values

        # {Real_movpath_name: fake_movpath_name_list}
        self.all_mov_dict = {}
        for mov_name in self.real_mov_list:
            fk = metadata[metadata['original'] == mov_name]['mov'].tolist()
            self.all_mov_dict.update({mov_name: fk})

    def __len__(self):
        return len(self.real_mov_list)

    def __getitem__(self, idx):
        # Get movpath  ###############################################
        real_mov = self.real_mov_list[idx]

        # Mov -> Image -> Face (Real)  ###############################################
        # Get Real_mov path
        real_mov_path = os.path.join(self.data_dir, real_mov)

        # Mov2Image
        cap_real = cv2.VideoCapture(real_mov_path)
        frames = int(cap_real.get(cv2.CAP_PROP_FRAME_COUNT))
        real_image_list = []

        for i in range(self.img_num):
            try:
                _, image_real = cap_real.read()

                image_real = cv2.cvtColor(image_real, cv2.COLOR_BGR2RGB)
                real_image_list.append(Image.fromarray(image_real))
                cap_real.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * self.frame_window)
                if cap_real.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
                    break
            except:
                pass
        cap_real.release()

        # Detect Faces
        face_r = self.detector(real_image_list)

        # Preprocessing
        face_r = [c / 255.0 for c in face_r if c is not None]
        while True:
            if len(face_r) != self.img_num:
                face_r.append(torch.randn(3, self.img_size, self.img_size).abs())
            else:
                break

        face_r = [self.normalize(face) for face in face_r]
        face_r = torch.stack(face_r)  # (img_num, channel, img_size, img_size)

        # Mov -> Image -> Face (Fake)  ###############################################
        # Get Fake_mov path
        fake_mov_list = self.all_mov_dict[real_mov]
        face_idx = random.choice(np.arange(len(fake_mov_list)))  # RandomChoice
        fake_mov = fake_mov_list[face_idx]
        fake_mov_path = os.path.join(self.data_dir, fake_mov)

        # Mov2Image
        cap_fake = cv2.VideoCapture(fake_mov_path)
        frames = int(cap_fake.get(cv2.CAP_PROP_FRAME_COUNT))
        fake_image_list = []
        for i in range(self.img_num):
            try:
                _, image_fake = cap_fake.read()

                image_fake = cv2.cvtColor(image_fake, cv2.COLOR_BGR2RGB)
                fake_image_list.append(Image.fromarray(image_fake))
                cap_fake.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * self.frame_window)
                if cap_fake.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
                    break
            except:
                pass
        cap_fake.release()

        # Detect Faces
        face_f = self.detector(fake_image_list)

        # Preprocessing
        face_f = [c / 255.0 for c in face_f if c is not None]
        while True:
            if len(face_f) != self.img_num:
                face_f.append(torch.randn(3, self.img_size, self.img_size))
            else:
                break

        face_f = [self.normalize(face) for face in face_f]
        face_f = torch.stack(face_f)  # (img_num, channel, img_size, img_size)

        return face_r, face_f



class DeepfakeDataset(Dataset):

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
