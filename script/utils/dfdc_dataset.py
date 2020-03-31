import os, cv2, random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

# OSの違いによるパス分割を定義
if os.name == 'nt':
    sep = '\\'
elif os.name == 'posix':
    sep = '/'


class DeepfakeDataset(Dataset):
    '''
    1画像ごとに出力する
    サンプルサイズを指定
    REALとFAKEの数を均等にできるように
    output_size: (channels=3, width, heights)

    '''

    def __init__(self, faces_img_path, metadata, transform, phase='train', sample_size=None, seed=0):
        print('#'*30)
        print(f'{phase} Dataset Info')
        self.faces_img_path = faces_img_path
        self.metadata = metadata
        self.transform = transform
        self.phase = phase

        if sample_size is not None:
            real_df = self.metadata[self.metadata["label"] == "REAL"]
            fake_df = self.metadata[self.metadata["label"] == "FAKE"]
            sample_size = np.min(np.array([sample_size, len(real_df), len(fake_df)]))
            print("%s: sampling %d from %d real videos" % (phase, sample_size, len(real_df)))
            print("%s: sampling %d from %d fake videos" % (phase, sample_size, len(fake_df)))
            real_df = real_df.sample(sample_size, random_state=seed)
            fake_df = fake_df.sample(sample_size, random_state=seed)
            self.df = pd.concat([real_df, fake_df])

        else:
            self.df = metadata

        num_real = len(self.df[self.df["label"] == "REAL"])
        num_fake = len(self.df[self.df["label"] == "FAKE"])
        print("%s dataset has %d real videos, %d fake videos" % (phase, num_real, num_fake))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["mov"]

        # Extract Specific Mov Name
        target = [c for c in self.faces_img_path if filename in c]
        # Extract Only One Img
        img_path = random.choice(target)
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


class DeepfakeDataset_2(Dataset):
    '''
    1動画ごとに出力する
    サンプルサイズを指定
    REALとFAKEの数を均等にできるように
    output_size: (img_num, channels=3, width, heights)

    '''

    def __init__(self, faces_img_path, metadata, transform, phase='train', sample_size=None, seed=0, img_num=15):
        print('#'*30)
        print(f'{phase} Dataset Info')
        self.faces_img_path = faces_img_path
        self.metadata = metadata
        self.transform = transform
        self.phase = phase
        self.img_num = img_num

        if sample_size is not None:
            real_df = self.metadata[self.metadata["label"] == "REAL"]
            fake_df = self.metadata[self.metadata["label"] == "FAKE"]
            sample_size = np.min(np.array([sample_size, len(real_df), len(fake_df)]))
            print("%s: sampling %d from %d real videos" % (phase, sample_size, len(real_df)))
            print("%s: sampling %d from %d fake videos" % (phase, sample_size, len(fake_df)))
            real_df = real_df.sample(sample_size, random_state=seed)
            fake_df = fake_df.sample(sample_size, random_state=seed)
            self.df = pd.concat([real_df, fake_df])

        else:
            self.df = metadata

        num_real = len(self.df[self.df["label"] == "REAL"])
        num_fake = len(self.df[self.df["label"] == "FAKE"])
        print("%s dataset has %d real videos, %d fake videos" % (phase, num_real, num_fake))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["mov"]

        # Extract Specific Mov Name
        target = [c for c in self.faces_img_path if filename in c]
        img_path = random.choice(target)
        img = [Image.open(t) for t in target]

        # Get Label
        label = img_path.split(sep)[1].split('_')[1]
        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Transform Images
        img = [self.transform(_img, self.phase) for _img in img]

        # Padding (to specified img_num)
        if len(img) < self.img_num:
            c, w, h = img[0].shape
            while True:
                img.append(torch.ones(c, w, h))
                if len(img) == self.img_num:
                    break
        elif len(img) > self.img_num:
            img = img[:self.img_num]

        img = torch.stack(img)

        return img, label
