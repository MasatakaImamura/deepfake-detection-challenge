import matplotlib.pyplot as plt
import cv2, os, random
import numpy as np
import time
from PIL import Image

from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision.transforms import Normalize

from models.model_init import model_init, convLSTM, convLSTM_resnet
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, detect_face, detect_face_mtcnn, get_img_from_mov, freeze_until
from utils.dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_2d, DeepfakeDataset_3d_realfake
# from utils.trainer import train_model
from facenet_pytorch import InceptionResnetV1, MTCNN
from models.mesonet import Meso4, MesoInception4
from utils.logger import create_logger, get_logger

from models.Facenet_3d import Facenet_3d

from efficientnet_pytorch import EfficientNet

from models.blazeface import BlazeFace

import pandas as pd
import glob


# Config  ################################################################
data_dir = ['../input']
output_dir_face = '../data/faces'
output_dir_meta = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_num = 30
window_size = 10

detector = MTCNN(image_size=224, margin=14, keep_all=False,
                 select_largest=False, factor=0.5, device=device, post_process=False).eval()

# Get Metadata  ################################################################
metadata = pd.DataFrame()

for d in data_dir:
    meta = get_metadata(d)
    metadata = pd.concat([metadata, meta], axis=0, ignore_index=True)

metadata.to_csv(os.path.join(output_dir_meta, 'meta.csv'))

all_mov_path = glob.glob('../input*/*.mp4')


for i in range(2):

    target = metadata.iloc[i:i+1]
    mov = target['mov'].values[0]
    label = target['label'].values[0]

    mov_path = [p for p in all_mov_path if mov in p]

    cap = cv2.VideoCapture(mov_path[0])

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(img_num):
        _, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        _image = image[np.newaxis, :, :, :]
        boxes, probs = detector.detect(_image, landmarks=False)

        x = int(boxes[0][0][0])
        y = int(boxes[0][0][1])
        z = int(boxes[0][0][2])
        w = int(boxes[0][0][3])
        image = image[y:w, x:z]

        image = cv2.resize(image, (224, 224))

        filename = f'{mov}_{label}_frame_{i * window_size}.jpg'

        # cv2.imwrite(os.path.join(output_dir_face, filename), image, [cv2.IMWRITE_JPEG_QUALITY, 100])

        image = Image.fromarray(image)
        image.save(os.path.join(output_dir_face, filename))

        cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * window_size)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
            break
    cap.release()

    # os.remove(mov_path)



