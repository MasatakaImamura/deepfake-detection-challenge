import matplotlib.pyplot as plt
import cv2
import numpy as np
import dlib

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
frame_window = 5
real_mov_num = None

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################


metadata = get_metadata(data_dir)

mov_path = get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=real_mov_num)


# for i in range(len(mov_path)):
#
#     img = get_img_from_mov(mov_path[i])[0]
#     try:
#         img, prob, points = detect_face_mtcnn(img, device)
#     except:
#         continue
#
#     print(i)
#     print(np.array(points).shape)
#     print('')


img = get_img_from_mov(mov_path[65])[0]
img, probs, points = detect_face_mtcnn(img, device)

import face_recognition

landmark = face_recognition.face_landmarks(img)

print(landmark)

from PIL import Image, ImageDraw
pil_image = Image.fromarray(img)
d = ImageDraw.Draw(pil_image)

for face_landmarks in landmark:

    # Print the location of each facial feature in this image
    for facial_feature in face_landmarks.keys():
        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

    # Let's trace out each facial feature in the image with a line!
    for facial_feature in face_landmarks.keys():
        d.line(face_landmarks[facial_feature], width=3)

# Show the picture
plt.imshow(pil_image)
plt.show()
