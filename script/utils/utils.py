import random, os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_metadata(data_dir):
    metadata_path = glob.glob(os.path.join(data_dir, '*.json'))

    # load metadata
    metadata = pd.DataFrame()
    for path in metadata_path:
        metadata = pd.concat([metadata, pd.read_json(path).T], axis=0)

    metadata.reset_index(inplace=True)
    metadata = metadata.rename(columns={'index': 'mov'})

    return metadata


def get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=500, train_size=0.9, seed=0):

    np.random.seed(seed)
    random.seed(seed)
    # 1Real movie 1 fake
    # real_mov_num: Number mov file for use
    fake_list = []
    real_list = metadata[metadata['label'] == 'REAL']['mov'].tolist()
    if real_mov_num is not None:
        real_list = random.sample(real_list, real_mov_num)
    for path in real_list:
        for i in range(fake_per_real):
            try:
                fake_list.append(metadata[metadata['original'] == path]['mov'].tolist()[i])
            except:
                pass

    # Train Test Split
    train_mov_path = []
    val_mov_path = []
    len_val_real = int(len(real_list)*train_size)
    len_val_fake = int(len(fake_list)*train_size)

    # Shuffle List
    real_list = random.sample(real_list, len(real_list))
    fake_list = random.sample(fake_list, len(fake_list))

    train_mov_path.extend(real_list[:len_val_real])
    train_mov_path.extend(fake_list[:len_val_fake])
    val_mov_path.extend(real_list[len_val_real:])
    val_mov_path.extend(fake_list[len_val_fake:])

    train_mov_path = [os.path.join(data_dir, path) for path in train_mov_path]
    val_mov_path = [os.path.join(data_dir, path) for path in val_mov_path]

    # Shuffle List
    train_mov_path = random.sample(train_mov_path, len(train_mov_path))
    val_mov_path = random.sample(val_mov_path, len(val_mov_path))

    return train_mov_path, val_mov_path


def get_img_from_mov(video_file):
    # https://note.nkmk.me/python-opencv-videocapture-file-camera/
    cap = cv2.VideoCapture(video_file)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_list = []
    for i in range(frames):
        _, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)
    cap.release()

    return image_list


def get_img_from_mov_2(video_file, num_img, frame_window):
    # https://note.nkmk.me/python-opencv-videocapture-file-camera/
    cap = cv2.VideoCapture(video_file)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    image_list = []
    for i in range(num_img):
        _, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)
        cap.set(cv2.CAP_PROP_POS_FRAMES, (i + 1) * frame_window)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
            break
    cap.release()

    return image_list


def detect_face(img, cascade_path):
    # Add Dataset "Haarcascades"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    face_crops = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

    if len(face_crops) == 0:
        return []

    crop_imgs = []
    for i in range(len(face_crops)):
        x = face_crops[i][0]
        y = face_crops[i][1]
        w = face_crops[i][2]
        h = face_crops[i][3]
        # x,y,w,h=ratio*x,ratio*y,ratio*w,ratio*h
        crop_imgs.append(img[y:y + h, x:x + w])
    return crop_imgs


def detect_face_mtcnn(img, device):
    _img = img[np.newaxis, :, :, :]
    mtcnn = MTCNN(keep_all=True, device=device, margin=14, factor=0.5).eval()
    boxes, probs, points = mtcnn.detect(_img, landmarks=True)
    # pointsは「nose, mouth_right, right_eye, left_eye, mouse_left」の(x, y)を表現したランドマーク

    if len(boxes) == 0:
        raise ValueError('Error!')

    x = int(boxes[0][0][0])
    y = int(boxes[0][0][1])
    z = int(boxes[0][0][2])
    w = int(boxes[0][0][3])
    crop_img = img[y:w, x:z]
    return crop_img


def plot_loss(df_loss, figname):
    plt.plot(df_loss['Train_loss'], label='Train')
    plt.plot(df_loss['Val_loss'], label='Val')
    plt.legend()
    plt.savefig('../loss/{}.png'.format(figname))
