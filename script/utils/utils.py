import random, os, glob
import numpy as np
import pandas as pd

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


def get_mov_path(metadata, data_dir, fake_per_real=1):
    # 1Real movie 1 fake
    mov_path = []
    real_list = metadata[metadata['label'] == 'REAL']['mov'].tolist()
    for path in real_list:
        for i in range(fake_per_real):
            try:
                mov_path.append(metadata[metadata['original'] == path]['mov'].tolist()[i])
            except:
                pass

    mov_path.extend(real_list)
    mov_path = [os.path.join(data_dir, path) for path in mov_path]

    return mov_path


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


def detect_face(img):
    # Add Dataset "Haarcascades"
    face_cascade = cv2.CascadeClassifier('../../haarcascade/haarcascade_frontalface_alt.xml')
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
    mtcnn = MTCNN(keep_all=True, device=device).eval()
    boxes, probs, points = mtcnn.detect(_img, landmarks=True)

    if len(boxes) == 0:
        raise ValueError('Error!')

    x = int(boxes[0][0][0])
    y = int(boxes[0][0][1])
    z = int(boxes[0][0][2])
    w = int(boxes[0][0][3])
    crop_img = img[y:w, x:z]
    return crop_img

