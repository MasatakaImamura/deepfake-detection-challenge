import cv2, os
import numpy as np
from PIL import Image
import torch
from utils.utils import get_metadata
from facenet_pytorch import InceptionResnetV1, MTCNN

import pandas as pd
import glob
from tqdm import tqdm


# Config  ################################################################
data_dir = ['../input', '../input_2']
output_dir_face = '../data/faces_temp'
output_dir_meta = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
img_num = 15
window_size = 20

detector = MTCNN(image_size=224, margin=14, keep_all=False,
                 select_largest=False, factor=0.5, device=device, post_process=False).eval()

# Get Metadata  ################################################################
metadata = pd.DataFrame()

for d in data_dir:
    meta = get_metadata(d)
    metadata = pd.concat([metadata, meta], axis=0, ignore_index=True)

metadata.to_csv(os.path.join(output_dir_meta, 'meta_0_31.csv'))

# 動画全ファイルのパスを取得
all_mov_path = glob.glob('../input*/*.mp4')


for i in tqdm(range(len(metadata))):
    # 一行ずつ処理
    target = metadata.iloc[i:i+1]
    # File名
    mov = target['mov'].values[0]
    # REAL or FAKE
    label = target['label'].values[0]
    # フルパスを取得
    mov_path = [p for p in all_mov_path if mov in p]

    # VideoCapture
    try:
        cap = cv2.VideoCapture(mov_path[0])
    except:
        continue

    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for j in range(img_num):
        try:
            _, image = cap.read()
            #
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Face Crop
            _image = image[np.newaxis, :, :, :]
            boxes, probs = detector.detect(_image, landmarks=False)
            x = int(boxes[0][0][0])
            y = int(boxes[0][0][1])
            z = int(boxes[0][0][2])
            w = int(boxes[0][0][3])
            image = image[y-15:w+15, x-15:z+15]
            # Resize
            image = cv2.resize(image, (224, 224))

            # Save Image
            filename = f'{mov}_{label}_frame_{j * window_size}.jpg'
            image = Image.fromarray(image)
            image.save(os.path.join(output_dir_face, filename))
        except:
            pass

        cap.set(cv2.CAP_PROP_POS_FRAMES, (j + 1) * window_size)
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= frames:
            break
    cap.release()

    # 抽出したファイルを削除
    os.remove(mov_path[0])



