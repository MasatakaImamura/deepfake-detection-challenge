import os, glob
import pandas as pd

import torch
import torch.nn as nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.trainer import Trainer

from utils.lightning import DFDCLightningSystem
from models.Efficientnet_3d import Efficientnet_3d
from utils.data_augumentation import GroupImageTransform

# Config  ################################################################
faces_dir = '../data/faces_temp'
meta_dir = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
img_num = 15
batch_size = 4
img_size = 224
epoch = 100

# Load Data  ##################################################################
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
metadata = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

# ImageTransform  ##################################################################
transform = GroupImageTransform(size=img_size)

# Model  ##################################################################
net = Efficientnet_3d(output_size=1)

# Pytorch Lightning
# Train  ##################################################################
output_path = '../lightning'

model = DFDCLightningSystem(faces, metadata, net, device, transform, criterion, img_num, img_size, batch_size)

checkpoint_callback = ModelCheckpoint(filepath='../lightning/ckpt', monitor='avg_val_loss',
                                      mode='min', save_weights_only=True)
earlystopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10)

trainer = Trainer(
    max_epochs=epoch,
    min_epochs=5,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=earlystopping_callback,
    overfit_pct=0.1,
    gpus=[0],
)

trainer.fit(model)
