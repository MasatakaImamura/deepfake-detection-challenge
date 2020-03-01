import os, glob
import pandas as pd

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.trainer import Trainer

from utils.lightning import DFDCLightningSystem_2d
from models.Efficientnet_3d import Efficientnet_2d
from models.Facenet_3d import Facenet_3d
from utils.data_augumentation import ImageTransform, ImageTransform_2
from utils.utils import freeze_until
from utils.radam import RAdam

# Config  ################################################################
faces_dir = '../data/faces_temp'
meta_dir = '../data/meta'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
img_num = 15
batch_size = 4
img_size = 120
epoch = 100

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Load Data  ##################################################################
faces = glob.glob(os.path.join(faces_dir, '*.jpg'))
metadata = pd.read_csv(os.path.join(meta_dir, 'meta.csv'))

# ImageTransform  ##################################################################
transform = ImageTransform_2(size=img_size, mean=mean, std=std)

# Model  ##################################################################
net = Efficientnet_2d(output_size=1, model_name='efficientnet-b0')

# Optimizer  ################################################################
optimizer = RAdam(params=net.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# Pytorch Lightning
# Train  ##################################################################
output_path = '../lightning'

model = DFDCLightningSystem_2d(faces, metadata, net, device, transform, criterion,
                            optimizer, scheduler, batch_size)

checkpoint_callback = ModelCheckpoint(filepath='../lightning/ckpt', monitor='avg_val_loss',
                                      mode='min', save_weights_only=True)

earlystopping_callback = EarlyStopping(monitor='avg_val_loss', patience=20, mode='min')

trainer = Trainer(
    max_epochs=epoch,
    min_epochs=10,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    early_stop_callback=earlystopping_callback,
    overfit_pct=0.02,
    gpus=[0],
)

trainer.fit(model)
