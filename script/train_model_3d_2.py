from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.model_init import model_init
from models.Conv3D import Efficientnet_3d, Facenet_3d
from models.eco import ECO_Lite

from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, plot_loss
from utils.dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_3d_faster
from utils.trainer import train_model
from utils.logger import create_logger, get_logger
from utils.lightning import LightningSystem, LightningSystem_2

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from facenet_pytorch import InceptionResnetV1, MTCNN

# Convolution3Dを使用
# Datasetは連続した画像を出力するように設定

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 8
epoch = 20
lr = 0.001
model_name = 'mynet'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Image Num per 1 movie
img_num = 14
# frame number for extracting image from movie
frame_window = 20
# Use movie number per 1 epoch
# If set "None", all real movies are used
real_mov_num = None
# Version of Logging
version = model_name + '_000'

# Face Detector
detector = MTCNN(image_size=img_size, margin=14, keep_all=False,
                 select_largest=False, factor=0.5, device=device, post_process=False).eval()

# Set Seed
seed_everything(seed)

# Loss Function  ################################################################
criterion = nn.BCEWithLogitsLoss(reduction='sum')

# Model  ################################################################
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
net = Facenet_3d(output_size=1)

# Pytorch Lightning
# Train  ################################################################
output_path = '../lightning'

model = LightningSystem_2(net, data_dir, device, detector, img_num, img_size,
                          frame_window, batch_size, criterion)

checkpoint_callback = ModelCheckpoint(filepath='../lightning/ckpt', monitor='val_loss',
                                      save_best_only=True, mode='min', save_weights_only=True)
trainer = Trainer(
    max_nb_epochs=epoch,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    # gpus=[0]
)

trainer.fit(model)

