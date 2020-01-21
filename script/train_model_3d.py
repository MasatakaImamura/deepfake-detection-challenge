from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from models.model_init import model_init
from models.Conv3D import Conv3dnet, MyNet
from models.eco import ECO_Lite

from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, plot_loss
from utils.dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_3d_faster
from utils.trainer import train_model
from utils.logger import create_logger, get_logger
from utils.lightning import LightningSystem

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
# Label_Smoothing
label_smooth = 0
# Version of Logging
version = model_name + '_000'

# Face Detector
detector = MTCNN(image_size=img_size, margin=14, keep_all=False,
                 select_largest=False, factor=0.5, device=device, post_process=False).eval()

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                            real_mov_num=real_mov_num, train_size=0.9, seed=seed)

# Loss Function  ################################################################
criterion = nn.BCEWithLogitsLoss(reduction='sum')

# Preprocessing  ################################################################
# Dataset
train_dataset = DeepfakeDataset_3d_faster(
    train_mov_path, metadata, device, detector, img_num, img_size, frame_window
)

val_dataset = DeepfakeDataset_3d_faster(
    val_mov_path, metadata, device, detector, img_num, img_size, frame_window
)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}
# Output Shape -> (batch_size, img_num, channel=3, img_size, img_size)

print('DataLoader Already')

# Model  ################################################################
torch.cuda.empty_cache()
net = MyNet(output_size=1)

# Fine Tuning  ###############################################################
# 後半のConv3dのみを更新
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

update_params_name_1 = ['resnet_3d_1']
update_params_name_2 = ['resnet_3d_2']
update_params_name_3 = ['fc']

update_params_name = update_params_name_1 + update_params_name_2 + update_params_name_3

for name, param in net.named_parameters():
    if update_params_name_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
    elif update_params_name_2[0] in name:
        param.requires_grad = True
        params_to_update_2.append(param)
    elif update_params_name_3[0] in name:
        param.requires_grad = True
        params_to_update_3.append(param)
    else:
        param.requires_grad = False

# Setting Optimizer  ################################################################
optimizer = optim.Adam([
    {'params': params_to_update_1, 'lr': 1e-3},
    {'params': params_to_update_2, 'lr': 5e-3},
    {'params': params_to_update_3, 'lr': 1e-2},
])
print('Model Already')

# logging  ################################################################
create_logger(version)
get_logger(version).info('------- Config ------')
get_logger(version).info(f'Random Seed: {seed}')
get_logger(version).info(f'Batch Size: {batch_size}')
get_logger(version).info(f'Loss: {criterion.__class__.__name__}')
get_logger(version).info(f'Optimizer: {optimizer.__class__.__name__}')
get_logger(version).info(f'Learning Rate: {lr}')
get_logger(version).info('------- Train Start ------')

# # Train  ################################################################
# net, best_loss, df_loss = train_model(net, dataloader_dict, criterion, optimizer,
#                                       num_epoch=epoch, device=device, model_name=model_name,
#                                       label_smooth=label_smooth, version=version)
#
# # Save Model  ################################################################
# date = datetime.datetime.now().strftime('%Y%m%d')
# torch.save(net.state_dict(), "../model/{}_loss{:.3f}_{}.pth".format(model_name, best_loss, date))
#
# # Save Loss
# df_loss.to_csv('../loss/LossTable_{}_loss{:.3f}_{}.csv'.format(model_name, best_loss, date))
# plot_loss(df_loss, 'LossPlot_{}_loss{:.3f}_{}'.format(model_name, best_loss, date))

# Pytorch Lightning
# Train  ################################################################
output_path = '../lightning'

model = LightningSystem(net, dataloader_dict, criterion, optimizer)
checkpoint_callback = ModelCheckpoint(filepath='../model', monitor='avg_val_loss',
                                      save_best_only=True, mode='min', save_weights_only=True)
trainer = Trainer(
    max_nb_epochs=epoch,
    default_save_path=output_path,
    checkpoint_callback=checkpoint_callback,
    gpus=[0]
)

trainer.fit(model)

