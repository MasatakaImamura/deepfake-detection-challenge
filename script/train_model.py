from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, plot_loss
from utils.dfdc_dataset import DeepfakeDataset
from utils.trainer import train_model
from utils.logger import create_logger, get_logger


# 1枚の画像のみをピックアップして学習する

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 64
epoch = 20
lr = 0.001
model_name = 'resnet152'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Use movie number per 1 epoch
# If set "None", all real movies are used
real_mov_num = None
# Label_Smoothing
label_smooth = 0
# Version of Logging
version = model_name + '_000'

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
train_dataset = DeepfakeDataset(
    train_mov_path, metadata, device, transform=ImageTransform(img_size), phase='train', getting_idx=0)

val_dataset = DeepfakeDataset(
    val_mov_path, metadata, device, transform=ImageTransform(img_size), phase='val', getting_idx=0)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
dataloader_dict = {
    'train': train_dataloader,
    'val': val_dataloader
}

print('DataLoader Already')

# Model  ################################################################
torch.cuda.empty_cache()
net = model_init(model_name, classes=1)

# Transfer Learning  ################################################################
# Specify The Layers for updating
# Resnet: fc.weight, fc.bias
# Efficientnet: _fc.weight, _fc.bias
params_to_update = []
update_params_name = ['fc.weight', 'fc.bias']

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

# Setting Optimizer  ################################################################
optimizer = optim.Adam(params=params_to_update, lr=lr)
print('Model Already')

# logging  ################################################################
create_logger(version)
get_logger(version).info('------- Config ------')
get_logger(version).info(f'Random Seed: {seed}')
get_logger(version).info(f'Batch Size: {batch_size}')
get_logger(version).info(f'Loss: {criterion.__class__.__name__}')
get_logger(version).info(f'Optimizer: {optimizer.__class__.__name__}')
get_logger(version).info(f'Learning Rate: {lr}')
get_logger(version).info(f'Update Params: {update_params_name}')
get_logger(version).info('------- Train Start ------')

# Train  ################################################################
net, best_loss, df_loss = train_model(net, dataloader_dict, criterion, optimizer,
                                      num_epoch=epoch, device=device, model_name=model_name,
                                      label_smooth=label_smooth, version=version)

# Save Model  ################################################################
date = datetime.datetime.now().strftime('%Y%m%d')
torch.save(net.state_dict(), "../model/{}_loss{:.3f}_{}.pth".format(model_name, best_loss, date))

# Save Loss
df_loss.to_csv('../loss/LossTable_{}_loss{:.3f}_{}.csv'.format(model_name, best_loss, date))
plot_loss(df_loss, 'LossPlot_{}_loss{:.3f}_{}'.format(model_name, best_loss, date))
