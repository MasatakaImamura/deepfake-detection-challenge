from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init
from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path
from utils.dfdc_dataset import DeepfakeDataset_idx0, DeepfakeDataset_continuous
from utils.trainer import train_model


# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 4
epoch = 6
model_name = 'resnet50'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
mov_path = get_mov_path(metadata, data_dir, fake_per_real=1)

# Preprocessing  ################################################################
# Divide Train, Vaild Data
train_mov_path, val_mov_path = train_test_split(mov_path, test_size=0.1, random_state=seed)

# Dataset
train_dataset = DeepfakeDataset_continuous(train_mov_path, metadata, transform=ImageTransform(img_size), phase='train')
val_dataset = DeepfakeDataset_continuous(val_mov_path, metadata, transform=ImageTransform(img_size), phase='val')

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
net = model_init(model_name)

# Transfer Learning  ################################################################
# Specify The Layers for updating
params_to_update = []
update_params_name = ['fc.weight', 'fc.bias']

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
print('Model Already')

# Train  ################################################################
net, best_loss = train_model(net, dataloader_dict, criterion, optimizer, num_epoch=epoch, device=device)

# Save Model  ################################################################
date = datetime.datetime.now().strftime('%Y%m%d')
torch.save(net.state_dict(), "../model/{}_acc{:.3f}_{}.pth".format(model_name, best_loss, date))
