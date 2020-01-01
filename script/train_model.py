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
from utils.trainer import train_model, train_model_2


# 1枚の画像のみをピックアップして学習する

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 64
epoch = 30
model_name = 'efficientnet-b7'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# Use movie number per 1 epoch
# If set "None", all real movies are used
real_mov_num = None

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
mov_path = get_mov_path(metadata, data_dir, fake_per_real=1, real_mov_num=real_mov_num)

# Preprocessing  ################################################################
# Divide Train, Vaild Data
train_mov_path, val_mov_path = train_test_split(mov_path, test_size=0.1, random_state=seed)

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
net = model_init(model_name, classes=2)
# net = convLSTM(out_classes=2)
# net = convLSTM_resnet()

# Transfer Learning  ################################################################
# Specify The Layers for updating
# Resnet: fc.weight, fc.bias
# Efficientnet: _fc.weight, _fc.bias
params_to_update = []
update_params_name = ['_fc.weight', '_fc.bias']

for name, param in net.named_parameters():
    if name in update_params_name:
        param.requires_grad = True
        params_to_update.append(param)
    else:
        param.requires_grad = False

optimizer = optim.Adam(params=params_to_update)
print('Model Already')

# Train  ################################################################
net, best_loss, df_loss = train_model(net, dataloader_dict, criterion, optimizer,
                                      num_epoch=epoch, device=device, model_name=model_name)

# Save Model  ################################################################
date = datetime.datetime.now().strftime('%Y%m%d')
torch.save(net.state_dict(), "../model/{}_loss{:.3f}_{}.pth".format(model_name, best_loss, date))

# Save Loss
df_loss.to_csv('../loss/LossTable_{}_loss{:.3f}_{}.csv'.format(model_name, best_loss, date))
plot_loss(df_loss, 'LossPlot_{}_loss{:.3f}_{}'.format(model_name, best_loss, date))
