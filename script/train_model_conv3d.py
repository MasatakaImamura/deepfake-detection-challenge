from sklearn.model_selection import train_test_split
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

from utils.model_init import model_init
from utils.Conv3D import Conv3dnet

from utils.data_augumentation import ImageTransform
from utils.utils import seed_everything, get_metadata, get_mov_path, plot_loss
from utils.dfdc_dataset import DeepfakeDataset, DeepfakeDataset_continuous
from utils.trainer import train_model

# Convolution3Dを使用
# Datasetは連続した画像を出力するように設定
# num_img=10が学習時間ギリギリのライン
# img_sizeは特に影響は受けなさそう

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 100
batch_size = 8
epoch = 10
model_name = 'conv3D'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.BCEWithLogitsLoss()
# Image Num per 1 movie
img_num = 10
# frame number for extracting image from movie
frame_window = 10
# Use movie number per 1 epoch
# If set "None", all real movies are used
real_mov_num = None
# Label_Smoothing
label_smooth = 0.15

# Set Seed
seed_everything(seed)

# Set Mov_file path  ################################################################
metadata = get_metadata(data_dir)
train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=3,
                                            real_mov_num=real_mov_num, train_size=0.9, seed=seed)


print(train_mov_path)

# Preprocessing  ################################################################
# Dataset
train_dataset = DeepfakeDataset_continuous(
    train_mov_path, metadata, device, transform=ImageTransform(img_size),
    phase='train', img_size=img_size, img_num=img_num, frame_window=frame_window)

val_dataset = DeepfakeDataset_continuous(
    val_mov_path, metadata, device, transform=ImageTransform(img_size),
    phase='val', img_size=img_size, img_num=img_num, frame_window=frame_window)

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
net = Conv3dnet(output_size=1)

# Setting Optimizer  ################################################################

optimizer = optim.Adam(params=net.parameters())
print('Model Already')

# Train  ################################################################
net, best_loss, df_loss = train_model(net, dataloader_dict, criterion, optimizer,
                                      num_epoch=epoch, device=device, model_name=model_name,
                                      label_smooth=label_smooth)

# Save Model  ################################################################
date = datetime.datetime.now().strftime('%Y%m%d')
torch.save(net.state_dict(), "../model/{}_loss{:.3f}_{}.pth".format(model_name, best_loss, date))

# Save Loss
df_loss.to_csv('../loss/LossTable_{}_loss{:.3f}_{}.csv'.format(model_name, best_loss, date))
plot_loss(df_loss, 'LossPlot_{}_loss{:.3f}_{}'.format(model_name, best_loss, date))
