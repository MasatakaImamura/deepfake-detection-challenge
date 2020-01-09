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
from utils.eco import ECO_2D, ECO_3D

# Convolution3Dを使用
# Datasetは連続した画像を出力するように設定

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
batch_size = 8
epoch = 10
model_name = 'eco'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
# Image Num per 1 movie
img_num = 16
# frame number for extracting image from movie
frame_window = 5
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


class ECO_Lite(nn.Module):
    def __init__(self):
        super(ECO_Lite, self).__init__()
        self.eco_2d = ECO_2D()
        self.eco_3d = ECO_3D()
        self.fc_final = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        out = x.view(-1, c, h, w)

        out = self.eco_2d(out)
        out = out.view(-1, ns, 96, 28, 28)

        out = self.eco_3d(out)
        out = self.fc_final(out)

        return out


net = ECO_Lite()

# Setting Optimizer  ################################################################

optimizer = optim.Adam(params=net.parameters())
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
