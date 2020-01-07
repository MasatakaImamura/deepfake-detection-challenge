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
from utils.trainer import train_model, train_model_by_generator


# 1動画を複数の画像に分割する
# generatorを使用して画像ごとにモデルに通す
# 1動画分のlossを計算して、動画ごとにパラメータの調整を行う

# Config  ################################################################
data_dir = '../input'
seed = 0
img_size = 224
frame_window = 10  # 学習する画像のフレーム間隔
epoch = 30
model_name = 'efficientnet-b0'
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

train_mov_path = train_mov_path[:10]

# Model  ################################################################
torch.cuda.empty_cache()
net = model_init(model_name, classes=2)

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
net = train_model_by_generator(net, metadata, train_mov_path, ImageTransform(resize=img_size),
                               val_mov_path, criterion, optimizer,
                               num_epoch=epoch, device=device, frame_window=10, max_frame=100)

# Save Model  ################################################################
date = datetime.datetime.now().strftime('%Y%m%d')
torch.save(net.state_dict(), "../model/{}_{}.pth".format(model_name, date))
