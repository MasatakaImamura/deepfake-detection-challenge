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


model = model_init('resnet50')

z = torch.randn(1, 3, 224, 224)

out = model(z)

label = torch.ones(1, 1)

loss = nn.BCEWithLogitsLoss()

losses = loss(out, label)

print(out)
print(torch.softmax(out))
print(label)
print(losses)
