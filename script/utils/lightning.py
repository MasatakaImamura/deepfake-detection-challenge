import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .utils import get_metadata, get_mov_path
from .dfdc_dataset import DeepfakeDataset_3d_faster

from facenet_pytorch import training, fixed_image_standardization


class LightningSystem(pl.LightningModule):

    def __init__(self, net, dataloader_dict, criterion, optimizer):
        super(LightningSystem, self).__init__()
        self.net = net
        self.dataloader_dict = dataloader_dict
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        # バッチ平均のlossを計算する
        loss = self.criterion(pred, label.unsqueeze(1).float()) / img.size(0)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        loss = self.criterion(pred, label.unsqueeze(1).float()) / img.size(0)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        torch.cuda.empty_cache()

        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        return [self.optimizer]

    @pl.data_loader
    def train_dataloader(self):
        return self.dataloader_dict['train']

    @pl.data_loader
    def val_dataloader(self):
        return self.dataloader_dict['val']


class LightningSystem_2(pl.LightningModule):

    def __init__(self, net, data_dir, device, detector, img_num, img_size,
                 frame_window, batch_size, criterion):

        super(LightningSystem_2, self).__init__()
        self.net = net
        self.device = device
        self.img_num = img_num
        self.batch_size = batch_size

        # Data Loading  ################################################################
        # Set Mov_file path
        metadata = get_metadata(data_dir)
        train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                                    real_mov_num=None, train_size=0.9, seed=0)

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset_3d_faster(
            train_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        self.val_dataset = DeepfakeDataset_3d_faster(
            val_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################
        # 後半のConv3dのみを更新
        params_to_update_1, params_to_update_2, params_to_update_3, params_to_update_4 = [], [], [], []
        update_params_name = ['repeat_2', 'resnet_3d_1', 'resnet_3d_2', 'fc']

        for name, param in net.named_parameters():
            if update_params_name[0] in name:
                param.requires_grad = True
                params_to_update_1.append(param)
            elif update_params_name[1] in name:
                param.requires_grad = True
                params_to_update_2.append(param)
            elif update_params_name[2] in name:
                param.requires_grad = True
                params_to_update_3.append(param)
            elif update_params_name[3] in name:
                param.requires_grad = True
                params_to_update_4.append(param)
            else:
                param.requires_grad = False

        # Optimizer  ################################################################
        self.optimizer = optim.Adam([
            {'params': params_to_update_1, 'lr': 0.001},
            {'params': params_to_update_2, 'lr': 0.005},
            {'params': params_to_update_3, 'lr': 0.01},
            {'params': params_to_update_4, 'lr': 0.05},
        ])

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        # LogLoss
        # バッチ平均のlossを計算する
        loss = self.criterion(pred, label.unsqueeze(1).float()) / img.size(0)

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        acc = torch.sum(pred == label).float() / self.img_num / pred.size(0)

        logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        # LogLoss
        loss_val = self.criterion(pred, label.unsqueeze(1).float()) / img.size(0)

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        val_acc = torch.sum(pred == label).item() / (len(label) * 1.0)

        logs = {'val_loss': loss_val, 'val_acc': val_acc}

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': torch.tensor(val_acc),
            'log': logs
        })

        return output

    def validation_end(self, outputs):

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {'avg_val_loss': val_loss_mean.item(), 'avg_val_acc': val_acc_mean.item()}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'avg_val_loss': val_loss_mean.item(), 'avg_val_acc': val_acc_mean.item()}
        }
        return results

    def configure_optimizers(self):
        return [self.optimizer]


class LightningSystem_2d(pl.LightningModule):
    '''
    2dnetを使用する
    batch_size=1として、連続した画像を2dnetに投入
    '''

    def __init__(self, net, data_dir, device, detector, img_num, img_size,
                 frame_window, batch_size, criterion):

        super(LightningSystem_2d, self).__init__()
        self.net = net
        self.device = device
        self.img_num = img_num
        self.batch_size = batch_size

        assert self.batch_size == 1, 'Set batch_size = 1'

        # Data Loading  ################################################################
        # Set Mov_file path
        metadata = get_metadata(data_dir)
        train_mov_path, val_mov_path = get_mov_path(metadata, data_dir, fake_per_real=1,
                                                    real_mov_num=None, train_size=0.8, seed=0)

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset_3d_faster(
            train_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        self.val_dataset = DeepfakeDataset_3d_faster(
            val_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################

        # Optimizer  ################################################################
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=1e-3)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img.squeeze())
        label = torch.full((self.img_num, 1), label.item()).to(self.device)
        # LogLoss
        # バッチ平均のlossを計算する
        loss = self.criterion(pred, label.float()) / img.size(1)  # img_num

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / self.img_num / pred.size(0)

        logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img.squeeze())
        label = torch.full((self.img_num, 1), label.item()).to(self.device)
        # LogLoss
        # バッチ平均のlossを計算する
        loss_val = self.criterion(pred, label.float()) / img.size(1)  # img_num

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        val_acc = torch.sum(pred == label).item() / (len(label) * 1.0)

        logs = {'val_loss': loss_val, 'val_acc': val_acc}

        output = OrderedDict({
            'val_loss': loss_val,
            'val_acc': torch.tensor(val_acc),
            'log': logs
        })

        return output

    def validation_end(self, outputs):

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {'avg_val_loss': val_loss_mean.item(), 'avg_val_acc': val_acc_mean.item()}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'progress_bar': tqdm_dict,
            'log': {'avg_val_loss': val_loss_mean.item(), 'avg_val_acc': val_acc_mean.item()}
        }
        return results

    def configure_optimizers(self):
        return [self.optimizer]
