import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .utils import get_metadata, get_mov_path
from .dfdc_dataset import DeepfakeDataset_3d_faster


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

        loss = self.criterion(pred, label.unsqueeze(1).float())
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        loss = self.criterion(pred, label.unsqueeze(1).float())

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
        params_to_update_1, params_to_update_2, params_to_update_3 = [], [], []
        update_params_name = ['resnet_3d_1', 'resnet_3d_2', 'fc']

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
            else:
                param.requires_grad = False

        # Optimizer  ################################################################
        self.optimizer = optim.Adam([
                                    {'params': params_to_update_1, 'lr': 1e-3},
                                    {'params': params_to_update_2, 'lr': 5e-3},
                                    {'params': params_to_update_3, 'lr': 1e-2},
        ])

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=True,
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        # LogLoss
        loss = self.criterion(pred, label.unsqueeze(1).float())

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        acc = torch.sum(pred == label) / pred.size()[0]

        logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        pred = self.forward(img)

        # LogLoss
        loss = self.criterion(pred, label.unsqueeze(1).float())

        # Accuracy
        pred = torch.sigmoid(pred)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        acc = torch.sum(pred == label) / pred.size()[0]

        logs = {'val_loss': loss, 'val_acc': acc}

        return {'val_loss': loss, 'val_acc': acc, 'log': logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        logs = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_acc}
        torch.cuda.empty_cache()

        return {'avg_val_loss': avg_loss, 'log': logs}

    def configure_optimizers(self):
        return [self.optimizer]

