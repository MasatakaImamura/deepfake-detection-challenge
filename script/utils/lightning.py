import numpy as np
from sklearn.metrics import log_loss

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Normalize
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .utils import get_metadata, get_mov_path, freeze_until
from .dfdc_dataset import DeepfakeDataset, DeepfakeDataset_per_img
from .centerloss import CenterLoss


class DFDCLightningSystem(pl.LightningModule):

    def __init__(self, faces, metadata, net, device, transform, criterion, optimizer,
                 scheduler, img_num, img_size, batch_size):
        super(DFDCLightningSystem, self).__init__()
        self.faces = faces
        self.metadata = metadata
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.img_num = img_num
        self.img_size = img_size
        self.batch_size = batch_size

        # Data Loading  ################################################################

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset(
            faces, metadata, transform, phase='train', img_size=self.img_size, img_num=self.img_num
        )

        self.val_dataset = DeepfakeDataset(
            faces, metadata, transform, phase='val', img_size=self.img_size, img_num=self.img_num
        )

        # Set Sampler
        img_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(img_idx)
        self.train_idx = img_idx[:int(0.8 * len(img_idx))]
        self.val_idx = img_idx[int(0.8 * len(img_idx)):]

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################
        # EfficientNet-b4
        # freeze_until(self.net, "_blocks.28._expand_conv.weight")

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.train_idx),
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.val_idx),
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        inp, label = batch
        label = label.unsqueeze(1).float()

        pred = self.forward(inp)

        loss = self.criterion(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        logs = {'train/loss': loss, 'train/acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        inp, label = batch
        label = label.unsqueeze(1).float()

        pred = self.forward(inp)

        loss = self.criterion(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        logs_val = {'val/loss': avg_val_loss, 'val/acc': avg_val_acc}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': avg_val_loss,
            'log': logs_val
        }
        return results


class DFDCLightningSystem_centerloss(pl.LightningModule):

    def __init__(self, faces, metadata, net, device, transform, criterion, optimizer,
                 scheduler, img_num, img_size, batch_size):
        super(DFDCLightningSystem_centerloss, self).__init__()
        self.faces = faces
        self.metadata = metadata
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.img_num = img_num
        self.img_size = img_size
        self.batch_size = batch_size

        self.centerloss = CenterLoss(1, 2).to(device)
        self.optimizer4center = optim.SGD(self.centerloss.parameters(), lr=0.5)

        # Data Loading  ################################################################

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset(
            faces, metadata, transform, phase='train', img_size=self.img_size, img_num=self.img_num
        )

        self.val_dataset = DeepfakeDataset(
            faces, metadata, transform, phase='val', img_size=self.img_size, img_num=self.img_num
        )

        # Set Sampler
        img_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(img_idx)
        self.train_idx = img_idx[:int(0.8 * len(img_idx))]
        self.val_idx = img_idx[int(0.8 * len(img_idx)):]

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################
        # EfficientNet-b4
        # freeze_until(self.net, "_blocks.28._expand_conv.weight")

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.train_idx),
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.val_idx),
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return [self.optimizer, self.optimizer4center], [self.scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        inp, label = batch
        _label = label.detach().unsqueeze(1).float()

        pred, lp1 = self.forward(inp)

        xent_loss = self.criterion(pred, _label)
        cent_loss = self.centerloss(lp1, label)

        loss = xent_loss + cent_loss

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == _label).float() / pred.size(0)

        logs = {'train/loss': xent_loss, 'train/acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        inp, label = batch
        _label = label.detach().unsqueeze(1).float()
        # label = label.unsqueeze(1).float()

        pred, lp1 = self.forward(inp)

        xent_loss = self.criterion(pred, _label)
        cent_loss = self.centerloss(lp1, label)

        loss = xent_loss + cent_loss

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == _label).float() / pred.size(0)

        return {'val_loss': loss, 'val_acc': acc, 'val_xent_loss': xent_loss}

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_xent_loss = torch.stack([x['val_xent_loss'] for x in outputs]).mean()

        logs_val = {'val/loss': avg_val_xent_loss, 'val/acc': avg_val_acc}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': avg_val_loss,
            'log': logs_val
        }
        return results


class DFDCLightningSystem_2d(pl.LightningModule):

    def __init__(self, faces, metadata, net, device, transform, criterion, optimizer,
                 scheduler, batch_size):
        super(DFDCLightningSystem_2d, self).__init__()
        self.faces = faces
        self.metadata = metadata
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size

        # Data Loading  ################################################################

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset_per_img(
            faces, metadata, transform, phase='train'
        )

        self.val_dataset = DeepfakeDataset_per_img(
            faces, metadata, transform, phase='val'
        )

        # Set Sampler
        img_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(img_idx)
        self.train_idx = img_idx[:int(0.8 * len(img_idx))]
        self.val_idx = img_idx[int(0.8 * len(img_idx)):]

        # Loss Function  ################################################################
        self.criterion = criterion

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.train_idx),
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.val_idx),
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        inp, label = batch
        label = label.unsqueeze(1).float()

        pred = self.forward(inp)

        loss = self.criterion(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        logs = {'train/loss': loss, 'train/acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        inp, label = batch
        label = label.unsqueeze(1).float()

        pred = self.forward(inp)

        loss = self.criterion(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        logs_val = {'val/loss': avg_val_loss, 'val/acc': avg_val_acc}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': avg_val_loss,
            'log': logs_val
        }
        return results


class DFDCLightningSystem_2d_2(pl.LightningModule):

    def __init__(self, faces, metadata, net, device, transform, criterion, optimizer,
                 scheduler, batch_size):
        super(DFDCLightningSystem_2d_2, self).__init__()
        self.faces = faces
        self.metadata = metadata
        self.net = net
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size

        # Data Loading  ################################################################

        # Dataset  ################################################################
        self.train_dataset = DeepfakeDataset_per_img(
            faces, metadata, transform, phase='train'
        )

        self.val_dataset = DeepfakeDataset_per_img(
            faces, metadata, transform, phase='val'
        )

        # Set Sampler
        img_idx = np.arange(len(self.train_dataset))
        np.random.shuffle(img_idx)
        self.train_idx = img_idx[:int(0.8 * len(img_idx))]
        self.val_idx = img_idx[int(0.8 * len(img_idx)):]

        # Loss Function  ################################################################
        self.criterion = criterion

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.train_idx),
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.val_idx),
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch, batch_idx):
        inp, label = batch
        label = label.float()
        pred = self.forward(inp)
        pred = pred.squeeze()

        loss = F.binary_cross_entropy_with_logits(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        logs = {'train/loss': loss, 'train/acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        inp, label = batch
        label = label.unsqueeze(1).float()

        pred = self.forward(inp)

        loss = self.criterion(pred, label)

        # Accuracy
        pred = torch.sigmoid(pred.detach())
        pred[pred > 0.5] = 1.0
        pred[pred < 0.5] = 0.0
        acc = torch.sum(pred == label).float() / pred.size(0)

        return {'val_loss': loss, 'val_acc': acc}

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        logs_val = {'val/loss': avg_val_loss, 'val/acc': avg_val_acc}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': avg_val_loss,
            'log': logs_val
        }
        return results