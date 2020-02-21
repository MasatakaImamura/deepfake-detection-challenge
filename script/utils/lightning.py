import numpy as np
from sklearn.metrics import log_loss

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import Normalize
from collections import OrderedDict

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from .utils import get_metadata, get_mov_path, freeze_until
from .dfdc_dataset import DeepfakeDataset_3d, DeepfakeDataset_3d_realfake, DeepfakeDataset


class LightningSystem_3d(pl.LightningModule):

    def __init__(self, net, data_dir, device, detector, img_num, img_size,
                 frame_window, batch_size, criterion):

        super(LightningSystem_3d, self).__init__()
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
        self.train_dataset = DeepfakeDataset_3d(
            train_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        self.val_dataset = DeepfakeDataset_3d(
            val_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################
        # Facenet_3d
        freeze_until(self.net, "facenet.repeat_3.0.branch0.conv.weight")

        # Optimizer  ################################################################
        self.optimizer = optim.Adam(params=self.net.parameters())

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
        self.train_dataset = DeepfakeDataset_3d(
            train_mov_path, metadata, device, detector, img_num, img_size, frame_window)

        self.val_dataset = DeepfakeDataset_3d(
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


class LightningSystem_realfake(pl.LightningModule):
    '''
    2dnetを使用する
    Dataset_realfakeを使用し、REAL画像とFAKE画像を同時に学習する
    '''

    def __init__(self, net, data_dir, device, detector, img_num, img_size,
                 frame_window, batch_size, criterion):

        super(LightningSystem_realfake, self).__init__()
        self.net = net
        self.device = device
        self.img_num = img_num
        self.img_size = img_size
        self.batch_size = batch_size

        # Data Loading  ################################################################
        # Set Mov_file path
        metadata = get_metadata(data_dir)

        # Dataset  ################################################################
        self.dataset = DeepfakeDataset_3d_realfake(
            data_dir, metadata, device, detector, img_num, img_size, frame_window)

        # Set Sampler
        img_idx = np.arange(len(self.dataset))
        np.random.shuffle(img_idx)
        self.train_idx = img_idx[:int(0.8 * len(img_idx))]
        self.val_idx = img_idx[int(0.8 * len(img_idx)):]

        # Loss Function  ################################################################
        self.criterion = criterion

        # Fine Tuning  ###############################################################
        # EfficientNet-b4
        freeze_until(self.net, "_blocks.28._expand_conv.weight")

        # Optimizer  ################################################################
        self.optimizer = optim.Adam(params=self.net.parameters(), lr=1e-3)

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.train_idx),
                          pin_memory=True)

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(self.dataset,
                          batch_size=self.batch_size,
                          sampler=SubsetRandomSampler(self.val_idx),
                          pin_memory=True)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        face_r, face_f = batch
        # LogLoss - Real Face
        face_r = face_r.view(-1, 3, self.img_size, self.img_size)
        pred_r = self.forward(face_r)
        label_r = torch.full((pred_r.size(0), 1), 0).float().to(self.device)  # REAL == 0
        loss_r = self.criterion(pred_r, label_r) / pred_r.size(0)

        # LogLoss - Fake Face
        face_f = face_f.view(-1, 3, self.img_size, self.img_size)
        pred_f = self.forward(face_f)
        label_f = torch.full((pred_r.size(0), 1), 1).float().to(self.device)  # FAKE == 1
        loss_f = self.criterion(pred_f, label_f) / pred_r.size(0)

        # LogLoss (REAL Loss + FAKE Loss)
        loss = (loss_r + loss_f) / 2

        # Accuracy - Real Face
        pred_r = torch.sigmoid(pred_r)
        pred_r[pred_r > 0.5] = 1.0
        pred_r[pred_r < 0.5] = 0.0
        acc_r = torch.sum(pred_r == label_r).float() / self.img_num / pred_r.size(0)

        # Accuracy - Fake Face
        pred_f = torch.sigmoid(pred_f)
        pred_f[pred_f > 0.5] = 1.0
        pred_f[pred_f < 0.5] = 0.0
        acc_f = torch.sum(pred_f == label_f).float() / self.img_num / pred_f.size(0)

        acc = (acc_r + acc_f) / 2

        logs = {'train_loss': loss, 'train_acc': acc}

        return {'loss': loss, 'log': logs, 'progress_bar': logs}

    def validation_step(self, batch, batch_idx):
        face_r, face_f = batch
        # LogLoss - Real Face
        face_r = face_r.view(-1, 3, self.img_size, self.img_size)
        pred_r = self.forward(face_r)
        label_r = torch.full((pred_r.size(0), 1), 0).float().to(self.device)  # REAL == 0
        loss_r = self.criterion(pred_r, label_r) / pred_r.size(0)

        # LogLoss - Fake Face
        face_f = face_f.view(-1, 3, self.img_size, self.img_size)
        pred_f = self.forward(face_f)
        label_f = torch.full((pred_r.size(0), 1), 1).float().to(self.device)  # FAKE == 1
        loss_f = self.criterion(pred_f, label_f) / pred_r.size(0)

        # LogLoss (REAL Loss + FAKE Loss)
        val_loss = (loss_r + loss_f) / 2

        # Accuracy - Real Face
        pred_r = torch.sigmoid(pred_r)
        pred_r[pred_r > 0.5] = 1.0
        pred_r[pred_r < 0.5] = 0.0
        acc_r = torch.sum(pred_r == label_r).float() / self.img_num / pred_r.size(0)

        # Accuracy - Fake Face
        pred_f = torch.sigmoid(pred_f)
        pred_f[pred_f > 0.5] = 1.0
        pred_f[pred_f < 0.5] = 0.0
        acc_f = torch.sum(pred_f == label_f).float() / self.img_num / pred_f.size(0)

        val_acc = (acc_r + acc_f) / 2

        logs = {'val_loss': val_loss, 'val_acc': val_acc}

        output = {
            'val_loss': val_loss,
            'val_acc': val_acc,
            'progress_bar': logs
        }

        return output

    def validation_end(self, outputs):

        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()

        tqdm_dict = {'avg_val_loss': val_loss_mean.item(), 'avg_val_acc': val_acc_mean.item()}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': val_acc_mean,
            'progress_bar': tqdm_dict,
            'log': {'avg_val_loss': val_loss_mean, 'avg_val_acc': val_acc_mean}
        }
        return results

    def configure_optimizers(self):
        return [self.optimizer]


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

        # LogLoss
        logloss = 0
        _label = label.cpu().numpy().reshape(-1)
        _pred = pred.cpu().numpy().reshape(-1)
        eps = 1e-10
        _pred = np.clip(_pred, eps, 1 - eps)

        for p, r in zip(_pred, _label):
            if r == 0:
                logloss += np.log(1 - _pred)
            elif r == 1:
                logloss += np.log(_pred)

        logloss = torch.tensor(logloss)

        return {'val_loss': loss, 'val_acc': acc, 'val_logloss': logloss}

    def validation_end(self, outputs):

        avg_val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        avg_val_logloss = torch.stack([x['val_logloss'] for x in outputs]).mean()

        logs_val = {'val/loss': avg_val_loss, 'val/acc': avg_val_acc, 'val/logloss': avg_val_logloss}
        # show val_loss and val_acc in progress bar but only log val_loss
        results = {
            'avg_val_loss': avg_val_loss,
            'log': logs_val
        }
        return results
