import torch
from torch import nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer


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
        logs = {'log': loss}

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
