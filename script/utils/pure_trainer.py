import time, copy, gc, datetime, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def train_model(dataloaders, net, device, optimizer, scheduler, batch_num,
                num_epochs, exp='test0', saveweightpath='../weights'):

    print('DeepFake Detection Challenge Training Model')
    train_i = 0
    val_i = 0

    net = net.to(device)
    writer = SummaryWriter(f'../tensorboard/{exp}')
    best_loss = 1e+9

    writer.add_text('Config/optimizer', optimizer.__class__.__name__)
    writer.add_text('Config/scheduler', scheduler.__class__.__name__)

    for epoch in range(num_epochs):
        print('#'*30)

        for phase in ['train', 'val']:

            bce_loss = 0
            total_examples = 0

            if phase == 'train':
                net.train()
            else:
                net.eval()

            for i, (img, label) in enumerate(dataloaders[phase]):

                if phase == 'train':
                    optimizer.zero_grad()
                    img = img.to(device)
                    label = label.to(device).float()

                    pred = net(img)
                    pred = pred.squeeze()

                    loss = F.binary_cross_entropy_with_logits(pred, label)

                    loss.backward()
                    optimizer.step()

                else:
                    with torch.no_grad():
                        img = img.to(device)
                        label = label.to(device).float()

                        pred = net(img)
                        pred = pred.squeeze()

                        loss = F.binary_cross_entropy_with_logits(pred, label)

                bce_loss += loss.item() * img.size(0)
                total_examples += img.size(0)

                # Tensorboard
                if phase == 'train':
                    writer.add_scalar('train/batch_loss', loss, train_i)
                    train_i += 1
                else:
                    writer.add_scalar('val/batch_loss', loss, val_i)
                    val_i += 1

                if phase == 'train' and i > batch_num:
                    break

            bce_loss /= total_examples

            # Tensorboard
            writer.add_scalar(f'{phase}/epoch_loss', bce_loss, epoch)
            print(f'Epoch {epoch}  {phase} BCELoss: {bce_loss:.4f}')

            if phase == 'val' and bce_loss < best_loss:
                best_loss = bce_loss
                best_weights = net.state_dict()
                torch.save(best_weights, os.path.join(saveweightpath, f'{exp}_epoch_{epoch}_loss_{bce_loss:.3f}.pth'))

        if scheduler is not None:
            scheduler.step()

    writer.close()






