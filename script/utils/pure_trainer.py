import time, copy, gc, datetime, os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from .centerloss import CenterLoss


def train_model(dataloaders, net, device, optimizer, scheduler, batch_num,
                num_epochs, exp='test0', saveweightpath='../weights'):

    print('DeepFake Detection Challenge Training Model')
    train_i, val_i = 0, 0
    net = net.to(device)
    writer = SummaryWriter(f'../tensorboard/{exp}')
    best_loss = 1e+9

    writer.add_text('Config/optimizer', optimizer.__class__.__name__)
    writer.add_text('Config/scheduler', scheduler.__class__.__name__)

    for epoch in range(num_epochs):
        print('#'*30)

        for phase in ['train', 'val']:

            bce_loss = 0
            accuracy = 0
            total_examples = 0

            if phase == 'train':
                net.train()
            else:
                net.eval()

            for i, (img, label) in enumerate(dataloaders[phase]):
                batch_size = img.size(0)

                # Training
                if phase == 'train':
                    optimizer.zero_grad()
                    img = img.to(device)
                    label = label.to(device).float()

                    pred = net(img)
                    pred = pred.squeeze()

                    # Binary Cross Entropy
                    loss = F.binary_cross_entropy_with_logits(pred, label)
                    loss.backward()
                    optimizer.step()

                    # Accuracy
                    pred = torch.sigmoid(pred.detach())
                    pred[pred > 0.5] = 1.0
                    pred[pred < 0.5] = 0.0
                    acc = torch.sum(pred == label).float() / pred.size(0)

                # Validation
                else:
                    with torch.no_grad():
                        img = img.to(device)
                        label = label.to(device).float()

                        pred = net(img)
                        pred = pred.squeeze()

                        loss = F.binary_cross_entropy_with_logits(pred, label)

                        # Accuracy
                        pred = torch.sigmoid(pred.detach())
                        pred[pred > 0.5] = 1.0
                        pred[pred < 0.5] = 0.0
                        acc = torch.sum(pred == label).float() / pred.size(0)

                bce_loss += loss.item() * batch_size
                accuracy += acc.item() * batch_size
                total_examples += batch_size

                # Tensorboard
                if phase == 'train':
                    writer.add_scalar('train/batch_loss', loss, train_i)
                    writer.add_scalar('train/batch_acc', acc, train_i)
                    train_i += 1
                else:
                    writer.add_scalar('val/batch_loss', loss, val_i)
                    writer.add_scalar('val/batch_acc', acc, val_i)
                    val_i += 1

                if phase == 'train' and i > batch_num:
                    break

            bce_loss /= total_examples
            accuracy /= total_examples

            # Tensorboard
            writer.add_scalar(f'{phase}/epoch_loss', bce_loss, epoch)
            writer.add_scalar(f'{phase}/epoch_acc', accuracy, epoch)
            print(f'Epoch {epoch}  {phase} BCELoss: {bce_loss:.4f} Accuracy: {accuracy:.4f}')

            if phase == 'val' and bce_loss < best_loss:
                best_loss = bce_loss
                best_weights = net.state_dict()
                torch.save(best_weights, os.path.join(saveweightpath, f'{exp}_epoch_{epoch}_loss_{bce_loss:.3f}.pth'))

        if scheduler is not None:
            scheduler.step()

    writer.close()



def train_model_centerloss(dataloaders, net, device, optimizer, scheduler, batch_num,
                           num_epochs, exp='test0', saveweightpath='../weights'):

    print('DeepFake Detection Challenge Training Model')
    train_i, val_i = 0, 0
    net = net.to(device)
    writer = SummaryWriter(f'../tensorboard/{exp}')
    best_loss = 1e+9
    centerloss = CenterLoss(1, 2, True).to(device)

    writer.add_text('Config/optimizer', optimizer.__class__.__name__)
    writer.add_text('Config/scheduler', scheduler.__class__.__name__)

    for epoch in range(num_epochs):
        print('#'*30)

        for phase in ['train', 'val']:

            bce_loss = 0
            accuracy = 0
            total_examples = 0

            if phase == 'train':
                net.train()
            else:
                net.eval()

            for i, (img, label) in enumerate(dataloaders[phase]):
                batch_size = img.size(0)

                # Training
                if phase == 'train':
                    optimizer.zero_grad()
                    img = img.to(device)
                    label = label.to(device).float()
                    _label = label.detach()

                    pred, lp1 = net(img)
                    pred = pred.squeeze()
                    lp1 = lp1.squeeze()

                    # Binary Cross Entropy
                    loss_xen = F.binary_cross_entropy_with_logits(pred, _label)
                    loss_cen = centerloss(lp1, label)
                    loss = loss_xen + loss_cen
                    loss.backward()
                    optimizer.step()

                    # Accuracy
                    pred = torch.sigmoid(pred.detach())
                    pred[pred > 0.5] = 1.0
                    pred[pred < 0.5] = 0.0
                    acc = torch.sum(pred == label).float() / pred.size(0)

                # Validation
                else:
                    with torch.no_grad():
                        img = img.to(device)
                        label = label.to(device).float()

                        pred = net(img)
                        pred = pred.squeeze()

                        loss_xen = F.binary_cross_entropy_with_logits(pred, label)

                        # Accuracy
                        pred = torch.sigmoid(pred.detach())
                        pred[pred > 0.5] = 1.0
                        pred[pred < 0.5] = 0.0
                        acc = torch.sum(pred == label).float() / pred.size(0)

                bce_loss += loss_xen.item() * batch_size
                accuracy += acc.item() * batch_size
                total_examples += batch_size

                # Tensorboard
                if phase == 'train':
                    writer.add_scalar('train/batch_loss', loss_xen, train_i)
                    writer.add_scalar('train/batch_acc', acc, train_i)
                    train_i += 1
                else:
                    writer.add_scalar('val/batch_loss', loss_xen, val_i)
                    writer.add_scalar('val/batch_acc', acc, val_i)
                    val_i += 1

                if phase == 'train' and i > batch_num:
                    break

            bce_loss /= total_examples
            accuracy /= total_examples

            # Tensorboard
            writer.add_scalar(f'{phase}/epoch_loss', bce_loss, epoch)
            writer.add_scalar(f'{phase}/epoch_acc', accuracy, epoch)
            print(f'Epoch {epoch}  {phase} BCELoss: {bce_loss:.4f} Accuracy: {accuracy:.4f}')

            if phase == 'val' and bce_loss < best_loss:
                best_loss = bce_loss
                best_weights = net.state_dict()
                torch.save(best_weights, os.path.join(saveweightpath, f'{exp}_epoch_{epoch}_loss_{bce_loss:.3f}.pth'))

        if scheduler is not None:
            scheduler.step()

    writer.close()


