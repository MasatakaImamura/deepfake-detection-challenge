import os
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def train_model(dataloaders, net, device, optimizer, scheduler, batch_num,
                num_epochs, exp='test0', saveweightpath='../weights'):

    print('DeepFake Detection Challenge Training Model')
    train_i, val_i = 0, 0
    net = net.to(device)
    writer = SummaryWriter(f'../tensorboard/{exp}')

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

            # Save Weights
            if phase == 'val':
                torch.save(net.state_dict(), os.path.join(saveweightpath, f'{exp}_epoch_{epoch}_loss_{bce_loss:.3f}.pth'))

        if scheduler is not None:
            scheduler.step()

    writer.close()


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_model_cutmix(dataloaders, net, device, optimizer, scheduler, batch_num,
                       num_epochs, cutmix_prob, beta, exp='test0', saveweightpath='../weights'):

    print('DeepFake Detection Challenge Training Model')
    train_i, val_i = 0, 0
    net = net.to(device)
    writer = SummaryWriter(f'../tensorboard/{exp}')

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
                batch_size = img.size(0)
                r = np.random.rand(1)

                # Training
                if phase == 'train':
                    optimizer.zero_grad()
                    img = img.to(device)
                    label = label.to(device).float()

                    # Cutmix Method
                    if beta > 0 and r < cutmix_prob:
                        # generate mixed sample
                        lam = np.random.beta(beta, beta)
                        rand_index = torch.randperm(img.size()[0]).cuda()
                        target_a = label
                        target_b = label[rand_index]
                        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
                        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]
                        # adjust lambda to exactly match pixel ratio
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
                        # compute output
                        pred = net(img)
                        pred = pred.squeeze()
                        loss = F.binary_cross_entropy_with_logits(pred, target_a) * lam + \
                               F.binary_cross_entropy_with_logits(pred, target_b) * (1. - lam)

                    else:
                        pred = net(img)
                        pred = pred.squeeze()

                        # Binary Cross Entropy
                        loss = F.binary_cross_entropy_with_logits(pred, label)

                    # Update
                    loss.backward()
                    optimizer.step()

                # Validation
                else:
                    with torch.no_grad():
                        img = img.to(device)
                        label = label.to(device).float()

                        pred = net(img)
                        pred = pred.squeeze()

                        loss = F.binary_cross_entropy_with_logits(pred, label)

                bce_loss += loss.item() * batch_size
                total_examples += batch_size

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
            # writer.add_scalar(f'{phase}/epoch_acc', accuracy, epoch)
            print(f'Epoch {epoch}  {phase} BCELoss: {bce_loss:.4f}')

            # Save Weights
            if phase == 'val':
                torch.save(net.state_dict(), os.path.join(saveweightpath,
                                                          f'{exp}_epoch_{epoch}_loss_{bce_loss:.3f}.pth'))

        if scheduler is not None:
            scheduler.step()

    writer.close()
