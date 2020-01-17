import time, copy, gc, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

from utils.dfdc_dataset import face_img_generator

from utils.logger import create_logger, get_logger


def train_model(net, dataloader_dict, criterion, optimizer, num_epoch, device, model_name, label_smooth=0,
                version='000'):
    print('')
    print('DFDC Training...')
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 1.0
    net = net.to(device)
    train_loss_list = []
    val_loss_list = []

    assert label_smooth < 0.4, 'You must set label_smooth < 0.4'

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels, _ in tqdm(dataloader_dict[phase]):

                if '3D' in model_name:
                    # b, D, C, H, W -> b, C, D, H, W
                    inputs = inputs.permute(0, 2, 1, 3, 4)

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # LossがBCEの場合有効にする
                _labels = labels.unsqueeze(1).float()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, _labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    # Accuracy
                    # Output_size = 1
                    outputs = torch.sigmoid(outputs)
                    # Output_size = 2
                    # outputs = torch.softmax(outputs, dim=1)[:, 1]
                    outputs[outputs > 0.5] = 1
                    outputs[outputs < 0.5] = 0
                    acc = torch.sum(outputs == labels)
                    epoch_corrects += acc.item() / outputs.size()[0]

                del inputs, labels, _labels
                gc.collect()
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloader_dict[phase].dataset)

            get_logger(version).info(
                f'Epoch {epoch + 1}/{num_epoch} {phase} Loss: {epoch_loss} Acc: {epoch_acc}'
            )

            # Save Epoch Loss
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(net.state_dict())
                torch.save(net.state_dict(), "../model/temp_{}.pth".format(model_name))

        df_loss = pd.DataFrame({
            'Train_loss': train_loss_list,
            'Val_loss': val_loss_list
        })

        df_loss.to_csv(f'../loss/{model_name}_loss.csv')

    time_elapsed = time.time() - since
    print('Training complete in {}'.format(str(datetime.timedelta(seconds=time_elapsed))))
    print('Best val Acc: {:4f}'.format(best_loss))

    df_loss = pd.DataFrame({
        'Epoch': np.arange(num_epoch),
        'Train_loss': train_loss_list,
        'Val_loss': val_loss_list
    })

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, best_loss, df_loss

# For Continuous
def train_model_by_generator(net, metadata, train_mov_path, transform, val_mov_path, criterion, optimizer,
                             num_epoch, device, frame_window, max_frame=300):

    net = net.to(device)

    for epoch in range(num_epoch):

        train_epoch_loss = 0.0
        train_epoch_correct = 0
        since = time.time()

        # Train
        for mov_path in train_mov_path:
            train_gen = face_img_generator(mov_path, metadata, device, transform=transform,
                                           phase='train', frame_window=frame_window)

            net = net.train()

            num_img = 0
            print(num_img)

            for inputs, labels, idx in train_gen:
                phase = 'train'
                if inputs is None:
                    continue

                inputs = inputs.to(device)
                labels = torch.tensor(labels)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs.unsqueeze(0))

                    if idx == 0:
                        loss = criterion(outputs, labels.unsqueeze(0))
                    else:
                        loss += criterion(outputs, labels.unsqueeze(0))
                    # loss.backward()
                    # optimizer.step()

                    train_epoch_loss += loss.item()

                    pred = torch.softmax(outputs, dim=1)[:, 1].item()
                    if pred > 0.5 and labels.item() == 1:
                        train_epoch_correct += 1
                    elif pred < 0.5 and labels.item() == 0:
                        train_epoch_correct += 1
                    else:
                        pass

                num_img += 1

                if idx > max_frame:
                    break

            loss.backward()
            optimizer.step()

            train_epoch_loss /= num_img
            train_epoch_correct /= num_img

        train_epoch_loss /= len(train_mov_path)
        train_epoch_correct /= len(train_mov_path)

        print('Epoch {}  Loss: {:.3f}  Acc: {:.3f}'.format(epoch+1, train_epoch_loss, train_epoch_correct))
        s = time.time() - since
        print('Elapsed Time: {}'.format(str(datetime.timedelta(seconds=s))))

    # Valid
    val_epoch_loss = 0.0
    val_epoch_correct = 0

    for mov_path in val_mov_path:
        val_gen = face_img_generator(mov_path, metadata, device, transform=transform,
                                       phase='val', frame_window=frame_window)

        net = net.eval()

        num_img = 0

        for inputs, labels, idx in val_gen:
            phase = 'train'
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = net(inputs)
                loss = criterion(outputs, labels.long())

                val_epoch_loss += loss.item()

                pred = torch.softmax(outputs, dim=1)[:, 1].item()
                if pred > 0.5 and labels.item() == 1:
                    val_epoch_correct += 1
                elif pred < 0.5 and labels.item() == 0:
                    val_epoch_correct += 1
                else:
                    pass

            num_img += 1

            if idx > 250:
                break

        val_epoch_loss /= num_img
        val_epoch_correct /= num_img

    print('Validation  Loss: {:.3f}  Acc: {:.3f}'.format(val_epoch_loss, val_epoch_correct))

    return net