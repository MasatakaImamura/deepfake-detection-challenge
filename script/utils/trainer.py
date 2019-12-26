import time, copy, gc, datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn


def train_model(net, dataloader_dict, criterion, optimizer, num_epoch, device, model_name):
    print('')
    print('DFDC Training...')
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_loss = 1.0
    net = net.to(device)
    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epoch):
        print('Epoch {}/{}'.format(epoch + 1, num_epoch))

        for phase in ['train', 'val']:

            if phase == 'train':
                net.train()
            else:
                net.eval()

            epoch_loss = 0.0
            epoch_corrects = 0

            i = 0
            for inputs, labels, _ in tqdm(dataloader_dict[phase]):
                if len(inputs) == 0:
                    continue
                # Replace 4 Dim
                if inputs.dim() == 5:
                    inputs = inputs.squeeze(0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    preds = torch.sigmoid(outputs.view(-1)).mean().unsqueeze(0).to(device)
                    # loss = criterion(preds, labels)
                    loss = criterion(preds, labels.float())

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item()
                    # replace binary
                    if preds.item() > 0.5:
                        preds = 1
                    else:
                        preds = 0
                    #
                    if preds == labels.item():
                        epoch_corrects += 1

                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()

                i += 1

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

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
