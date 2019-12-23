import time, copy, gc, datetime
from tqdm import tqdm
import torch


def train_model(net, dataloader_dict, criterion, optimizer, num_epoch, device):
    print('')
    print('DFDC Training...')
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    net = net.to(device)

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
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs.view(-1), labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    preds = torch.sigmoid(outputs.view(-1))
                    # replace binary
                    preds[preds > 0.5] = 1
                    preds[preds <= 0.5] = 0
                    epoch_corrects += torch.sum(preds == labels).item()

                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_acc = epoch_corrects / len(dataloader_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print('')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {}'.format(str(datetime.timedelta(seconds=time_elapsed))))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)
    return net, best_acc


def save_model_weights(net, model_name, best_acc):
    # Save Model
    date = datetime.datetime.now().strftime('%Y%m%d')
    torch.save(net.state_dict(), "../../model/{}_acc{:.3f}_{}.pth".format(model_name, best_acc, date))
