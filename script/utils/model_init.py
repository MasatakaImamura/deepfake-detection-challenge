from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from utils.convlstm import ConvLSTM
from facenet_pytorch import InceptionResnetV1


def model_init(model_name, pretrained=True, classes=1):

    model_dict = {
        'resnet50': models.resnet50(pretrained=pretrained),
        'resnet152': models.resnet152(pretrained=pretrained),
        'vgg16': models.vgg16(pretrained=pretrained),
        'vgg19': models.vgg19(pretrained=pretrained),
        'efficientnet-b0': EfficientNet.from_pretrained('efficientnet-b0', num_classes=classes),
        'efficientnet-b4': EfficientNet.from_pretrained('efficientnet-b4', num_classes=classes),
        'efficientnet-b7': EfficientNet.from_pretrained('efficientnet-b7', num_classes=classes),
        'facenet': InceptionResnetV1(pretrained='vggface2', num_classes=1, classify=True)
    }

    assert model_name in model_dict.keys()

    model = model_dict[model_name]

    if 'resnet' in model_name:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=classes)

    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(in_features=4096, out_features=classes)

    return model


class convLSTM(nn.Module):
    def __init__(self, input_size=224, lstm_hidden_dim=20, lstm_num_layer=1, out_classes=2):
        super(convLSTM, self).__init__()
        self.lstm = ConvLSTM(input_size=(input_size, input_size), input_dim=3, hidden_dim=lstm_hidden_dim,
                             kernel_size=(3, 3), num_layers=lstm_num_layer, batch_first=True)

        self.block1 = nn.Sequential(
            nn.Conv2d(lstm_hidden_dim, 64, kernel_size=7, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm2d(lstm_hidden_dim)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64, out_classes)

    def forward(self, x):
        _, x = self.lstm(x)
        x = x[0][0]
        x = self.bn(x)
        x = self.block1(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x


class convLSTM_resnet(nn.Module):
    def __init__(self, model_name='resnet50', input_size=224, lstm_hidden_dim=20, lstm_num_layer=1, out_classes=2):
        super(convLSTM_resnet, self).__init__()
        self.lstm = ConvLSTM(input_size=(input_size, input_size), input_dim=3, hidden_dim=lstm_hidden_dim,
                             kernel_size=(3, 3), num_layers=lstm_num_layer, batch_first=True)

        self.resnet = model_init(model_name, classes=out_classes)
        self.resnet.conv1 = nn.Conv2d(lstm_hidden_dim, 64, kernel_size=(7, 7),
                                      stride=(2, 2), padding=(3, 3), bias=False)

        assert 'resnet' in model_name, "You must use 'Resnet' Model. Please Check ModelName"

    def forward(self, x):
        _, x = self.lstm(x)
        # x1 = x[0][0]
        x2 = x[0][1]

        x2 = self.resnet(x2)

        return x2


# Convolution3Dを活用したモデル
class Conv3dnet(nn.Module):

    def __init__(self, output_size=2):
        super(Conv3dnet, self).__init__()

        self.block_3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool_1 = nn.MaxPool3d(2)

        self.block_3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.pool_2 = nn.MaxPool3d(2)

        self.block_3d_3 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        self.pool_3 = nn.MaxPool3d(2)

        self.block_2d_1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(256, output_size)

    def forward(self, x):
        print('Model')
        x = self.block_3d_1(x)
        print(x.size())
        x = self.pool_1(x)
        print(x.size())
        x = self.block_3d_2(x)
        print(x.size())
        x = self.pool_2(x)
        print(x.size())
        x = self.block_3d_3(x)
        print(x.size())
        x = self.pool_3(x)
        print(x.size())
        x = x.squeeze()
        print(x.size())
        x = self.block_2d_1(x)
        print(x.size())
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)

        return x
