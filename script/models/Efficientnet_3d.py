import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


class Resnet_3D(nn.Module):
    '''Resnet_3D'''

    def __init__(self, input_size=160, output_size=256):
        super(Resnet_3D, self).__init__()

        self.res3a_2 = nn.Conv3d(input_size, output_size, kernel_size=3, stride=2, padding=1)

        self.res3a_bn = nn.BatchNorm3d(
            output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(output_size, output_size, kernel_size=3, stride=1, padding=1)
        self.res3b_1_bn = nn.BatchNorm3d(
            output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(output_size, output_size, kernel_size=3, stride=1, padding=1)

        self.res3b_bn = nn.BatchNorm3d(
            output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res3a_2(x)
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_relu(out)
        out = self.res3b_2(out)

        out += residual

        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out


class Efficientnet_3d(nn.Module):

    def __init__(self, output_size, model_name='efficientnet-b0'):
        super(Efficientnet_3d, self).__init__()

        self.base = EfficientNet.from_pretrained(model_name, num_classes=output_size)
        self.model_name = model_name
        if self.model_name == 'efficientnet-b0':
            set_channel = 40
        elif self.model_name == 'efficientnet-b4':
            set_channel = 56

        self.resnet_3d_1 = Resnet_3D(input_size=set_channel, output_size=96)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc_1 = nn.Linear(96, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_last = nn.Linear(64, output_size, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        # 3次元に圧縮
        x = x.view(-1, c, h, w)

        x = self.base._conv_stem(x)
        x = self.base._bn0(x)

        if self.model_name == 'efficientnet-b0':
            for m in self.base._blocks[:4]:
                x = m(x)

        elif self.model_name == 'efficientnet-b4':
            for m in self.base._blocks[:10]:
                x = m(x)

        # 4次元に戻す
        _, c, w, h = x.shape
        x = x.view(-1, ns, c, w, h)
        x = x.transpose(2, 1)

        x = self.resnet_3d_1(x)

        x = self.global_pool(x)
        x = x.squeeze()

        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_last(x)

        return x


class Efficientnet_3d_centerloss(nn.Module):

    def __init__(self, output_size, model_name='efficientnet-b0'):
        super(Efficientnet_3d_centerloss, self).__init__()

        self.base = EfficientNet.from_pretrained(model_name, num_classes=output_size)
        self.model_name = model_name
        if self.model_name == 'efficientnet-b0':
            set_channel = 40
        elif self.model_name == 'efficientnet-b4':
            set_channel = 56

        self.resnet_3d_1 = Resnet_3D(input_size=set_channel, output_size=96)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc_1 = nn.Linear(96, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.ip1 = nn.Linear(64, 2)
        self.fc_last = nn.Linear(2, output_size, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        # 3次元に圧縮
        x = x.view(-1, c, h, w)

        x = self.base._conv_stem(x)
        x = self.base._bn0(x)

        if self.model_name == 'efficientnet-b0':
            for m in self.base._blocks[:4]:
                x = m(x)

        elif self.model_name == 'efficientnet-b4':
            for m in self.base._blocks[:10]:
                x = m(x)

        # 4次元に戻す
        _, c, w, h = x.shape
        x = x.view(-1, ns, c, w, h)
        x = x.transpose(2, 1)

        x = self.resnet_3d_1(x)

        x = self.global_pool(x)
        x = x.squeeze()

        x = self.fc_1(x)
        x = self.dropout(x)
        ip1 = self.ip1(x)
        x = self.fc_last(ip1)

        return x, ip1


class Efficientnet_2d(nn.Module):
    def __init__(self, output_size, model_name='efficientnet-b0'):
        super(Efficientnet_2d, self).__init__()
        self.base = EfficientNet.from_pretrained(model_name, num_classes=output_size)

    def forward(self, x):
        return self.base(x)

