import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Resnet_3D(nn.Module):
    '''Resnet_3D_1'''

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

    def __init__(self, output_size):
        super(Efficientnet_3d, self).__init__()

        self.base = EfficientNet.from_pretrained('efficientnet-b0', num_classes=output_size)
        self.resnet_3d_1 = Resnet_3D(input_size=40, output_size=96)
        self.resnet_3d_2 = Resnet_3D(input_size=96, output_size=96)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc_1 = nn.Linear(96, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_last = nn.Linear(64, output_size, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        # 3次元に圧縮
        x = x.view(-1, c, h, w)

        # Efficientnet-b4の0-22層目まで通す
        # Output_size -> (bs, 160, 14, 14)
        x = self.base._conv_stem(x)
        x = self.base._bn0(x)

        for m in self.base._blocks[:4]:
            x = m(x)

        # 4次元に戻す
        _, c, w, h = x.shape
        x = x.view(-1, ns, c, w, h)
        x = x.transpose(2, 1)

        x = self.resnet_3d_1(x)
        x = self.resnet_3d_2(x)

        x = self.global_pool(x)
        x = x.squeeze()

        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_last(x)

        return x

