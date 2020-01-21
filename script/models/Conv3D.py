import torch.nn as nn

from efficientnet_pytorch import EfficientNet

# Convolution3Dを活用したモデル
class Conv3dnet(nn.Module):

    def __init__(self, output_size=2):
        super(Conv3dnet, self).__init__()

        self.conv3d_down = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1)

        self.block_3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False)
        )

        self.block_3d_2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False)
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, output_size)

    def forward(self, x):

        residual = self.conv3d_down(x)
        out = self.block_3d_1(x)

        out += residual
        residual2 = out
        out = self.block_3d_2(out)
        out += residual2

        out = self.avgpool(out)
        out = out.squeeze()
        out = self.dropout(out)
        out = self.fc(out)

        return out


class Resnet_3D_1(nn.Module):
    '''Resnet_3D_1'''

    def __init__(self):
        super(Resnet_3D_1, self).__init__()

        self.res3a_2 = nn.Conv3d(160, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.res3a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res3b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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


class Resnet_3D_2(nn.Module):
    '''Resnet_3D_2'''
    def __init__(self):
        super(Resnet_3D_2, self).__init__()

        self.res4a_1 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res4a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_relu = nn.ReLU(inplace=True)
        self.res4a_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res4a_down = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.res4a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_relu = nn.ReLU(inplace=True)

        self.res4b_1 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res4b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(256, 256, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res4b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.res4a_down(x)

        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_relu(out)

        out = self.res4a_2(out)

        out += residual

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_relu(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_relu(out)

        out = self.res4b_2(out)

        out += residual2

        out = self.res4b_bn(out)
        out = self.res4b_relu(out)

        return out


class MyNet(nn.Module):

    def __init__(self, output_size):
        super(MyNet, self).__init__()

        self.ef = EfficientNet.from_pretrained('efficientnet-b4', num_classes=output_size)
        self.resnet_3d_1 = Resnet_3D_1()
        self.resnet_3d_2 = Resnet_3D_2()
        self.global_pool = nn.AvgPool3d(kernel_size=(4, 4, 4), stride=1, padding=0)
        self.fc_1 = nn.Linear(256, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_last = nn.Linear(64, output_size, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        # 3次元に圧縮
        x = x.view(-1, c, h, w)

        # Efficientnet-b4の0-22層目まで通す
        # Output_size -> (bs, 160, 14, 14)
        x = self.ef._conv_stem(x)
        x = self.ef._bn0(x)

        for m in self.ef._blocks[:22]:
            x = m(x)

        # 4次元に戻す
        x = x.view(-1, ns, 160, 14, 14)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.resnet_3d_1(x)
        x = self.resnet_3d_2(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_last(x)

        return x

