import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


class Facenet_Resnet_3D_1(nn.Module):
    '''Resnet_3D_1'''

    def __init__(self, _in, _out):
        super(Facenet_Resnet_3D_1, self).__init__()

        self.res3a_2 = nn.Conv3d(_in, _out, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.res3a_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(_out, _out, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res3b_1_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(_out, _out, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res3b_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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


class Facenet_Resnet_3D_2(nn.Module):
    '''Resnet_3D_2'''
    def __init__(self, _in, _out):
        super(Facenet_Resnet_3D_2, self).__init__()

        self.res4a_1 = nn.Conv3d(_in, _out, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.res4a_1_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_1_relu = nn.ReLU(inplace=True)
        self.res4a_2 = nn.Conv3d(_out, _out, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res4a_down = nn.Conv3d(_in, _out, kernel_size=(
            3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.res4a_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4a_relu = nn.ReLU(inplace=True)

        self.res4b_1 = nn.Conv3d(_out, _out, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.res4b_1_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(_out, _out, kernel_size=(
            3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

        self.res4b_bn = nn.BatchNorm3d(
            _out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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


class Facenet_3d(nn.Module):

    def __init__(self, output_size=1):
        super(Facenet_3d, self).__init__()

        self.facenet = InceptionResnetV1(pretrained='vggface2')
        self.resnet_3d_1 = Facenet_Resnet_3D_1(_in=896, _out=256)
        self.resnet_3d_2 = Facenet_Resnet_3D_2(_in=256, _out=256)
        self.global_pool = nn.AvgPool3d(kernel_size=(4, 3, 3), stride=1, padding=0)
        self.fc_1 = nn.Linear(256, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.fc_last = nn.Linear(64, output_size, bias=True)

    def forward(self, x):
        bs, ns, c, h, w = x.shape
        # 3次元に圧縮
        x = x.view(-1, c, h, w)

        # InceptionV1の0-層目まで通す
        # Output_size -> (bs, 160, 14, 14)
        x = self.facenet.conv2d_1a(x)
        x = self.facenet.conv2d_2a(x)
        x = self.facenet.conv2d_2b(x)
        x = self.facenet.maxpool_3a(x)
        x = self.facenet.conv2d_3b(x)
        x = self.facenet.conv2d_4a(x)
        x = self.facenet.conv2d_4b(x)  # (bn, 256, 25, 25)
        x = self.facenet.repeat_1(x)   # (bn, 256, 25, 25)
        x = self.facenet.mixed_6a(x)   # (bn, 896, 12, 12)
        x = self.facenet.repeat_2(x)   # (bn, 896, 12, 12)
        # x = self.facenet.mixed_7a(x)   # (bn, 1792, 5, 5)
        # x = self.facenet.repeat_3(x)   # (bn, 1792, 5, 5)
        # x = self.facenet.block8(x)     # (bn, 1792, 5, 5)

        # 4次元に戻す
        x = x.view(-1, ns, 896, 12, 12)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.resnet_3d_1(x)
        x = self.resnet_3d_2(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), x.size(1))

        x = self.fc_1(x)
        x = self.dropout(x)
        x = self.fc_last(x)

        return x


if __name__ == '__main__':

    z = torch.randn(4, 14, 3, 224, 224)
    net = Facenet_3d()
    out = net(z)

    print(out.size())
