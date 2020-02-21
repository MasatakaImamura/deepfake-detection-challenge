import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1


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


class Facenet_3d(nn.Module):

    def __init__(self, output_size=1):
        super(Facenet_3d, self).__init__()

        self.facenet = InceptionResnetV1(pretrained='vggface2')
        self.resnet_3d_1 = Resnet_3D(input_size=896, output_size=256)
        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
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


if __name__ == '__main__':

    z = torch.randn(4, 15, 3, 224, 224)
    net = Facenet_3d()
    out = net(z)

    print(out.size())
