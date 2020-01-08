import torch.nn as nn

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
        x = self.block_3d_1(x)
        x = self.pool_1(x)
        x = self.block_3d_2(x)
        x = self.pool_2(x)
        x = self.block_3d_3(x)
        x = self.pool_3(x)
        x = x.squeeze()
        x = self.block_2d_1(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.fc(x)

        return x