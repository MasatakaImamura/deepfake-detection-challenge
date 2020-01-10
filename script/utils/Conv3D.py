import torch.nn as nn

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
