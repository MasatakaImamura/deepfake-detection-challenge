import torch.nn as nn

# Convolution3Dを活用したモデル
class Conv3dnet(nn.Module):

    def __init__(self, output_size=2):
        super(Conv3dnet, self).__init__()

        self.block_3d_1 = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=5, stride=2),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.block_3d_1(x)
        x = self.avgpool(x)
        x = x.squeeze()
        x = self.dropout(x)
        x = self.fc(x)

        return x
