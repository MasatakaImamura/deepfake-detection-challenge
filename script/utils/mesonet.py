
# Reference
# https://arxiv.org/pdf/1809.00888.pdf

import torch.nn as nn
import torch
import torch.nn.functional as F


class Mesonet(nn.Module):
    def __init__(self, output_size=2):
        super(Mesonet, self).__init__()

        self.output_size = output_size

        self.meso_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2)
        )

        self.meso_block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2)
        )

        self.meso_block3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2)
        )

        self.meso_block4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(4)
        )

        self.dropout_2d = nn.Dropout2d(0.5)

        self.dropout_1d = nn.Dropout(0.5)

    def forward(self, x):

        # Input shape (3, 256, 256)
        x = self.meso_block1(x)
        x = self.meso_block2(x)
        x = self.meso_block3(x)
        x = self.meso_block4(x)
        x = self.dropout_2d(x)
        x = x.view(x.size()[0], -1)

        self.fc_1 = nn.Linear(x.size()[1], 16)
        self.fc_2 = nn.Linear(16, self.output_size)

        x = self.fc_1(x)
        x = self.dropout_1d(x)
        x = self.fc_2(x)

        return x
