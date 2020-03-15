import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet


class Efficientnet_centerloss(nn.Module):

    def __init__(self, output_size, model_name='efficientnet-b0'):
        super(Efficientnet_centerloss, self).__init__()

        self.base = EfficientNet.from_pretrained(model_name, num_classes=512)
        self.model_name = model_name

        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc_1 = nn.Linear(512, 64, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.ip1 = nn.Linear(64, 2)
        self.fc_last = nn.Linear(2, output_size, bias=True)

    def forward(self, x):
        x = self.base(x)

        x = self.fc_1(x)
        x = self.dropout(x)
        ip1 = self.ip1(x)
        x = self.fc_last(ip1)

        return x, ip1




