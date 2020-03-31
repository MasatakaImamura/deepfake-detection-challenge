import torch.nn as nn
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


class Head(nn.Module):

    def __init__(self, in_f, out_f):
        super(Head, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(in_f, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, out_f)
            )

    def forward(self, x):
        return self.block(x)


class Efficientnet_LSTM(nn.Module):

    def __init__(self, output_size, model_name='efficientnet-b0', img_num=15, dropout_rate=0.25):
        super(Efficientnet_LSTM, self).__init__()

        self.base = EfficientNet.from_pretrained(model_name, num_classes=512)
        self.model_name = model_name
        self.dropout_rate = dropout_rate

        self.rnn = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True)

        self.head = Head(in_f=img_num*128, out_f=output_size)

    def forward(self, x):
        b, f, c, w, h = x.shape

        x = x.view(b*f, c, w, h)
        x = self.base(x)
        x = x.view(b, -1, 512)
        x, _ = self.rnn(x)
        x = nn.Flatten()(x)
        x = nn.Dropout(self.dropout_rate)(x)
        x = self.head(x)

        return x
