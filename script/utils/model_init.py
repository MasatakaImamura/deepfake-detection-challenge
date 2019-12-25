from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet


def model_init(model_name, pretrained=True):

    model_dict = {
        'resnet50': models.resnet50(pretrained=pretrained),
        'resnet152': models.resnet152(pretrained=pretrained),
        'vgg16': models.vgg16(pretrained=pretrained),
        'vgg19': models.vgg19(pretrained=pretrained),
        'efficientnet-b0': EfficientNet.from_pretrained('efficientnet-b0', num_classes=1),
        'efficientnet-b4': EfficientNet.from_pretrained('efficientnet-b4', num_classes=1),
        'efficientnet-b7': EfficientNet.from_pretrained('efficientnet-b7', num_classes=1)
    }

    assert model_name in model_dict.keys()

    model = model_dict[model_name]

    if 'resnet' in model_name:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1)

    return model


class NormalCnn(nn.Module):
    def __init__(self):
        super(NormalCnn, self).__init__()
        self.convrelupool_1 = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.convrelupool_2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.convrelupool_3 = nn.Sequential(
            nn.Conv2d(16, 25, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.convrelupool_4 = nn.Sequential(
            nn.Conv2d(25, 50, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc_1 = nn.Linear(50 * 10 * 10, 1024)
        self.fc_2 = nn.Linear(1024, 128)
        self.fc_3 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.convrelupool_1(x)
        x = self.convrelupool_2(x)
        x = self.convrelupool_3(x)
        x = self.convrelupool_4(x)
        x = x.view(-1, 512 * 10 * 10)
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)

        return x
