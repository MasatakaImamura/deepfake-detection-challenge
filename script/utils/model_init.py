from torchvision import models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


def model_init(model_name, pretrained=True):

    model_dict = {
        'resnet50': models.resnet50(pretrained=pretrained),
        'resnet152': models.resnet152(pretrained=pretrained),
        'vgg16': models.vgg16(pretrained=pretrained),
        'vgg19': models.vgg19(pretrained=pretrained),
        'efficientnet-b1': EfficientNet.from_pretrained('efficientnet-b0'),
        'efficientnet-b4': EfficientNet.from_pretrained('efficientnet-b4'),
        'efficientnet-b7': EfficientNet.from_pretrained('efficientnet-b7')
    }

    assert model_name in model_dict.keys()

    model = model_dict[model_name]

    if 'resnet' in model_name:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=1)

    elif 'vgg' in model_name:
        model.classifier[6] = nn.Linear(in_features=4096, out_features=1)

    elif 'efficientnet' in model_name:
        model._fc = nn.Linear(in_features=model._fc.in_features, out_features=1)

    return model
