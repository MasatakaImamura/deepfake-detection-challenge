import cv2
import random
import torch
from torchvision import transforms
from PIL import Image


class ImageTransform:
    def __init__(self, size, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class ImageTransform_2:
    def __init__(self, size, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)


class ImageTransform_3:
    def __init__(self, size, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomGrayscale(),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                transforms.Resize((size, size), interpolation=Image.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
