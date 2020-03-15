import cv2
import random
import torch
from torchvision import transforms
from PIL import Image


# for dim=3
class NormalizeOrg:
    def __init__(self):
        pass

    def __call__(self, image):
        return ((image * 255) - 127.5) / 128.0


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










# Group Img  ###########################################################################################################

class GroupResize:
    def __init__(self, size):
        self.resize = transforms.Resize(size, interpolation=Image.BILINEAR)

    def __call__(self, img_group):
        return [self.resize(img) for img in img_group]


class GroupToTensor:
    def __init__(self):
        self.totensor = transforms.ToTensor()

    def __call__(self, img_group):
        return [self.totensor(img)*255 for img in img_group]


# https://github.com/timesler/facenet-pytorch/blob/7615394e8b63be9b81ef7e892921990018cf42d8/models/mtcnn.py#L387
# facenet-pytorch
# fixed_image_standardization
class GroupNormalize:
    def __init__(self):
        pass

    def __call__(self, img_group):
        return [(img - 127.5) / 128.0 for img in img_group]


class Stack:
    def __call__(self, img_group):
        ret = torch.stack(img_group)
        return ret


class GroupImageTransform:
    def __init__(self, size):
        self.transform = {
            'train': transforms.Compose([
                GroupResize(size),
                GroupToTensor(),
                GroupNormalize(),
                Stack()
            ]),
            'val': transforms.Compose([
                GroupResize(size),
                GroupToTensor(),
                GroupNormalize(),
                Stack()
            ])
        }

    def __call__(self, img_group, phase):
        return self.transform[phase](img_group)
