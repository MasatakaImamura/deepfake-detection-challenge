import cv2
import random
import torch
from torchvision import transforms
from PIL import Image


class Resize:
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


class RandomFlip:
    def __init__(self):
        pass

    def __call__(self, image):
        r = random.choice([0, 1, -1, 999])
        if r != 999:
            image = cv2.flip(image, r)
        else:
            pass
        return image


class RandomRotate:
    def __init__(self):
        pass

    def __call__(self, image):
        r = random.choice([0, 1, 2, 3])

        # 時計回り
        if r == 1:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # 反時計回り
        elif r == 2:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # 180°
        elif r == 3:
            image = cv2.rotate(image, cv2.ROTATE_180)
        else:
            pass

        return image


# Data Augumentation
class ImageTransform:
    def __init__(self, resize, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.data_transform = {
            'train': transforms.Compose([
                Resize(resize),
                # RandomFlip(),
                # RandomRotate(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]),
            'val': transforms.Compose([
                Resize(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        }
        self.size = resize

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
        return [self.totensor(img) for img in img_group]


class GroupNormalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.normalize(img) for img in img_group]

class Stack:
    def __call__(self, img_group):
        ret = torch.stack(img_group)
        return ret


class GroupImageTransform:

    def __init__(self, size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transform = {
            'train': transforms.Compose([
                GroupResize(size),
                GroupToTensor(),
                GroupNormalize(mean, std),
                Stack()
            ]),
            'val': transforms.Compose([
                GroupResize(size),
                GroupToTensor(),
                GroupNormalize(mean, std),
                Stack()
            ])
        }

    def __call__(self, img_group, phase):
        return self.transform[phase](img_group)
