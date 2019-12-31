import cv2
import random

from torchvision import transforms


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
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                Resize(resize),
                # RandomFlip(),
                # RandomRotate(),
                transforms.ToTensor(),
            ]),
            'val': transforms.Compose([
                Resize(resize),
                transforms.ToTensor(),
            ])
        }
        self.size = resize

    def __call__(self, img, phase):
        return self.data_transform[phase](img)
