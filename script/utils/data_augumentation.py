import cv2

from torchvision import transforms

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


# Data Augumentation
class ImageTransform():
    def __init__(self, resize):
        self.data_transform = {
            'train': transforms.Compose([
                Resize(resize),
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