import torch
from torch.utils.data import Dataset, DataLoader

from .utils import get_img_from_mov, detect_face


class DeepfakeDataset_idx0(Dataset):
    def __init__(self, file_list, metadata, transform=None, phase='train'):
        self.file_list = file_list
        self.metadata = metadata
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        mov_path = self.file_list[idx]

        # Label
        label = self.metadata[self.metadata['mov'] == mov_path.split('\\')[-1]]['label'].values

        if label == 'FAKE':
            label = 1.0
        else:
            label = 0.0

        # Movie to Image
        try:
            image = get_img_from_mov(mov_path)[0]  # Only First Frame Face
            # FaceCrop
            image = detect_face(image)[0]
            # Transform
            image = self.transform(image, self.phase)
        except:
            image = torch.ones((3, self.transform.size, self.transform.size), dtype=torch.float)
            label = 1.0

        return image, label, mov_path