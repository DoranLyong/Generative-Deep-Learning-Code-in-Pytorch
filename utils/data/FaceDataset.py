import torchvision
import os
import PIL
import numpy as np
import torch as t
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self, faces_path, img_sizes=[128, 128], limit=None):
        items = []
        labels = []
        for img in os.listdir(faces_path)[:limit] if limit else os.listdir(faces_path):
            item = os.path.join(faces_path, img)
            items.append(item)
            labels.append(img)
        self.items = items
        self.labels = labels
        self.img_sizes = img_sizes

    def __len__(self):
        return len(self.items)

    def _get_image_(self, idx):
        img = self.items[idx]
        img = PIL.Image.open(str(img)).convert('RGB')
        img = torchvision.transforms.Resize(self.img_sizes)(img)
        a = np.asarray(img)
        a = np.transpose(a, (1, 0, 2))
        a = np.transpose(a, (2, 1, 0))
        return t.from_numpy(a.astype(np.float32, copy=False)).div(255)

    def __getitem__(self, idx):
        return self._get_image_(idx), self.labels[idx]

    def get_item_by_jpg(self, jpg):
        return self._get_image_(self.labels.index(jpg))
