import torchvision.transforms as transforms
import os, glob
import PIL
from PIL import Image
from torch.utils.data import Dataset


class AppleOrangeDataset(Dataset):
    def __init__(self, ao_path='data/apple2orange', tfms=None, train=True):
        self.tfms = transforms.Compose(tfms)
        self.a = sorted(glob.glob(os.path.join(ao_path, ('train%s' if train else 'test%s') % 'A', '*.*')))
        self.o = sorted(glob.glob(os.path.join(ao_path, ('train%s' if train else 'test%s') % 'B', '*.*')))

    def __len__(self):
        return max(len(self.a), len(self.o))

    def _get_image_(self, img):
        img = PIL.Image.open(str(img)).convert('RGB')
        return self.tfms(img)

    def __getitem__(self, idx):
        a_img = self.a[idx % len(self.a)]
        o_img = self.o[idx % len(self.o)]
        return {'a': self._get_image_(a_img), 'o': self._get_image_(o_img)}
