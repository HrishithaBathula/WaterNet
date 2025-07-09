
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class UIEBDataset(Dataset):
    def __init__(self, raw_dir, ref_dir, transform=None):
        self.raw_dir = raw_dir
        self.ref_dir = ref_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(raw_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        raw_path = os.path.join(self.raw_dir, filename)
        ref_path = os.path.join(self.ref_dir, filename)

        raw = Image.open(raw_path).convert('RGB')
        ref = Image.open(ref_path).convert('RGB')

        if self.transform:
            raw = self.transform(raw)
            ref = self.transform(ref)

        wb = raw 
        he = raw
        gc = raw

        return raw, wb, he, gc, ref

def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.cpu().clamp(0, 1))
    img.save(path)

def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
