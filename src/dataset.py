import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
import albumentations as A


class SegDataset(Dataset):

    def __init__(self, image_paths, mask_paths, tone_to_ind, is_train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.tone_to_ind = tone_to_ind

        # Простейшие аугментации только для train
        if is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(p=0.3),
            ])
        else:
            self.transform = A.Compose([])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = np.array(Image.open(image_path).convert("RGB"))
        mask_tone = np.array(Image.open(mask_path))  # маска в оттенках серого


        mask_index = np.zeros_like(mask_tone, dtype=np.int64)
        unique_tones = np.unique(mask_tone)
        for t in unique_tones:
            if t in self.tone_to_ind:
                mask_index[mask_tone == t] = self.tone_to_ind[t]


        augmented = self.transform(image=image, mask=mask_index)
        image = augmented["image"]
        mask_index = augmented["mask"]


        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_index = torch.from_numpy(mask_index).long()

        return image, mask_index
