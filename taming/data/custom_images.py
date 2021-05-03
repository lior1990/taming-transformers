from typing import Optional

from torch.utils.data import Dataset, DataLoader
import imageio
from torchvision.transforms import transforms

import os
import pytorch_lightning as pl


class MultipleImageDataset(Dataset):
    def __init__(self, image_path, data_rep):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        images = []
        for img_path in os.listdir(image_path):
            image_full_scale = imageio.imread(os.path.join(image_path, img_path))[:, :, :3]
            images.append(transform(image_full_scale))
        assert len(images) > 0

        self.images = images
        self.number_of_images = len(images)
        self.data_rep = data_rep

    def __getitem__(self, idx):
        return self.images[idx % self.number_of_images]

    def __len__(self):
        return len(self.images) * self.data_rep


class MultipleImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, num_workers, train_path, validation_path):
        super().__init__()
        self.normalize = transforms.Normalize((0.5,), (0.5,))
        self.to_tensor = transforms.ToTensor()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

        if not (os.path.exists(train_path) and os.path.isdir(train_path)
                and os.path.exists(validation_path) and os.path.isdir(validation_path)):
            raise ValueError("invalid path")

        self.train_path = train_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.validation_path = validation_path
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        self.train_dataset = MultipleImageDataset(self.train_path, data_rep=1000)
        self.val_dataset = MultipleImageDataset(self.validation_path, data_rep=1)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
