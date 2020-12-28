import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder


class FractalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        transform=None,
        batch_size=128,
        train_ratio=0.8,
        num_workers=12,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.train_ratio = train_ratio
        self.num_workers = num_workers

    def prepare_data(self):
        # Todo download with get etc.
        pass

    def setup(self, setup="fit"):
        fdb_full = ImageFolder(self.data_dir, transform=self.transform)
        n_train, n_val = int(len(fdb_full) * self.train_ratio), int(
            len(fdb_full) * (1 - self.train_ratio) / 2
        )
        n_test = len(fdb_full) - n_train - n_val
        self.fdb_train, self.fdb_val, self.fdb_test = random_split(
            fdb_full,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self):
        return DataLoader(
            self.fdb_train, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.fdb_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.fdb_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
