import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import ImageFolder

class FractalDataModule(pl.LightningDataModule):

    def __init__(self, data_dir: str = './', batch_size=128) :
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # Todo download with get etc.
        pass

    def setup(self, train_ratio=0.8):
        fdb_full = ImageFolder(self.data_dir, transform=self.transform)
        n_train, n_val = int(len(fdb_full) * train_ratio), int(len(fdb_full) * (1 - train_ratio) / 2) 
        self.fdb_train, self.fdb_val, self.fdb_test = random_split(fdb_full, [n_train, n_val, n_val], generator=torch.Generator().manual_seed(42))


    def train_dataloader(self):
        return DataLoader(self.fdb_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.fdb_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.fdb_test, batch_size=self.batch_size)
