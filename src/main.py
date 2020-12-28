import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from rich import print, inspect
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.datasets.mnist import MNIST
from pl_bolts.datamodules import CIFAR10DataModule

from .dataloader import FractalDataModule
import torchvision.models as models


class Backbone(torch.nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.l1 = torch.nn.Linear(512 * 512, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.backbone = backbone
        self.save_hyperparameters()

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("valid_loss", loss, on_step=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument("--learning_rate", type=float, default=0.0001)
    #     return parser


@hydra.main(config_path="configs", config_name="basic")
def cli_main(cfg: DictConfig) -> None:
    inspect(cfg)
    pl.seed_everything(42)

    root = Path(hydra.utils.get_original_cwd())
    # ------------
    # data
    # ------------
    fractal_data_module = FractalDataModule(
        root.joinpath(cfg.dataset), transform=transforms.ToTensor()
    )

    # ------------
    # model
    # ------------
    _model = eval(f"models.{cfg.model}(pretrained=False)")
    model = LitClassifier(_model, cfg.learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(gpus=cfg.gpus, logger=None)
    trainer.fit(model, datamodule=fractal_data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


if __name__ == "__main__":
    cli_main()
