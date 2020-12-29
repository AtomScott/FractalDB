from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pl_bolts.datamodules import CIFAR10DataModule
from pytorch_lightning.metrics.classification import Accuracy
from rich import inspect, print
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
from pytorch_lightning.loggers import CSVLogger

from .dataloader import FractalDataModule
from .models import get_classifier

import pandas as pd


class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.save_hyperparameters()
        self.acc = Accuracy()
        self.df = pd.DataFrame({"Epoch": [], "Accuracy": [], "Loss": [], "Model": []})

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
        acc = self.acc(y_hat, y)
        self.log("valid_loss", loss, on_step=True)
        self.log("accuracy", acc, on_step=True)
        return {"loss": loss, "acc": acc}

    def validation_epoch_end(self, validation_step_outputs):
        acc = sum([v["acc"].item() for v in validation_step_outputs]) / len(
            validation_step_outputs
        )
        loss = sum([v["loss"].item() for v in validation_step_outputs]) / len(
            validation_step_outputs
        )
        self.df = self.df.append(
            {
                "Epoch": self.current_epoch,
                "Accuracy": acc,
                "Loss": loss,
                "Model": self.cfg.model,
            }, ignore_index=True
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def on_fit_end(self):
        print(self.df)
        self.df.to_csv("fit_result.csv")

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
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
        ]
    )

    fractal_data_module = FractalDataModule(
        root.joinpath(cfg.dataset), transform=transform
    )

    # ------------
    # model
    # ------------
    _model = get_classifier(cfg.model)
    model = LitClassifier(_model, cfg.learning_rate, cfg=cfg)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(gpus=cfg.gpus, logger=None, max_epochs=cfg.epochs)
    trainer.fit(model, datamodule=fractal_data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test()
    print(result)


if __name__ == "__main__":
    cli_main()
