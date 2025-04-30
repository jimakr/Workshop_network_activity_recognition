import pandas as pd
from torch import optim, nn
import lightning as L
from torch.utils.data import Dataset
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import io
from Dataset_loaders.Activity_dataset import ActivityDataset
from torchvision.transforms import ToTensor
from torch.nn import Dropout
from torchmetrics import ConfusionMatrix, F1Score
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
import random


# define the LightningModule
class LitMLP(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30 ,256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(256),
            Dropout(p=0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(512),
            Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.05),
            nn.BatchNorm1d(256),
            Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 23),
        )

        self.val_output = []
        self.val_real = []
        # self.conf = ConfusionMatrix(task='multiclass', num_classes=15)
        self.F1 = F1Score(task='multiclass', num_classes=23, average='micro')

        self.fig, self.ax = plt.subplots(figsize=(25, 25))


    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.argmax(y, dim=1)

        loss = nn.functional.cross_entropy(y_hat, y)

        self.log("train F1 score", self.F1(y_hat, y), on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)

        return loss

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)

        y = torch.argmax(y, dim=1)

        loss = nn.functional.cross_entropy(y_hat, y)

        # Logging to TensorBoard (if installed) by default
        self.log("val F1 score", self.F1(y_hat, y), on_epoch=True, on_step=False)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.val_output.append(y_hat)
        self.val_real.append(y)
        return loss

    def on_validation_epoch_end(self):
        outputs = torch.cat([pred for pred in self.val_output])
        labels = torch.cat([real for real in self.val_real])
        # labels = torch.argmax(labels, dim=1)

        self.val_output = []
        self.val_real = []

        self.conf = ConfusionMatrix(num_classes=23, task='multiclass').to(outputs.get_device())
        self.conf(outputs, labels)
        # computed_confusion = self.conf.compute().detach().cpu().numpy().astype(int)

        self.conf.plot(ax=self.ax)

        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        im = ToTensor()(Image.open(buf))

        self.logger.experiment.add_image(
            "val_confusoin_matrix",
            im,
            global_step=self.current_epoch,
        )
        plt.cla()

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y_hat = self(x)

        loss = nn.functional.cross_entropy(y_hat, y)
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, on_epoch=True)
        return y_hat

# init the autoencoder
if __name__ == "__main__":
    seed = random.randint(0, 123456789)
    print(seed)
    seed_everything(seed, workers=True)

    dataset = ActivityDataset()
    train, val, test = dataset.load().get_dataloaders()
    autoencoder = LitMLP()

    callbacks = [
        ModelCheckpoint(dirpath='checkpoints/MLP_activity', filename='{epoch}-{val F1 score:.5f}', save_last=True,monitor='val F1 score', mode='max', save_top_k=5)
    ]

    trainer = L.Trainer(max_epochs=80, accelerator='gpu', callbacks=callbacks)
    trainer.fit(model=autoencoder, train_dataloaders=train, val_dataloaders=val)

    print('predicting')
    trainer.test(ckpt_path='best', dataloaders=test)
