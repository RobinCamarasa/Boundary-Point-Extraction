import os
import mlflow
import argparse
import pytorch_lightning as pl
from scripts.utils import set_mlflow
from pathlib import Path
from scripts.example.utils import MNISTModel
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


@set_mlflow
def main(result_path: Path):
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--max-epoch', type=int)
    args = parser.parse_args()

    # Initialize our model
    mnist_model = MNISTModel()

    # Initialize DataLoader from MNIST Dataset
    train_ds = MNIST(os.getcwd(), train=True,
        download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_ds, batch_size=32)

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=args.max_epoch, progress_bar_refresh_rate=20)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model
    trainer.fit(mnist_model, train_loader)


if __name__ == '__main__':
    main()
