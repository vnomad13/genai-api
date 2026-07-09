"""Train WGAN on MNIST and save generator weights to app/generator.pth."""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from app.gan_model import train_wgan

WEIGHTS_OUT = Path("app/generator.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n-critic", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {args.epochs} epoch(s)...")

    transform = T.Compose([
        T.Resize(64),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),  # → [-1, 1]
    ])

    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    train_wgan(
        dataloader,
        n_epochs=args.epochs,
        lr=args.lr,
        n_critic=args.n_critic,
        device=device,
        save_path=str(WEIGHTS_OUT),
    )


if __name__ == "__main__":
    main()
