"""Train a standard GAN on MNIST and save generator weights to app/generator.pth."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from app.gan_model import Generator, Discriminator, LATENT_DIM

WEIGHTS_OUT = Path("app/generator.pth")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {args.epochs} epoch(s)...")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),  # → [-1, 1], matches generator's Tanh output
    ])
    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    gen = Generator().to(device)
    disc = Discriminator().to(device)
    criterion = nn.BCELoss()
    opt_gen = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_disc = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(args.epochs):
        disc_loss_total = 0.0
        gen_loss_total = 0.0

        for real, _ in dataloader:
            real = real.to(device)
            batch_size = real.size(0)
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            noise = torch.randn(batch_size, LATENT_DIM, device=device)
            fake = gen(noise)

            loss_disc = criterion(disc(real), real_labels) + criterion(
                disc(fake.detach()), fake_labels
            )
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            loss_gen = criterion(disc(fake), real_labels)
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            disc_loss_total += loss_disc.item()
            gen_loss_total += loss_gen.item()

        avg_disc_loss = disc_loss_total / len(dataloader)
        avg_gen_loss = gen_loss_total / len(dataloader)
        print(
            f"Epoch [{epoch+1}/{args.epochs}] | D loss: {avg_disc_loss:.4f} | "
            f"G loss: {avg_gen_loss:.4f}"
        )

    torch.save(gen.state_dict(), WEIGHTS_OUT)
    print(f"Generator weights saved to {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
