"""Train the Energy-Based Model on MNIST and save weights to weights/energy_model.pth."""

from pathlib import Path

import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from energy_model import EnergyModel, EBM

EPOCHS = 10
BATCH_SIZE = 128
WEIGHTS_DIR = Path("weights")
WEIGHTS_OUT = WEIGHTS_DIR / "energy_model.pth"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {EPOCHS} epoch(s)...")

    transform = T.Compose([
        T.Pad(2, -1),                   # 28x28 → 32x32
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),    # → [-1, 1]
    ])
    dataset = MNIST(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = EnergyModel().to(device)
    ebm = EBM(model, alpha=0.1, steps=60, step_size=10, noise=0.005, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.0, 0.999))

    for epoch in range(EPOCHS):
        metrics = None
        for real, _ in dataloader:
            metrics = ebm.train_step(real.to(device), optimizer)
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | loss: {metrics['loss']:.4f} | "
            f"cdiv: {metrics['cdiv']:.4f}"
        )

    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print("Saved weights/energy_model.pth")


if __name__ == "__main__":
    main()
