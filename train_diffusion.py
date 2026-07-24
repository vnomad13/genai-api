"""Train the UNet diffusion model on CIFAR-10 and save weights to weights/diffusion.pth."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from diffusion_model import UNet, DiffusionModel

EPOCHS = 20
BATCH_SIZE = 64
IMAGE_SIZE = 32
WEIGHTS_DIR = Path("weights")
WEIGHTS_OUT = WEIGHTS_DIR / "diffusion.pth"


def compute_normalizer(dataloader, device):
    """Per-channel mean/std over the training set, shaped 1x3x1x1."""
    total = 0
    channel_sum = torch.zeros(3, device=device)
    channel_sq_sum = torch.zeros(3, device=device)
    for images, _ in dataloader:
        images = images.to(device)
        n = images.size(0) * images.size(2) * images.size(3)
        channel_sum += images.sum(dim=[0, 2, 3])
        channel_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
        total += n
    mean = channel_sum / total
    std = (channel_sq_sum / total - mean ** 2).clamp(min=1e-12).sqrt()
    return mean.view(1, 3, 1, 1), std.view(1, 3, 1, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device} for {EPOCHS} epoch(s)...")

    transform = T.Compose([
        T.ToTensor(),                   # CIFAR-10 is already 32x32
    ])
    dataset = CIFAR10(root="data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, persistent_workers=True)
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches/epoch")

    unet = UNet(IMAGE_SIZE, 3, 64)
    model = DiffusionModel(unet)
    model.to(device)

    optimizer = torch.optim.AdamW(model.network.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.L1Loss()

    mean, std = compute_normalizer(dataloader, device)
    model.set_normalizer(mean, std)
    print(f"normalizer mean: {mean.flatten().tolist()}")
    print(f"normalizer std : {std.flatten().tolist()}")

    for epoch in range(EPOCHS):
        loss_total = 0.0
        for images, _ in dataloader:
            loss_total += model.train_step(images.to(device), optimizer, loss_fn)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | loss: {loss_total / len(dataloader):.4f}")

    # copy trained weights into the EMA net for safety
    model.ema_network.load_state_dict(model.network.state_dict())

    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save({
        'network': model.network.state_dict(),
        'mean': model.normalizer_mean.detach().cpu(),
        'std': model.normalizer_std.detach().cpu(),
    }, WEIGHTS_OUT)
    print("Saved weights/diffusion.pth")


if __name__ == "__main__":
    main()
