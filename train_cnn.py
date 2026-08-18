"""Assignment 2: train the CNN on CIFAR-10 and save to weights/cnn_cifar10.pth.

CIFAR-10 is 32x32; images are resized to 64x64 to match the required architecture.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from cnn_model import CNN

EPOCHS = 10
BATCH_SIZE = 128
LR = 1e-3
WEIGHTS_OUT = Path("weights/cnn_cifar10.pth")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def loaders():
    train_tf = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    test_tf = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
    train = CIFAR10(root="data", train=True, download=True, transform=train_tf)
    test = CIFAR10(root="data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                   persistent_workers=True),
        DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                   persistent_workers=True),
    )


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = loaders()
    print(f"Training on {device} | {len(train_loader)} batches/epoch")

    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            n += 1
        acc = accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | loss: {total/n:.4f} | test acc: {100*acc:.2f}%")

    WEIGHTS_OUT.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"Saved {WEIGHTS_OUT}")


if __name__ == "__main__":
    main()
