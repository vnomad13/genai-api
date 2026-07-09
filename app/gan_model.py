import torch
import torch.nn as nn

LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 7 * 7 * 128)
        self.deconv = nn.Sequential(
            # (128, 7, 7) → (64, 14, 14)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # (64, 14, 14) → (1, 28, 28)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 128, 7, 7)
        return self.deconv(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # (1, 28, 28) → (64, 14, 14)
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # (64, 14, 14) → (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def load_generator(weights_path: str, device: str = "cpu") -> Generator:
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(weights_path, map_location=device))
    gen.eval()
    return gen


def generate_sample(gen: Generator, device: str = "cpu") -> torch.Tensor:
    """Returns a (28, 28) tensor with values in [0, 1]."""
    with torch.no_grad():
        noise = torch.randn(1, LATENT_DIM, device=device)
        img = gen(noise).squeeze(0).squeeze(0)  # (28, 28), range [-1, 1]
    return (img + 1) / 2


if __name__ == "__main__":
    gen = Generator()
    disc = Discriminator()
    noise = torch.randn(2, LATENT_DIM)

    x = gen.fc(noise).view(-1, 128, 7, 7)
    print("Generator fc+reshape:", tuple(x.shape))
    x = gen.deconv[0](x)
    print("Generator deconv1 (7x7 -> 14x14):", tuple(x.shape))
    x = gen.deconv[1](x)
    x = gen.deconv[2](x)
    x = gen.deconv[3](x)
    print("Generator deconv2 (14x14 -> 28x28):", tuple(x.shape))
    x = gen.deconv[4](x)
    print("Generator output:", tuple(x.shape))

    fake = gen(noise)
    d_out = disc(fake)
    print("Discriminator output:", tuple(d_out.shape))
