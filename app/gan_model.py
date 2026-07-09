import torch
import torch.nn as nn

LATENT_DIM = 100
IMG_CHANNELS = 1
FEATURE_G = 64
FEATURE_C = 64


class Generator(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # (latent_dim, 1, 1) → (FEATURE_G*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, FEATURE_G * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(FEATURE_G * 8),
            nn.ReLU(True),
            # → (FEATURE_G*4, 8, 8)
            nn.ConvTranspose2d(FEATURE_G * 8, FEATURE_G * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_G * 4),
            nn.ReLU(True),
            # → (FEATURE_G*2, 16, 16)
            nn.ConvTranspose2d(FEATURE_G * 4, FEATURE_G * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_G * 2),
            nn.ReLU(True),
            # → (FEATURE_G, 32, 32)
            nn.ConvTranspose2d(FEATURE_G * 2, FEATURE_G, 4, 2, 1, bias=False),
            nn.BatchNorm2d(FEATURE_G),
            nn.ReLU(True),
            # → (IMG_CHANNELS, 64, 64)
            nn.ConvTranspose2d(FEATURE_G, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Critic(nn.Module):
    """WGAN critic (no sigmoid, no BatchNorm per WGAN paper)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (IMG_CHANNELS, 64, 64) → (FEATURE_C, 32, 32)
            nn.Conv2d(IMG_CHANNELS, FEATURE_C, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (FEATURE_C*2, 16, 16)
            nn.Conv2d(FEATURE_C, FEATURE_C * 2, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (FEATURE_C*4, 8, 8)
            nn.Conv2d(FEATURE_C * 2, FEATURE_C * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (FEATURE_C*8, 4, 4)
            nn.Conv2d(FEATURE_C * 4, FEATURE_C * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # → (1, 1, 1)
            nn.Conv2d(FEATURE_C * 8, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


def weights_init(m: nn.Module) -> None:
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train_wgan(
    dataloader,
    n_epochs: int = 5,
    lr: float = 5e-5,
    n_critic: int = 5,
    clip_value: float = 0.01,
    device: str = "cpu",
    save_path: str = "app/generator.pth",
) -> Generator:
    gen = Generator().to(device)
    critic = Critic().to(device)
    gen.apply(weights_init)
    critic.apply(weights_init)

    opt_gen = torch.optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = torch.optim.RMSprop(critic.parameters(), lr=lr)

    for epoch in range(n_epochs):
        for i, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)

            # Train critic
            for _ in range(n_critic):
                noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
                fake = gen(noise).detach()
                loss_critic = -(critic(real).mean() - critic(fake).mean())
                opt_critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # Train generator
            noise = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            loss_gen = -critic(gen(noise)).mean()
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if i % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{n_epochs}] Step {i} | "
                    f"Critic: {-loss_critic.item():.4f} | Gen: {loss_gen.item():.4f}"
                )

    torch.save(gen.state_dict(), save_path)
    print(f"Generator weights saved to {save_path}")
    return gen


def load_generator(weights_path: str, device: str = "cpu") -> Generator:
    gen = Generator().to(device)
    gen.load_state_dict(torch.load(weights_path, map_location=device))
    gen.eval()
    return gen


def generate_sample(gen: Generator, device: str = "cpu") -> torch.Tensor:
    """Returns a (1, 64, 64) tensor with values in [0, 255]."""
    with torch.no_grad():
        noise = torch.randn(1, LATENT_DIM, 1, 1, device=device)
        img = gen(noise).squeeze(0)  # (1, 64, 64), range [-1, 1]
    return ((img + 1) / 2 * 255).clamp(0, 255).byte()
