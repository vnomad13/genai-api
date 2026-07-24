import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_energy_samples(nn_energy_model, inp_imgs, steps, step_size, noise_std):
    """Langevin dynamics: gradient descent on the INPUT pixels (not the weights)."""
    nn_energy_model.eval()
    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn_like(inp_imgs) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)
        inp_imgs.requires_grad_(True)
        energy = nn_energy_model(inp_imgs)
        grads, = torch.autograd.grad(energy, inp_imgs, grad_outputs=torch.ones_like(energy))
        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)
    return inp_imgs.detach()


class Buffer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.examples = [torch.rand((1, 1, 32, 32), device=device) * 2 - 1 for _ in range(128)]

    def sample_new_exmps(self, steps, step_size, noise):
        import numpy as np, random
        n_new = np.random.binomial(128, 0.05)
        new_rand_imgs = torch.rand((n_new, 1, 32, 32), device=self.device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=128 - n_new), dim=0)
        inp_imgs = torch.cat([new_rand_imgs, old_imgs], dim=0)
        new_imgs = generate_energy_samples(self.model, inp_imgs, steps, step_size, noise)
        self.examples = list(torch.split(new_imgs, 1, dim=0)) + self.examples
        self.examples = self.examples[:8192]
        return new_imgs


class EBM:
    """Trains the EnergyModel with contrastive divergence:
       push energy DOWN on real images, UP on generated (model) images."""

    def __init__(self, model, alpha, steps, step_size, noise, device):
        self.device = device
        self.model = model
        self.buffer = Buffer(model, device=device)
        self.alpha, self.steps, self.step_size, self.noise = alpha, steps, step_size, noise

    def train_step(self, real_imgs, optimizer):
        real_imgs = torch.clamp(real_imgs + torch.randn_like(real_imgs) * self.noise, -1.0, 1.0)
        fake_imgs = self.buffer.sample_new_exmps(self.steps, self.step_size, self.noise)
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0).clone().detach().to(self.device)
        out_scores = self.model(inp_imgs)
        real_out, fake_out = torch.split(out_scores, [real_imgs.size(0), fake_imgs.size(0)], dim=0)
        cdiv_loss = real_out.mean() - fake_out.mean()
        reg_loss = self.alpha * (real_out.pow(2).mean() + fake_out.pow(2).mean())
        loss = cdiv_loss + reg_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
        optimizer.step()
        return {"loss": loss.item(), "cdiv": cdiv_loss.item()}
