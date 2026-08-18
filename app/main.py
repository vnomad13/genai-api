import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from app.embedding_model import get_word_vector
from app.gan_model import generate_sample, load_generator
from energy_model import EnergyModel, generate_energy_samples
from diffusion_model import UNet, DiffusionModel
from cnn_model import CNN, CLASSES

WEIGHTS_PATH = Path(__file__).parent / "generator.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _generator
    if WEIGHTS_PATH.exists():
        _generator = load_generator(str(WEIGHTS_PATH), device=DEVICE)
        print(f"Generator loaded from {WEIGHTS_PATH} on {DEVICE}")
    else:
        print("No generator weights found — /generate-digit unavailable until trained.")
    yield
    _generator = None


app = FastAPI(title="Applied GenAI API", version="0.1.0", lifespan=lifespan)


class WordRequest(BaseModel):
    word: str


class EmbeddingResponse(BaseModel):
    word: str
    vector: list[float]
    dim: int


@app.get("/")
def root():
    return {"status": "ok", "message": "Applied GenAI API is running"}


@app.post("/embedding", response_model=EmbeddingResponse)
def embedding(req: WordRequest):
    try:
        vector = get_word_vector(req.word.strip())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return EmbeddingResponse(word=req.word, vector=vector, dim=len(vector))


@app.post("/generate-digit")
def generate_digit():
    if _generator is None:
        raise HTTPException(
            status_code=503,
            detail="Generator weights not loaded. Run train_gan.py first.",
        )
    img_tensor = generate_sample(_generator, device=DEVICE)  # (28, 28) float in [0, 1]
    img_array = (img_tensor * 255).clamp(0, 255).byte().cpu().numpy()
    pil_img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


def _tensor_to_png(img_tensor):
    # img_tensor: (C,H,W) in [0,1]
    arr = (img_tensor.clamp(0, 1).cpu().numpy() * 255).astype('uint8')
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode='L')
    else:
        pil = Image.fromarray(arr.transpose(1, 2, 0), mode='RGB')
    buf = io.BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)
    return buf


@app.get("/generate-energy")
def generate_energy():
    device = 'cpu'
    model = EnergyModel().to(device)
    model.load_state_dict(torch.load('weights/energy_model.pth', map_location='cpu'))
    x = torch.rand((1, 3, 32, 32), device=device) * 2 - 1
    out = generate_energy_samples(model, x, steps=256, step_size=10.0, noise_std=0.01)
    img = (out[0] + 1) / 2  # [-1,1] -> [0,1]
    return StreamingResponse(_tensor_to_png(img), media_type="image/png")


@app.get("/generate-diffusion")
def generate_diffusion():
    device = 'cpu'
    ckpt = torch.load('weights/diffusion.pth', map_location='cpu')
    unet = UNet(32, 3, 64)
    model = DiffusionModel(unet).to(device)
    model.network.load_state_dict(ckpt['network'])
    model.ema_network.load_state_dict(ckpt['network'])   # load trained weights into BOTH
    model.set_normalizer(ckpt['mean'].to(device), ckpt['std'].to(device))
    out = model.generate(num_images=1, diffusion_steps=20, image_size=32)
    return StreamingResponse(_tensor_to_png(out[0]), media_type="image/png")


CNN_WEIGHTS = Path(__file__).parent.parent / "weights" / "cnn_cifar10.pth"
CNN_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CNN_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)


@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    if not CNN_WEIGHTS.exists():
        raise HTTPException(
            status_code=503,
            detail="Classifier weights not found. Run train_cnn.py first.",
        )
    model = CNN()
    model.load_state_dict(torch.load(CNN_WEIGHTS, map_location="cpu"))
    model.eval()

    try:
        img = Image.open(io.BytesIO(await file.read())).convert("RGB").resize((64, 64))
    except Exception:
        raise HTTPException(status_code=422, detail="Could not read image file.")

    x = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float() / 255.0
    x = ((x - CNN_MEAN) / CNN_STD).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
    idx = int(probs.argmax())
    return {
        "label": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
        "scores": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
    }
