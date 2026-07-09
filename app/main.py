import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
from pydantic import BaseModel

from app.embedding_model import get_word_vector
from app.gan_model import generate_sample, load_generator

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
