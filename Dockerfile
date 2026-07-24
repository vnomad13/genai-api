FROM python:3.12-slim

WORKDIR /workspace

RUN pip install --no-cache-dir uv

# CPU-only torch/torchvision first: large and rarely changes, so this layer stays cached
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Whole project: app/, weights/, and the course-reference model modules at the root
# (energy_model.py / diffusion_model.py are imported directly by app/main.py)
COPY . .

# Remaining deps come from pyproject.toml: fastapi, uvicorn, spacy, pillow, numpy
RUN uv pip install --system --no-cache .
RUN python -m spacy download en_core_web_md

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
