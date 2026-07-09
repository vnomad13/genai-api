FROM python:3.12-slim

WORKDIR /workspace

RUN pip install uv

COPY pyproject.toml .
COPY app/ ./app/

RUN uv pip install --system --no-cache .
RUN python -m spacy download en_core_web_md

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
