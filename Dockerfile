# Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system libs for faiss & sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev git wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip
# Force install CPU-only torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

# Bake the model into the image
COPY download_model.py .
RUN python download_model.py
ENV MODEL_NAME="/app/model_cache"

# Force cache bust for index update
COPY . /app

EXPOSE 8080

CMD ["uvicorn", "detector.main:app", "--host", "0.0.0.0", "--port", "8080"]
