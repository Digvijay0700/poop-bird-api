# Use a slim Python image (faster cold start)
FROM python:3.10-slim

# System deps for torch/torchvision (Pillow needs some libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy files first (to cache pip layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest
COPY . /app

# (Optional) speedups: set threads lower on tiny CPUs
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Expose the port Spaces expects (7860 by convention)
EXPOSE 7860

# Gunicorn to serve Flask, binding to 0.0.0.0:7860
CMD ["gunicorn", "-w", "1", "-k", "gthread", "--threads", "8", "-b", "0.0.0.0:7860", "app:app"]
