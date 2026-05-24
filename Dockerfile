FROM python:3.11-slim

# Install system dependencies useful across benchmarks
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    jq \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency file first for caching
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system -e .

# Copy source code
COPY src/ ./src/
COPY amber-manifest.json5 ./

WORKDIR /app/src

# Expose the A2A server port
EXPOSE 9009

CMD ["python", "server.py"]
