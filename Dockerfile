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

# Create venv and install dependencies (so uv run finds it at startup)
RUN uv venv /app/.venv && uv pip install --python /app/.venv/bin/python -e .

# Copy source code
COPY src/ ./src/
COPY amber-manifest.json5 ./

# Expose the A2A server port
EXPOSE 9009

CMD ["uv", "run", "--no-sync", "python", "src/server.py", "--host", "0.0.0.0", "--port", "9009"]
