FROM python:3.11-slim

# Runtime dependencies used by the CLI and Claude Code SDK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package first from the standalone repository checkout.
COPY pyproject.toml README.md VERSION ./
COPY dokumen ./dokumen
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Claude Code is required by claude-agent-sdk for executor and judge runs.
RUN npm install -g @anthropic-ai/claude-code

RUN useradd --create-home --shell /bin/bash dokumen \
    && mkdir -p /workspace \
    && chown -R dokumen:dokumen /workspace /app

USER dokumen
WORKDIR /workspace

CMD ["dokumen", "--help"]
