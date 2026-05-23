FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./

# Install CPU-only torch first (the CUDA wheel is ~2 GB; CPU is ~200 MB).
# sentence-transformers depends on torch but will accept whichever is already installed.
RUN pip install --no-cache-dir --timeout 600 --retries 5 \
    --index-url https://download.pytorch.org/whl/cpu \
    torch

RUN pip install --no-cache-dir --timeout 600 --retries 5 -e ".[dev]"

COPY firm/ ./firm/
COPY config/ ./config/
# scripts/ folded into firm/ops/; COPY firm/ above already pulls them in.
COPY data/precomputed/ ./data/precomputed/

ENV FIRM_DB_PATH=/data/firm.db
ENV FIRM_REPORTS_ROOT=/data/reports
ENV FIRM_BROKER=FAKE
ENV HF_HOME=/data/hf_cache

CMD ["python", "-m", "firm.cli", "run", "--once"]
