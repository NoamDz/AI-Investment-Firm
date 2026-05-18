FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"

COPY firm/ ./firm/
COPY config/ ./config/
COPY scripts/ ./scripts/

ENV FIRM_DB_PATH=/data/firm.db
ENV FIRM_REPORTS_ROOT=/data/reports
ENV FIRM_BROKER=FAKE

CMD ["python", "-m", "firm.cli", "run", "--once"]
