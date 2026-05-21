# GnuWin32 Make defaults to cmd.exe; force bash so recipes use POSIX syntax.
SHELL := bash

.PHONY: install test demo demo-docker reconcile ingest report clean litestream-drill

install:
	pip install -e ".[dev]"

test:
	pytest

demo:
	FIRM_REPLAY_AT=2024-03-13T14:30:00+00:00 \
	FIRM_HMAC_SECRET=$$(python -c "import secrets; print(secrets.token_hex(32))") \
	python -m firm.cli run --once

demo-docker:
	docker compose up --build --abort-on-container-exit

reconcile:
	FIRM_HMAC_SECRET=$${FIRM_HMAC_SECRET:-placeholder} python -m firm.cli reconcile

ingest:
	FIRM_LLM_MODE=$${FIRM_LLM_MODE:-cached} python -m firm.cli ingest

report:
	python -m firm.cli report --date $(DATE)

litestream-drill:
	python scripts/litestream_drill.py

clean:
	rm -rf data/firm.db data/firm.db-wal data/firm.db-shm data/reports data/litestream
