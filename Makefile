# GnuWin32 Make defaults to cmd.exe; force bash so recipes use POSIX syntax.
SHELL := bash

.PHONY: install test demo demo-docker reconcile ingest report clean litestream-drill check-determinism red-team eval deploy-dev

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

check-determinism:
	bash scripts/check_reports_clean.sh

red-team:
	python -m firm.cli red-team

# REGIME (optional) lets the caller scope `make eval` to a single regime
# (e.g. `make eval REGIME=r1` for PR-speed CI). When unset, firm.cli eval
# runs the full r1+r2+r3 sweep — preserving the historical default for
# `make eval` callers that don't pass REGIME.
#
# The `$(if $(REGIME),...)` guard treats an empty assignment (`REGIME=`)
# as falsy and emits no `--regime` flag — same as unset. Intentional: CI
# callers can pass `REGIME=` to explicitly request the full sweep without
# editing the recipe.
eval:
	FIRM_LLM_MODE=$${FIRM_LLM_MODE:-cached} \
	FIRM_VCR_MODE=$${FIRM_VCR_MODE:-replay} \
	FIRM_PRICES_MODE=$${FIRM_PRICES_MODE:-replay} \
	FIRM_RANDOM_SEED=$${FIRM_RANDOM_SEED:-42} \
	FIRM_HMAC_SECRET=$${FIRM_HMAC_SECRET:-$$(printf '00%.0s' {1..32})} \
	FIRM_EVAL_SKIP_MISCONFIG=$${FIRM_EVAL_SKIP_MISCONFIG:-1} \
	python -m firm.cli eval $(if $(REGIME),--regime $(REGIME))

deploy-dev:  ## Apply Terraform against the dev environment (HUMAN-GATED — creates real AWS resources)
	@echo ""
	@echo "============================================================"
	@echo "WARNING: This will create real AWS resources in your account."
	@echo "  - ECS Fargate cluster (~$$15/mo)"
	@echo "  - RDS Postgres db.t4g.micro (~$$15/mo)"
	@echo "  - NAT Gateway (~$$32/mo + per-GB egress)"
	@echo "  - S3 buckets, CloudWatch logs, Secrets Manager (negligible idle)"
	@echo ""
	@echo "Run 'terraform -chdir=infra/terraform destroy -var-file=envs/dev.tfvars'"
	@echo "when done to avoid ongoing charges."
	@echo "============================================================"
	@echo ""
	@read -p "Type 'DEPLOY' to continue: " confirm; \
	  if [ "$$confirm" != "DEPLOY" ]; then \
	    echo "Aborted (entered '$$confirm' instead of 'DEPLOY')."; \
	    exit 1; \
	  fi
	terraform -chdir=infra/terraform apply -var-file=envs/dev.tfvars -auto-approve
