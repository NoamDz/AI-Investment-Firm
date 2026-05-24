#!/usr/bin/env bash
# End-to-end demo run against the live Anthropic API.
#
# Drives the full Plan 1-4 stack from a clean state:
#   1. Pre-flight  (Qdrant up? deps importable?)
#   2. Ingest      30 FinanceBench docs (vs the 1-doc AAPL fixture in data/demo_run)
#   3. Configure   universe focused on PepsiCo (non-AAPL, well-represented in FB)
#   4. Smoke       --once heartbeat
#   5. Loop        30 heartbeats at 10s intervals (~5 min wall)
#   6. Validate    scripts/e2e_demo_validate.py
#   7. Report      make report DATE=<today>
#
# Cost: ~$1 Anthropic spend.  Wall: ~10-15 min.
#
# Required env: ANTHROPIC_API_KEY (live).

set -euo pipefail

VENV_PY="/e/Teeth_Segmentation/venv/Scripts/python.exe"
ART_DIR="data/e2e_demo"
DB_PATH="$ART_DIR/firm.db"
TRACE_DIR="$ART_DIR/traces"
REPORT_DIR_ROOT="$ART_DIR/reports"
N_INGEST_DOCS="${N_INGEST_DOCS:-30}"
N_HEARTBEATS="${N_HEARTBEATS:-30}"
INTERVAL_S="${INTERVAL_S:-10}"

# Make every firm.cli invocation write to the demo artifact dir.
export FIRM_DB_PATH="$DB_PATH"
export FIRM_TRACES_ROOT="$TRACE_DIR"
export FIRM_REPORTS_ROOT="$REPORT_DIR_ROOT"

# Live mode — we want a real demo, not a cassette replay.
export FIRM_LLM_MODE=live
export FIRM_VCR_MODE=disabled
export FIRM_BROKER=FAKE
export FIRM_HMAC_SECRET="00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff"
# Seed a non-zero AMD position so the new-ticker risk check does not
# escalate every single trade on the first tick. AMD is in the alphabetic
# head of the FinanceBench manifest so the N=30 ingest cap reliably
# includes it (earlier we hit PepsiCo, which is not in the first 30 docs).
export FIRM_INITIAL_POSITIONS='{"AMD":"10"}'
# Anchor the clock so price quotes are stable across heartbeats.
export FIRM_REPLAY_AT="${FIRM_REPLAY_AT:-2024-03-13T14:30:00+00:00}"
export QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
banner() { echo; echo "════════════════════════════════════════════════════════════════════"; echo " $*"; echo "════════════════════════════════════════════════════════════════════"; }

mkdir -p "$ART_DIR" "$TRACE_DIR" "$REPORT_DIR_ROOT"

# --------------------------------------------------------------------------
banner "STEP 1/6  pre-flight"
# --------------------------------------------------------------------------
if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERR: ANTHROPIC_API_KEY not set in env" >&2; exit 2
fi
log "Anthropic key set (length: ${#ANTHROPIC_API_KEY})"
log "Qdrant probe: $QDRANT_URL"
curl -sf "$QDRANT_URL/healthz" >/dev/null || { echo "ERR: Qdrant unreachable" >&2; exit 2; }
log "Qdrant healthy"

# --------------------------------------------------------------------------
banner "STEP 2/6  ingest FinanceBench  (N=$N_INGEST_DOCS docs)"
# --------------------------------------------------------------------------
if [ "${SKIP_INGEST:-0}" = "1" ]; then
    log "SKIP_INGEST=1 — assuming Qdrant collection already populated"
else
    log "Starting ingest (this may take 3-10 min — embeddings + BM25 + contextual augment)..."
    "$VENV_PY" -m firm.cli ingest \
        --config config/rag.yaml \
        --max-docs "$N_INGEST_DOCS" \
        --source financebench \
        2>&1 | tee "$ART_DIR/ingest.log" | tail -10
fi

# --------------------------------------------------------------------------
banner "STEP 3/6  configure universe  (AMD-focused — in N=30 ingest head)"
# --------------------------------------------------------------------------
# universe.yaml.bak preserves the original. The first ticker is what
# firm/agents/research.py:293 picks each heartbeat. All three tickers
# below are in the alphabetic head of FinanceBench so the N=30 cap
# reliably gets chunks for them. AMD goes first because we've seeded a
# starting position above.
cp -f config/universe.yaml config/universe.yaml.bak
cat > config/universe.yaml <<'YAML'
as_of: 2023-11-01
tickers:
  - AMD
  - Adobe
  - Boeing
sector_map:
  AMD: tech
  Adobe: tech
  Boeing: industrials
YAML
log "universe.yaml -> [AMD, Adobe, Boeing]"

# --------------------------------------------------------------------------
banner "STEP 4/6  smoke heartbeat  (--once)"
# --------------------------------------------------------------------------
"$VENV_PY" -m firm.cli run --once 2>&1 | tee "$ART_DIR/heartbeat_smoke.log" | tail -15

# --------------------------------------------------------------------------
banner "STEP 5/6  continuous loop  (N=$N_HEARTBEATS heartbeats × ${INTERVAL_S}s)"
# --------------------------------------------------------------------------
# Wrap loop in a background process + timeout-style kill so we cap wall.
# Total cap: heartbeats * interval + 60s slack.
TIMEOUT_S=$(( N_HEARTBEATS * INTERVAL_S + 120 ))
log "Wall-cap: ${TIMEOUT_S}s"

# Use a tick-counter approach: cycle --once N times. This avoids the
# loop-mode signal handling complexity and gives us a deterministic count.
for i in $(seq 1 "$N_HEARTBEATS"); do
    log "tick $i/$N_HEARTBEATS"
    if ! "$VENV_PY" -m firm.cli run --once 2>&1 | tail -3 >> "$ART_DIR/loop.log"; then
        log "  ! tick $i exited non-zero — continuing (loop is crash-resilient by design)"
    fi
    if [ "$i" -lt "$N_HEARTBEATS" ]; then sleep "$INTERVAL_S"; fi
done

# --------------------------------------------------------------------------
banner "STEP 6/6  validate + render daily report"
# --------------------------------------------------------------------------
TODAY="$(date -u +%Y-%m-%d)"
REPORT_DAY_DIR="$REPORT_DIR_ROOT/$TODAY"
mkdir -p "$REPORT_DAY_DIR"

# Render the daily report — make target wraps firm.cli reports.
"$VENV_PY" -m firm.cli report --date "$TODAY" 2>&1 | tail -10 || \
    log "report cmd failed; continuing to validation"

# Locate trace file. The tracer writes to <root>/YYYY-MM-DD/run-<run_id>.jsonl.
# CLI never calls init_tracer(run_id=...), so _DEFAULT_RUN_ID (26 zeros) is used.
TRACE_FILE="$TRACE_DIR/$TODAY/run-00000000000000000000000000.jsonl"
if [ ! -f "$TRACE_FILE" ]; then
    # Fall back to any .jsonl under today's dir, then anywhere under traces.
    TRACE_FILE=$(ls "$TRACE_DIR/$TODAY"/*.jsonl 2>/dev/null | head -1 || \
                 ls "$TRACE_DIR"/*/*.jsonl 2>/dev/null | head -1 || echo "")
fi
log "trace file: $TRACE_FILE"

"$VENV_PY" scripts/e2e_demo_validate.py \
    --db "$DB_PATH" \
    --trace "$TRACE_FILE" \
    --report-dir "$REPORT_DAY_DIR" \
    | tee "$ART_DIR/validation_report.txt"

banner "DONE"
log "Artifacts in: $ART_DIR/"
log "Validation:   $ART_DIR/validation_report.txt"

# Restore the original universe.yaml.
mv -f config/universe.yaml.bak config/universe.yaml
log "Restored config/universe.yaml from .bak"
