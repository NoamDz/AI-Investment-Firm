# Sample runs

Three regime snapshots committed for reviewer inspection. Start with the per-date `README.md` for the narrated walkthrough.

| Date | Regime | What this day demonstrates |
| --- | --- | --- |
| [2024-03-13](2024-03-13/README.md) | earnings_heavy | Earnings-day BUY/HOLD flow |
| [2024-08-07](2024-08-07/README.md) | vol_spike | Volatility-spike decision behaviour |
| [2023-11-08](2023-11-08/README.md) | pre_news | Pre-CPI flat day |

Each per-date directory contains:
- `README.md` - narrated walkthrough (start here)
- `dashboard.png` - Tab 1 (Today's Report) screenshot, captured via `scripts/capture_dashboard_png.py`
- `daily_report.html` - open in browser, save as PDF via Ctrl-P
- `daily_report.md` - legacy plain-text summary
- `positions.xlsx` - Positions / P&L / Decisions sheets
- `decisions.jsonl`, `trace.jsonl` - raw artifacts

Regenerate any of these bundles from the committed `decisions.jsonl` + `trace.jsonl` — useful after a template/format change so the HTML / xlsx / per-date README pick up the new shape without re-running the firm. The committed JSONL files are the source of truth; everything else is a derived view.

```bash
# 1. Rebuild the temp firm.db. Hydrate prints a FIRM_INITIAL_POSITIONS line
#    matching the date's seeded positions — copy that line verbatim into
#    step 2 so the report's reconcile block renders ✓ instead of ❌ MISMATCH.
python scripts/hydrate_sample_db.py --date YYYY-MM-DD --out /tmp/firm.db
# e.g. for 2024-03-13 it prints:
#   FIRM_INITIAL_POSITIONS={"AAPL":{"shares":"100","avg_cost":"201.20"}}
# The object form ({shares, avg_cost}) seeds an explicit cost basis so the
# Positions sheet shows non-zero unrealized P&L; the legacy shares-only form
# ('{"AAPL":"100"}') still works but leaves avg_cost = current quote → P&L 0.

# 2. Render daily_report.md / daily_report.html / positions.xlsx into the bundle.
FIRM_DB_PATH=/tmp/firm.db FIRM_REPORTS_ROOT=sample_runs \
FIRM_INITIAL_POSITIONS='{"AAPL":{"shares":"100","avg_cost":"201.20"}}' \
    python -m firm.cli report --date YYYY-MM-DD

# 3. Rebuild the per-date README's decisions table + "walking one trade" block.
python scripts/build_sample_run_readme.py --date YYYY-MM-DD

# 4. (Optional) re-capture the dashboard screenshot. Needs the Streamlit
#    dashboard running locally on http://localhost:8501.
python scripts/capture_dashboard_png.py --date YYYY-MM-DD \
    --out sample_runs/YYYY-MM-DD/dashboard.png
```
