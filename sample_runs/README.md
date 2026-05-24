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

Regenerate any of these with:

```bash
# hydrate prints the FIRM_INITIAL_POSITIONS line below — set it so the
# report's reconcile block sees the broker holding the same positions
# as the hydrated DB (otherwise the block renders ❌ MISMATCH).
python scripts/hydrate_sample_db.py --date YYYY-MM-DD --out /tmp/firm.db
# e.g. for 2024-03-13: export FIRM_INITIAL_POSITIONS='{"AAPL":"100"}'
FIRM_DB_PATH=/tmp/firm.db FIRM_REPORTS_ROOT=sample_runs \
FIRM_INITIAL_POSITIONS='{"AAPL":"100"}' \
    python -m firm.cli report --date YYYY-MM-DD
python scripts/build_sample_run_readme.py --date YYYY-MM-DD
python scripts/capture_dashboard_png.py --date YYYY-MM-DD \
    --out sample_runs/YYYY-MM-DD/dashboard.png
```
