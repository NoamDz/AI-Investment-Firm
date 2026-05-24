# Plan — Report-Delivery Overhaul

Status: **proposal — not yet approved.** Drafted 2026-05-24.

## 1. Problem statement

- The README sells the Streamlit dashboard as a co-equal observability tool while the brief asks for a *report-delivery* channel; today the dashboard's framing and layout emphasise live monitoring (cash/exposure metrics, "Recent decisions") and the daily report itself is buried as a `download_button` in the footer captioned "channel #2".
- The "Daily report bundle" is positioned as channel #2 but it is an on-disk directory of `.md` / `.jsonl` files — that is the *artifact*, not a delivery channel a stakeholder receives.
- Sample runs are an uncurated dump of four files per date with no narration, no rendered view, no entry-point README, no annotated trace excerpt, and no way for a reviewer to see *what happened that day* without reading raw JSONL line-by-line.
- The existing Markdown report is plain text dressed as `.md` (no headings, no tables) and contains no per-decision detail — the per-decision data lives in `decisions.jsonl` only.
- `positions.xlsx` exists and is real, but it is invisible unless the reviewer opens the dashboard and clicks "Download".
- README §"Two report channels, one source of truth" must be rewritten once channels change.

## 2. Channels chosen

**Channel A — Streamlit dashboard (real-time, in-browser).** Already shipped, runs on `localhost:8501` with zero external infrastructure. After redesign (see §3) its top-of-page is the **rendered daily report** itself; live observability moves below the fold. This satisfies the "real-time monitoring" half of the brief's example list (`web dashboard`).

**Channel B — Self-contained HTML report file (`daily_report.html`) + Excel workbook (`positions.xlsx`), both written by the reporter and opened by the dashboard "Download report bundle" button.** The HTML report is a single file with inline CSS — no JS, no external assets — that renders exactly like an email body would. The reviewer double-clicks it on their laptop and gets a printable PDF-like view. This satisfies the "durable handoff" half of the brief's example list. We deliberately frame the HTML file as "the email body" we *would* send if SMTP were wired; the Excel workbook is its attachment.

**Why these two cover the brief.**
- *Real-time vs. durable*: Dashboard is live (5 s refresh, useful while the firm is running); HTML+XLSX is frozen at EOD (useful for off-line review, archiving, attaching to a real email later).
- *Different audiences*: Operators who want to dig (dashboard) vs. stakeholders who want a one-page summary they can forward (HTML).
- *Implementable on the reviewer's laptop*: No SMTP, no Slack workspace, no S3 bucket, no GitHub Pages publish step. Both run inside `docker compose up`.
- *Both read the same source*: `firm.db` + the reporter's already-written per-date directory under `data/reports/` (or `sample_runs/`). They cannot disagree.

**Rejected alternatives.**
- *Slack channel* — requires a workspace + bot token + ngrok tunnel for the interactive endpoint; reviewer cannot exercise it cold.
- *Real email (SMTP)* — requires SMTP creds and a recipient; we keep the HTML body as a stand-in.
- *PDF via `weasyprint`/`reportlab`* — adds heavy native deps (Cairo/Pango); HTML is sufficient and printable to PDF from any browser. Documented in the README as the "print to PDF" path.
- *GitHub Pages publish* — externalises the channel; reviewer would need to wait for CI to publish.
- *RSS feed* — over-engineered for a paper-trading take-home.

## 3. Dashboard role redefinition

New page layout — top-to-bottom, with a **selectable date** at the top so the dashboard works for both live runs and committed sample runs.

```
+--------------------------------------------------------------+
|  AI Investment Firm — Daily Report                           |
|  Date: [▼ 2024-03-13]   Source: firm.db / sample_runs/...    |
+--------------------------------------------------------------+
|  TAB 1: Today's Report   |  TAB 2: Live Desk  |  TAB 3: Trace |
+--------------------------------------------------------------+
```

**Tab 1 — Today's Report (DEFAULT, opens here).** Renders the same content as the HTML report (single source). Sections in this order:
1. Header banner: date, regime label (if known), one-sentence narrative ("5 decisions: 2 BUY, 1 SELL, 1 HOLD, 1 ESCALATE; books tied to broker.").
2. Decision table — every decision for the date as a sortable `st.dataframe`: `ts | action | ticker | shares | confidence | citations | failure_mode | rationale`. (Today this data is only in `decisions.jsonl`; the daily report only carries counts.)
3. Cost summary — same block as today's `daily_report.md`, but rendered as `st.metric` chips.
4. Reconciliation — broker vs. local positions side-by-side, colored ✓/❌ row.
5. **Download bundle** prominently in this tab — `daily_report.html`, `positions.xlsx`, `decisions.jsonl`, `trace.jsonl`, all four as separate `st.download_button`s with sizes shown.

**Tab 2 — Live Desk (the current dashboard, demoted).** All existing widgets stay: cash/exposure/spend/cache-hit metric chips, positions table, HITL queue, recent-25 decisions card stream, reconciliation status. Labelled clearly as "Live observability — updates every 5 s while `firm run --loop` is running."

**Tab 3 — Trace.** A `text_input` for `decision_id` plus an `st.dataframe` view of the matching spans (parent_span_id → tree). One-click "Copy `grep '<id>' trace.jsonl | jq .name` command" button. Satisfies brief §4: "A reviewer should be able to replay a trade end-to-end from the trace alone."

**What moves vs. what stays:**
- *Stays in place (in Tab 2)*: `_read_cash`, `_read_positions`, `_read_decisions`, `_read_hitl`, `_read_cost_today`, `_read_recon`, the 5-second auto-refresh.
- *Moves up (to Tab 1)*: `_latest_xlsx` download button, plus new per-date readers that key off the selectbox value rather than `today`.
- *New readers*: `_read_decisions_for_date(date)`, `_read_cost_for_date(date)`, `_read_recon_for_date(date)` (the "today" variants already exist — generalize the SQL filter).
- *New source switch*: a `pathlib.Path` resolver that prefers `data/reports/<date>/` for live runs but falls back to `sample_runs/<date>/` so the dashboard can render committed snapshots even with an empty `firm.db`.

## 4. The other channel — `daily_report.html` + `positions.xlsx`

**`daily_report.html`** — one self-contained file, no external assets, written by the reporter into the same `data/reports/<date>/` directory next to the existing `.md`. Layout, top-to-bottom:

1. **Header bar** (dark band): "AI Investment Firm — Daily Report — 2024-03-13 (r1_earnings)".
2. **Executive summary** (1 paragraph, auto-generated from counts): "On 2024-03-13 the firm took 5 decisions: 2 BUY, 1 SELL, 1 HOLD, 1 ESCALATE. Total LLM spend $0.007 (17% cache hit). Books tied to broker."
3. **Decisions table** — `<table>` with columns: `Time | Action | Ticker | Shares | Confidence | Citations | Failure mode | Rationale (truncated 120 chars)`. Action cells coloured using the same `ACTION_COLORS` palette already in `dashboard.py`. Rows striped.
4. **Cost summary block** — same model-grouped table as `daily.py:_render_cost_block`, but as `<table>`.
5. **Reconciliation block** — broker dict vs. local dict, position diff highlighted in red if non-empty.
6. **Trace pointer** — small footer block with: `data/traces/<date>/run-*.jsonl` path, and a literal `grep '"decision_id":"<id>"' ...` command per decision_id (one per BUY/SELL).
7. **Footer**: "Generated 2024-03-13 by `firm.reports.html`. View live: http://localhost:8501. To save as PDF: Ctrl-P → Save as PDF."

Single CSS block in `<head>` (≤80 lines of CSS, no JS). Determinism: the writer must avoid wall-clock timestamps in the footer — use the eval date or the reporter's `Clock.now()` already used elsewhere; pass the same epoch into the template.

**`positions.xlsx`** — keep the two-sheet shape (`Positions`, `P&L`) already in `firm/reports/xlsx.py`. Add one new sheet:
- **`Decisions`** — same columns as the HTML table, so the spreadsheet stands alone if someone forwards just the xlsx. Sort by `ts ASC`. Header row bolded; freeze panes at row 2.

No styling changes needed beyond bold header row + freeze panes (openpyxl one-liners). Deterministic since rows come from the DB in `ORDER BY created_at`.

## 5. Sample-runs overhaul

Per-date `sample_runs/<date>/README.md` becomes the *only* file a reviewer needs to open. Required sections:

1. **What this day demonstrates** — 3 lines.
   - Regime label and why this date was picked.
   - What the firm should do (e.g. "Earnings-heavy: research has signal; expect at least one BUY.").
   - What to look for (e.g. "ESCALATE row triggered by risk_limit_breached — see decision `dec-buy-1`.").
2. **Decisions table** — committed inline as Markdown, 5–10 rows max:

   ```
   | ts | action | ticker | shares | conf | citations | failure_mode | rationale |
   ```
   Generated by a small script (`scripts/build_sample_run_readme.py`) that reads `decisions.jsonl` and writes the table; committed alongside the data so the reviewer never has to run it.
3. **One annotated trace excerpt** — pick the most interesting `decision_id` for the day and paste ~10 spans inline in a fenced ```jsonl block, with a leading sentence: "Walk this trade with: `grep '\"decision_id\":\"dec-buy-1\"' trace.jsonl | jq .name`. The 10 spans below are the full chain."
4. **Rendered dashboard screenshot** — `dashboard.png` committed at `sample_runs/<date>/dashboard.png`. Generated once via a Playwright capture script `scripts/capture_dashboard_png.py` that:
   - Sets `FIRM_DASHBOARD_DATE=<date>` and `FIRM_REPORTS_ROOT=sample_runs`
   - Starts streamlit, waits for `:8501`, takes a 1600×900 PNG of Tab 1, kills streamlit.
   - Documented but not run in CI (would need a browser binary); committed PNG is the source of truth. The script is the "how I made this" so a reviewer trusts it.
5. **Channel links** — bulleted:
   - `daily_report.html` (open in browser) ← primary
   - `positions.xlsx` (open in Excel)
   - `daily_report.md` (legacy plain-text)
   - `decisions.jsonl`, `trace.jsonl` (raw)

Also add a top-level `sample_runs/README.md` (the index) with the three regime rows from the existing README "Sample run committed" table, each linking to the per-date README.

## 6. Implementation tasks (≤8, ordered by dependency)

### T1 — Add HTML report template + writer (~80 LOC)
- **Files**: `firm/reports/templates/daily_report.html.j2` (new), `firm/reports/html.py` (new, ~60 LOC).
- **What**: New `render_daily_html(date, db_path, broker, reports_root, reconcile_block)` mirroring `render_daily_report` in `daily.py` but emitting HTML. Reads same `decisions` + `cost_ledger` rows. Inline CSS, no JS. Use `keep_trailing_newline=True` and explicit `encoding="utf-8", newline="\n"` to keep determinism.
- **Verify**: `python -m firm.cli report --date 2024-03-13` produces `data/reports/2024-03-13/daily_report.html`; open in browser; `bash firm/ops/check_reports_clean.sh` still passes.

### T2 — Wire HTML writer into the `report` CLI + `make report` (~10 LOC)
- **Files**: `firm/cli.py` (after `write_positions_xlsx(...)` block near line 645).
- **What**: Add `render_daily_html(...)` call alongside the existing `render_daily_report` and `write_positions_xlsx` calls. No new flags.
- **Verify**: `make report DATE=2024-03-13` lists all three artifacts in the "Report bundle written:" message.

### T3 — Add `Decisions` sheet to `positions.xlsx` (~20 LOC)
- **Files**: `firm/reports/xlsx.py`.
- **What**: After the `P&L` sheet, create a third sheet `Decisions` with the columns spec'd in §4. Pull rows with the same `SELECT ... FROM decisions WHERE created_at < ? ORDER BY created_at ASC` query already there; widen the projection to include `confidence`, `citations` (JSON length), `failure_mode`, `rationale`. Bold header row via `Font(bold=True)`; `ws.freeze_panes = "A2"`.
- **Verify**: Open the xlsx in Excel; new sheet is present; rows match `decisions.jsonl` count; openpyxl save is deterministic (already verified for the existing two sheets — same code path).

### T4 — Dashboard restructure into 3 tabs (~120 LOC, the largest change)
- **Files**: `firm/dashboard.py` (rewrite `render()`).
- **What**:
  - Add date selectbox at top: `st.selectbox` over `sorted(REPORTS_ROOT.glob("*"))` plus `sample_runs/*`, default = latest.
  - Wrap existing widget tree in `tab1, tab2, tab3 = st.tabs(["Today's Report", "Live Desk", "Trace"])`.
  - Tab 1 reads the chosen date's `daily_report.html` and shows it via `st.components.v1.html(...)` plus 4 download buttons.
  - Tab 2 = the current `render()` body unchanged (gives "Live Desk" feel) — but reads cash/positions live from `firm.db`.
  - Tab 3 = trace search input + dataframe view.
  - Disable auto-refresh on Tab 1 (only refresh Tab 2). Easiest: keep the existing `st.rerun()` loop but only inside Tab 2's container.
- **Verify**: `streamlit run firm/dashboard.py` opens on Tab 1 showing the report; switch to Tab 2 sees live data; switch to Tab 3, type a `decision_id`, see span list.

### T5 — `scripts/build_sample_run_readme.py` (~70 LOC)
- **Files**: new file under `scripts/`.
- **What**: Takes `--date YYYY-MM-DD`, reads `sample_runs/<date>/decisions.jsonl` + `trace.jsonl`, picks the highest-citation decision_id, writes `sample_runs/<date>/README.md` in the layout from §5. Idempotent — running twice produces the same bytes.
- **Verify**: Run for all three dates; `git diff --stat` shows three new READMEs; running again produces no diff.

### T6 — Commit sample-run READMEs and dashboard PNGs (manual one-time)
- **Files**: `sample_runs/2024-03-13/README.md`, `sample_runs/2024-08-07/README.md`, `sample_runs/2023-11-08/README.md`, three `dashboard.png`s, `sample_runs/README.md` (index).
- **What**: Run T5 script for each date; capture three PNGs via the procedure in §5 (script committed at `scripts/capture_dashboard_png.py`, ≤40 LOC, Playwright-based — not run in CI).
- **Verify**: Each per-date directory listing shows `README.md` + `dashboard.png` alongside the four data files. `git log` of those files shows one commit.

### T7 — README rewrite (see §7) (~50 LOC of prose changes)
- **Files**: `README.md`.
- **What**: Sections enumerated in §7.
- **Verify**: README references to `daily_report.md` no longer appear in the "channels" section; references to `dashboard.png` resolve; `markdown-lint` (if hooked) passes.

### T8 — Quickstart doc patch (~15 LOC)
- **Files**: `docs/quickstart.md` (§"Generate a daily report" near line 134).
- **What**: Replace "Writes `data/reports/2024-03-13/daily_report.md` … and `positions.xlsx`" with the new three-artifact list, plus a one-line "Open `daily_report.html` in your browser" call-out.
- **Verify**: Manual proofread.

**Total LOC budget**: ~365 source + ~150 docs/READMEs ≈ within 500 LOC.

## 7. README sections to rewrite

Once the plan is implemented, these exact README headings need updating (do **not** rewrite now — list only):

1. `### Two report channels, one source of truth` — rename to `### Two report channels — dashboard and bundle`, rewrite bullets so dashboard is *the daily report* and the bundle (HTML + XLSX) is the durable handoff. Drop the `.md` from the channel description; mention it as legacy.
2. `### Observability — replay any trade from the trace` — mention the new dashboard Tab 3 "Trace" search.
3. `### Sample run committed` — rewrite the bullet list under "Each directory contains:" to lead with `README.md` (the per-date walkthrough), then `dashboard.png`, then the raw artifacts. Add a sentence: "Start with `README.md` for the narrated walkthrough."
4. `![Dashboard](docs/images/dashboard.png)` — confirm the asset exists. Either commit one or remove the line. (We already removed this in the current README; re-add once captured.)
5. `## Quickstart` — add a line under Terminal 2 noting "the dashboard opens on the *Today's Report* tab; switch to *Live Desk* for the streaming view."

## 8. Out of scope

- **No real email delivery.** No SMTP integration, no `aiosmtpd`, no Postmark/SendGrid. The HTML file is the email body we *would* send.
- **No Slack OAuth or workspace.** The existing optional Slack notifier stays as-is.
- **No production observability stack.** No Honeycomb, no Grafana, no Loki. OTLP export remains opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT`.
- **No PDF library.** Users print HTML → PDF from the browser.
- **No new agent.** The reporter agent gains one extra call (`render_daily_html`); no graph-shape change.
- **No CI capture of dashboard PNGs.** Committed PNGs are the source of truth; the capture script is documented but not pipeline-gated (would need a browser binary in CI).
- **No change to `daily_report.md` format.** Kept for back-compat; downgraded to "legacy plain-text" in the README so existing golden files / determinism tests don't churn.
- **No change to `firm.db` schema.**

---

## Biggest design risk

The new tabbed dashboard plus a date selectbox that switches between `data/reports/` (live) and `sample_runs/` (frozen) could quietly break the determinism gate if the HTML writer leaks a wall-clock timestamp — must thread `Clock.now()` through the template the same way `reporter.py` already does, and add the HTML to `check_reports_clean.sh`'s diff scope.
