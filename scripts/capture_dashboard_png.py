"""Capture a PNG of the Streamlit dashboard's Tab 1 ("Today's Report").

Documented in ``docs/PLAN_reports_overhaul.md`` §5 step 4. The script:

1. Sets ``FIRM_REPORTS_ROOT`` / ``FIRM_SAMPLE_RUNS_ROOT`` / ``FIRM_DB_PATH``
   so the dashboard sees the committed sample bundle and an empty live DB.
2. Launches ``streamlit run firm/dashboard.py --server.headless true`` on
   the requested port.
3. Polls ``/_stcore/health`` until OK or a 20 s timeout elapses.
4. Drives a headless Chromium via Playwright, picks the requested date from
   the dashboard's date selectbox, and saves a 1600x900 PNG of Tab 1.
5. Kills the Streamlit subprocess.

Playwright is an optional dependency — the script exits 0 with a clear
``[capture]`` log message when Playwright is missing or the Chromium binary
hasn't been downloaded, so the surrounding regeneration workflow can keep
going and surface a ``dashboard.png.MISSING`` placeholder instead.

CLI:
    python scripts/capture_dashboard_png.py \\
        --date YYYY-MM-DD --out sample_runs/<date>/dashboard.png [--port 8501]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HEALTH_TIMEOUT_S = 20.0
HEALTH_POLL_S = 0.5
RENDER_SETTLE_S = 2.0
VIEWPORT_WIDTH = 1600
VIEWPORT_HEIGHT = 900


def _poll_health(port: int, timeout: float) -> bool:
    """Return True once /_stcore/health responds 200, or False after timeout."""
    url = f"http://localhost:{port}/_stcore/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def _spawn_streamlit(port: int, env: dict[str, str]) -> subprocess.Popen[bytes]:
    """Start a headless Streamlit subprocess. Returns the Popen handle."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "firm/dashboard.py",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _capture_with_playwright(port: int, date_str: str, out_path: Path) -> str:
    """Drive Chromium to render Tab 1 and save the PNG.

    Returns a short status string. Raises ImportError when Playwright is not
    installed; raises FileNotFoundError / Exception subclasses for browser
    binary or runtime issues (caller maps these to exit 0 with a log line).
    """
    from playwright.sync_api import sync_playwright  # may raise ImportError

    with sync_playwright() as pw:
        # Chromium binary may be missing — sync_playwright().__enter__ succeeds
        # but launch() raises an Error pointing at `playwright install`.
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT}
        )
        page = context.new_page()
        page.goto(f"http://localhost:{port}", wait_until="networkidle")

        # Wait for the dashboard's tabs to render. The Tab 1 label "Today's
        # Report" is the most reliable signal the script is interactive.
        page.wait_for_selector("text=Today's Report", timeout=15000)

        # Pick the requested date. Streamlit's selectbox is a custom combobox.
        try:
            page.get_by_label("Date").click(timeout=5000)
            page.get_by_role("option", name=date_str).click(timeout=5000)
            page.wait_for_load_state("networkidle")
        except Exception:
            # If the selectbox already shows the requested date (e.g. only one
            # date is available) the click sequence may fail — fall through
            # and screenshot whatever is currently rendered.
            pass

        # Let the embedded HTML iframe settle.
        time.sleep(RENDER_SETTLE_S)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(
            path=str(out_path),
            full_page=False,
            clip={
                "x": 0,
                "y": 0,
                "width": VIEWPORT_WIDTH,
                "height": VIEWPORT_HEIGHT,
            },
        )

        browser.close()

    return f"wrote {out_path}"


def _build_env() -> dict[str, str]:
    """Build the env passed to Streamlit. Keeps Live Desk empty."""
    env = os.environ.copy()
    # Empty in-memory-style live DB so the Live Desk tab shows the empty
    # state — Tab 1 is the only thing we screenshot. ":memory:" doesn't work
    # for Streamlit's separate connection, so use a tmp path that init_db
    # will create lazily on first access.
    env.setdefault("FIRM_DB_PATH", str(REPO_ROOT / "data" / "_capture_tmp.db"))
    env.setdefault("FIRM_REPORTS_ROOT", str(REPO_ROOT / "data" / "reports"))
    env.setdefault("FIRM_SAMPLE_RUNS_ROOT", str(REPO_ROOT / "sample_runs"))
    env.setdefault("FIRM_TRACES_ROOT", str(REPO_ROOT / "data" / "traces"))
    return env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Capture a 1600x900 PNG of dashboard Tab 1 for a given date."
    )
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path (e.g. sample_runs/2024-03-13/dashboard.png).",
    )
    parser.add_argument("--port", default=8501, type=int)
    args = parser.parse_args(argv)

    out_path = Path(args.out)

    # Try to import Playwright BEFORE spinning up Streamlit so the failure
    # path is cheap.
    try:
        import playwright  # noqa: F401
    except ImportError:
        print(
            "[capture] Playwright not installed. Run "
            "`pip install playwright && playwright install chromium`. "
            "PNG not captured.",
            file=sys.stderr,
        )
        return 0

    env = _build_env()
    proc = _spawn_streamlit(args.port, env)
    try:
        if not _poll_health(args.port, HEALTH_TIMEOUT_S):
            print(
                f"[capture] Streamlit health check timed out after "
                f"{HEALTH_TIMEOUT_S:.0f}s on port {args.port}. PNG not captured.",
                file=sys.stderr,
            )
            return 0

        try:
            status = _capture_with_playwright(args.port, args.date, out_path)
        except ImportError:
            # Re-raise path covered above, but keep this defensive in case the
            # second import (inside _capture_with_playwright) fails.
            print(
                "[capture] Playwright not installed. PNG not captured.",
                file=sys.stderr,
            )
            return 0
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if "Executable doesn't exist" in msg or "playwright install" in msg:
                print(
                    "[capture] Chromium binary missing. Run "
                    "`playwright install chromium`. PNG not captured.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"[capture] Playwright capture failed ({type(exc).__name__}): "
                    f"{exc}. PNG not captured.",
                    file=sys.stderr,
                )
            return 0

        print(status)
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    sys.exit(main())
