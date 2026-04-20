"""
alerts.py — Alert system for the Taoist CIO dashboard.

Compares current signal state against the previous run's state, fires alerts
for anything that changed, and appends them to logs/alerts.log.

Alert levels:
  MAX_ALERT — S-1 filed (Ji Moment)
  HIGH      — CONVERGENCE detected, Wu Wei Entry triggered
  MEDIUM    — Forge price moved >10%, any threshold status change
  LOW       — Routine state updates (logged only)

Run with: python src/alerts.py
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

ALERT_LOG_FILE  = LOGS_DIR / "alerts.log"
STATE_FILE      = DATA_DIR / "last_alert_state.json"

sys.path.insert(0, str(ROOT / "src"))

console = Console()

# ── Named file logger — only captures our own writes, not urllib3/requests ────
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

_file_log = logging.getLogger("taoist_alerts")
_file_log.setLevel(logging.DEBUG)
_file_log.propagate = False   # don't bubble to root logger
if not _file_log.handlers:
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        _fh = logging.FileHandler(LOGS_DIR / "alerts.log")
        _fh.setFormatter(logging.Formatter("%(message)s"))
        _file_log.addHandler(_fh)
    except (OSError, PermissionError):
        _file_log.addHandler(logging.StreamHandler())

# ── Level styles ───────────────────────────────────────────────────────────────
LEVEL_STYLE = {
    "MAX_ALERT": "bold white on red",
    "HIGH":      "bold red",
    "MEDIUM":    "bold yellow",
    "LOW":       "dim",
}
LEVEL_RANK = {"MAX_ALERT": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}


# ── JSON helpers ───────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _now_local_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_state() -> dict | None:
    try:
        if STATE_FILE.exists():
            with STATE_FILE.open() as f:
                return json.load(f)
    except Exception as e:
        console.print(f"[dim]State file unreadable: {e}[/dim]")
    return None


def _save_state(state: dict) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with STATE_FILE.open("w") as f:
            json.dump(state, f, indent=2, default=str)
    except (OSError, PermissionError):
        pass  # read-only filesystem — state won't persist between runs
    except Exception as e:
        console.print(f"[red]Failed to save state: {e}[/red]")


def _make_alert(level: str, source: str, message: str, details: str = "") -> dict:
    return {
        "level":     level,
        "source":    source,
        "message":   message,
        "details":   details,
        "timestamp": _now_local_str(),
    }


# ── State snapshot helpers ─────────────────────────────────────────────────────

def _snapshot_verdicts(verdicts: list[dict]) -> dict[str, str]:
    return {v["asset"]: v["verdict"] for v in verdicts}


def _snapshot_mason(fund_sigs: list[dict]) -> dict[str, str]:
    return {
        f"{s['asset']}|{s['signal']}": s["status"]
        for s in fund_sigs
    }


def _snapshot_edgar(edgar: dict) -> list[str]:
    """Return sorted list of all known accession numbers from seen filings."""
    try:
        seen_path = DATA_DIR / "edgar_seen_filings.json"
        if seen_path.exists():
            with seen_path.open() as f:
                data = json.load(f)
            return sorted(data.get("seen_accessions", []))
    except Exception:
        pass
    return []


def _snapshot_forge(forge: dict) -> float | None:
    try:
        return float(forge["current_price"]) if forge.get("current_price") else None
    except Exception:
        return None


def _build_state(
    verdicts: list[dict],
    fund_sigs: list[dict],
    edgar: dict,
    forge: dict,
) -> dict:
    return {
        "timestamp":        _now_local_str(),
        "verdicts":         _snapshot_verdicts(verdicts),
        "mason_signals":    _snapshot_mason(fund_sigs),
        "edgar_accessions": _snapshot_edgar(edgar),
        "forge_price":      _snapshot_forge(forge),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ALERT GENERATORS — one per source
# ══════════════════════════════════════════════════════════════════════════════

def _alerts_edgar(edgar: dict, prev_accessions: list[str]) -> list[dict]:
    alerts = []
    current_accessions = set(_snapshot_edgar(edgar))
    prev_set = set(prev_accessions)
    new_accessions = current_accessions - prev_set

    # Cross-reference new accessions with tracked company filings
    all_found = edgar.get("all_found", [])
    adsh_to_filing = {f["accession_no"]: f for f in all_found}

    for adsh in sorted(new_accessions):
        filing = adsh_to_filing.get(adsh)
        if filing:
            company  = filing.get("tracked_as", filing.get("company_name", "Unknown"))
            msg      = f"S-1 filed by {company} — Ji Moment triggered"
            detail   = (
                f"Accession: {adsh} | "
                f"Filed: {filing.get('file_date', '?')} | "
                f"URL: {filing.get('filing_url', '?')}"
            )
            alerts.append(_make_alert("MAX_ALERT", "EDGAR", msg, detail))
        else:
            # Accession in seen file but not in all_found — might be from a prior run
            alerts.append(_make_alert(
                "MAX_ALERT", "EDGAR",
                f"New S-1 accession detected: {adsh}",
                "Cross-reference edgar_seen_filings.json for details",
            ))
    return alerts


def _alerts_convergence(
    current_verdicts: dict[str, str],
    prev_verdicts: dict[str, str],
) -> list[dict]:
    alerts = []

    for asset, verdict in current_verdicts.items():
        prev = prev_verdicts.get(asset)

        # CONVERGENCE: fire HIGH alert when it appears
        if verdict == "CONVERGENCE" and prev != "CONVERGENCE":
            alerts.append(_make_alert(
                "HIGH", "INTEGRATION",
                f"CONVERGENCE detected — {asset}",
                f"Fundamental GREEN + Technical BULLISH + RSI <50 all aligned. "
                f"Previous verdict: {prev or 'none'}",
            ))

        # CONFIRMS: fire MEDIUM when it appears
        elif verdict == "CONFIRMS" and prev not in ("CONFIRMS", "CONVERGENCE"):
            alerts.append(_make_alert(
                "MEDIUM", "INTEGRATION",
                f"CONFIRMS verdict — {asset}",
                f"Fundamental entry signal confirmed by technical posture. "
                f"Previous: {prev or 'none'}",
            ))

        # Any verdict degraded from CONVERGENCE/CONFIRMS → log LOW
        elif prev in ("CONVERGENCE", "CONFIRMS") and verdict not in ("CONVERGENCE", "CONFIRMS"):
            alerts.append(_make_alert(
                "LOW", "INTEGRATION",
                f"Verdict cleared for {asset}: {prev} → {verdict}",
                "Entry signal no longer active",
            ))

        # Any other change that isn't just NO SIGNAL ↔ NO SIGNAL
        elif verdict != prev and prev is not None:
            if not (verdict == "NO SIGNAL" and prev == "NO SIGNAL"):
                alerts.append(_make_alert(
                    "LOW", "INTEGRATION",
                    f"Verdict changed — {asset}: {prev} → {verdict}",
                ))

    return alerts


def _alerts_mason(
    current_mason: dict[str, str],
    prev_mason: dict[str, str],
) -> list[dict]:
    alerts = []

    for key, status in current_mason.items():
        prev = prev_mason.get(key)
        if prev is None or status == prev:
            continue

        asset, signal = key.split("|", 1)

        # Determine alert level by transition
        if status == "GREEN" and prev in ("NO SIGNAL", "AMBER"):
            level  = "MEDIUM"
            msg    = f"Entry signal ACTIVATED — {asset}: {signal}"
            detail = f"Status: {prev} → {status}"
        elif status == "RED" and prev in ("NO SIGNAL", "AMBER"):
            level  = "MEDIUM"
            msg    = f"Warning ACTIVATED — {asset}: {signal}"
            detail = f"Status: {prev} → {status}"
        elif prev in ("GREEN", "RED") and status == "NO SIGNAL":
            level  = "LOW"
            msg    = f"Signal cleared — {asset}: {signal}"
            detail = f"Status: {prev} → {status}"
        else:
            level  = "LOW"
            msg    = f"Signal changed — {asset}: {signal}"
            detail = f"Status: {prev} → {status}"

        alerts.append(_make_alert(level, "MASON", msg, detail))

    return alerts


def _alerts_forge(
    current_price: float | None,
    prev_price: float | None,
    threshold: float = 0.10,
) -> list[dict]:
    if current_price is None or prev_price is None or prev_price == 0:
        return []

    pct_change = (current_price - prev_price) / prev_price
    if abs(pct_change) >= threshold:
        direction = "UP" if pct_change > 0 else "DOWN"
        return [_make_alert(
            "MEDIUM", "FORGE",
            f"SpaceX Forge price moved {pct_change * 100:+.1f}% ({direction})",
            f"Previous: ${prev_price:,.2f} → Current: ${current_price:,.2f}",
        )]
    return []


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def check_all_alerts() -> list[dict]:
    """
    Collect current signal state from all modules, compare against
    data/last_alert_state.json, and return a list of alert dicts.

    First run: initialises state file and returns empty list.
    Subsequent runs: returns only changes since last run.
    """
    alerts: list[dict] = []
    errors: list[str]  = []

    # ── Load signal modules ────────────────────────────────────────────────────
    verdicts  = []
    fund_sigs = []
    edgar     = {}
    forge     = {}

    try:
        from data_feed import get_all_data
        from tatum_indicators import analyze_all
        data     = get_all_data()
        tech     = analyze_all(data)
    except Exception as e:
        errors.append(f"data_feed/tatum_indicators: {e}")
        data = tech = None

    try:
        from mason_signals import get_fundamental_signals
        fund_sigs = get_fundamental_signals(data) if data else []
    except Exception as e:
        errors.append(f"mason_signals: {e}")

    try:
        from integration import get_confirmation_verdicts
        verdicts = get_confirmation_verdicts(tech, fund_sigs, data or {}) if tech else []
    except Exception as e:
        errors.append(f"integration: {e}")

    try:
        from venture_signals import get_edgar_status, get_spacex_forge_price
        edgar = get_edgar_status()
        forge = get_spacex_forge_price()
    except Exception as e:
        errors.append(f"venture_signals: {e}")

    # ── Build current state snapshot ───────────────────────────────────────────
    current_state = _build_state(verdicts, fund_sigs, edgar, forge)

    # ── Load previous state ────────────────────────────────────────────────────
    prev_state = _load_state()

    if prev_state is None:
        # ── First run: initialise, no alerts ──────────────────────────────────
        console.print(
            "[dim]Alert system: first run detected — "
            "initialising state snapshot. No alerts on first run.[/dim]"
        )
        _save_state(current_state)
        return []

    # ── Compare each source ────────────────────────────────────────────────────
    try:
        alerts += _alerts_edgar(
            edgar,
            prev_state.get("edgar_accessions", []),
        )
    except Exception as e:
        errors.append(f"edgar diff: {e}")

    try:
        alerts += _alerts_convergence(
            current_state["verdicts"],
            prev_state.get("verdicts", {}),
        )
    except Exception as e:
        errors.append(f"convergence diff: {e}")

    try:
        alerts += _alerts_mason(
            current_state["mason_signals"],
            prev_state.get("mason_signals", {}),
        )
    except Exception as e:
        errors.append(f"mason diff: {e}")

    try:
        alerts += _alerts_forge(
            current_state["forge_price"],
            prev_state.get("forge_price"),
        )
    except Exception as e:
        errors.append(f"forge diff: {e}")

    # ── Log module errors as LOW alerts ───────────────────────────────────────
    for err in errors:
        alerts.append(_make_alert("LOW", "SYSTEM", f"Module error: {err}"))

    # ── Sort by level (highest first) ─────────────────────────────────────────
    alerts.sort(key=lambda a: -LEVEL_RANK.get(a["level"], 0))

    # ── Persist new state ──────────────────────────────────────────────────────
    _save_state(current_state)

    return alerts


def log_alerts(alerts: list[dict]) -> None:
    """
    Write each alert to logs/alerts.log and print to terminal.
    MAX_ALERT gets a prominent panel at the top.
    """
    if not alerts:
        console.print("[dim]Alert check complete — no changes detected.[/dim]")
        return

    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError):
        pass

    # ── Write to file (via named logger, not root) ────────────────────────────
    for a in alerts:
        line = (
            f"[{a['timestamp']}] [{a['level']:10s}] [{a['source']:12s}] "
            f"{a['message']}"
        )
        if a.get("details"):
            line += f" | {a['details']}"
        _file_log.info(line)

    # ── MAX_ALERT panels (one per filing) ──────────────────────────────────────
    max_alerts = [a for a in alerts if a["level"] == "MAX_ALERT"]
    for a in max_alerts:
        console.print(Panel(
            f"[bold white]{a['message']}[/bold white]\n\n"
            f"[dim]{a.get('details', '')}[/dim]\n\n"
            f"[bold]Timestamp: {a['timestamp']}[/bold]",
            title="[bold white on red]  ⚡  MAX ALERT — ACT NOW  ⚡  [/bold white on red]",
            border_style="bold red",
            padding=(1, 3),
        ))

    # ── Alert summary table ────────────────────────────────────────────────────
    table = Table(
        show_header=True, header_style="bold", show_lines=True,
        border_style="dim", box=None, pad_edge=False,
    )
    table.add_column("Time",      style="dim",  no_wrap=True, min_width=19)
    table.add_column("Level",     justify="center",           min_width=12)
    table.add_column("Source",    justify="center",           min_width=12)
    table.add_column("Message",                               min_width=45)
    table.add_column("Details",   style="dim",                min_width=30)

    for a in alerts:
        lvl   = a["level"]
        style = LEVEL_STYLE.get(lvl, "white")
        detail = a.get("details", "")
        if len(detail) > 60:
            detail = detail[:57] + "…"
        table.add_row(
            a["timestamp"],
            f"[{style}]{lvl}[/{style}]",
            a["source"],
            a["message"],
            detail,
        )

    counts = {}
    for a in alerts:
        counts[a["level"]] = counts.get(a["level"], 0) + 1

    summary_parts = " | ".join(
        f"[{LEVEL_STYLE.get(lvl, 'white')}]{n} {lvl}[/{LEVEL_STYLE.get(lvl, 'white')}]"
        for lvl, n in sorted(counts.items(), key=lambda x: -LEVEL_RANK.get(x[0], 0))
    )

    console.print()
    console.print(f"[bold]Alert summary:[/bold]  {summary_parts}")
    console.print(table)
    console.print(f"[dim]Logged to: {ALERT_LOG_FILE}[/dim]")


def run_alert_check() -> list[dict]:
    """
    Main entry point. Run check_all_alerts() then log_alerts().
    Returns the alert list (for use by dashboard.py).
    """
    console.print("[dim]Running alert check…[/dim]")
    alerts = check_all_alerts()
    log_alerts(alerts)
    return alerts


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_alert_check()
