"""
venture_signals.py — IPO and Pre-IPO signals module (Layer 3) for the Taoist CIO dashboard.

Three sub-systems:
  1. SEC EDGAR S-1 monitor  — real-time filing detection for five tracked companies
  2. SpaceX Forge price     — secondary market price with time-series cache
  3. Anthropic status       — manually maintained fundamentals

IPO signal logic:
  Ji Moment     — any tracked company files an S-1 (MAX_ALERT)
  Wu Wei Entry  — any listed company trades >20% below its IPO opening price
"""

import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths / logging ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"

try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=LOGS_DIR / "venture.log",
        level=logging.ERROR,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
except (OSError, PermissionError):
    logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)
console = Console()

# ── File paths ─────────────────────────────────────────────────────────────────
EDGAR_CACHE_FILE   = DATA_DIR / "edgar_last_check.json"
EDGAR_SEEN_FILE    = DATA_DIR / "edgar_seen_filings.json"
FORGE_HISTORY_FILE = DATA_DIR / "forge_price_history.json"
ANTHROPIC_FILE     = DATA_DIR / "anthropic_status.json"

# ── HTTP headers ───────────────────────────────────────────────────────────────
EDGAR_HEADERS = {
    "User-Agent": "Taoist CIO Dashboard contact@example.com",
    "Accept":     "application/json",
}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# ── Tracked companies ──────────────────────────────────────────────────────────
# Each entry: (display_name, [legal_name_variants_for_EDGAR_search])
TRACKED_COMPANIES: list[tuple[str, list[str]]] = [
    ("SpaceX",     ["Space Exploration Technologies Corp",
                    "Space Exploration Technologies"]),
    ("Anthropic",  ["Anthropic PBC", "Anthropic"]),
    ("OpenAI",     ["OpenAI Inc", "OpenAI Holdings", "OpenAI LLC", "OpenAI"]),
    ("Databricks", ["Databricks Inc", "Databricks"]),
    ("Stripe",     ["Stripe Inc", "Stripe"]),
]

EDGAR_CACHE_TTL_HOURS  = 1
FORGE_ALERT_THRESHOLD  = 0.10   # 10% move triggers alert
ANTHROPIC_STALE_DAYS   = 30


# ── Utility helpers ────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: Path) -> dict | list | None:
    try:
        if path.exists():
            with path.open() as f:
                return json.load(f)
    except Exception as e:
        log.error(f"JSON read failed {path}: {e}")
    return None


def _save_json(path: Path, data) -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(data, f, indent=2, default=str)
    except (OSError, PermissionError):
        pass  # read-only filesystem — skip persistence
    except Exception as e:
        log.error(f"JSON write failed {path}: {e}")


def _filing_url(adsh: str, cik: str) -> str:
    """Build the SEC EDGAR filing index URL from accession number and CIK."""
    clean_adsh = adsh.replace("-", "")
    cik_num    = cik.lstrip("0")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{clean_adsh}/"


# ══════════════════════════════════════════════════════════════════════════════
# PART 1 — SEC EDGAR S-1 MONITOR
# ══════════════════════════════════════════════════════════════════════════════

def _query_edgar_for_company(
    legal_name: str,
    start_dt: date,
    end_dt: date,
) -> list[dict]:
    """
    Query EDGAR full-text search for S-1 filings that both:
      (a) mention the legal name in the document, AND
      (b) list the company as the filer (display_names filter).

    Returns a deduplicated list of filing dicts (one per unique accession number).
    """
    try:
        r = requests.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q":         f'"{legal_name}"',
                "forms":     "S-1",
                "dateRange": "custom",
                "startdt":   start_dt.isoformat(),
                "enddt":     end_dt.isoformat(),
            },
            headers=EDGAR_HEADERS,
            timeout=20,
        )
        r.raise_for_status()
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception as e:
        log.error(f"EDGAR query failed for '{legal_name}': {e}")
        return []

    # Filter: the company itself must be the filer
    name_lower = legal_name.lower()
    own = [
        h for h in hits
        if any(
            name_lower in dn.lower()
            for dn in h["_source"].get("display_names", [])
        )
    ]

    # Deduplicate by accession number (one filing = many index entries)
    seen_adsh: set[str] = set()
    results: list[dict] = []
    for h in own:
        src  = h["_source"]
        adsh = src.get("adsh", "")
        if adsh in seen_adsh:
            continue
        seen_adsh.add(adsh)
        cik = (src.get("ciks") or ["0"])[0]
        results.append({
            "accession_no":   adsh,
            "file_date":      src.get("file_date", ""),
            "form_type":      src.get("form", "S-1"),
            "company_name":   src.get("display_names", [legal_name])[0],
            "cik":            cik,
            "filing_url":     _filing_url(adsh, cik),
            "legal_name_hit": legal_name,
        })

    return results


def check_edgar_for_s1_filings(lookback_days: int = 7) -> list[dict]:
    """
    Check EDGAR for new S-1 filings from all tracked companies.

    - Serves from cache if last check was < EDGAR_CACHE_TTL_HOURS ago.
    - Compares against edgar_seen_filings.json to detect truly NEW filings.
    - Tags new filings with priority="MAX_ALERT" (Ji Moment).

    Returns list of new-filing dicts.
    """
    # ── Cache check ────────────────────────────────────────────────────────────
    cache = _load_json(EDGAR_CACHE_FILE)
    if cache:
        try:
            last_ts  = datetime.fromisoformat(cache["timestamp"])
            age_hrs  = (_now_utc() - last_ts).total_seconds() / 3600
            if age_hrs < EDGAR_CACHE_TTL_HOURS:
                console.print(
                    f"[dim]EDGAR: using cached check "
                    f"({age_hrs:.1f}h old, TTL {EDGAR_CACHE_TTL_HOURS}h)[/dim]"
                )
                return cache.get("new_filings", [])
        except Exception:
            pass

    # ── Live query ─────────────────────────────────────────────────────────────
    today     = date.today()
    start_dt  = today - timedelta(days=lookback_days)
    seen_data = _load_json(EDGAR_SEEN_FILE) or {}
    seen_set: set[str] = set(seen_data.get("seen_accessions", []))

    all_found:  list[dict] = []
    new_filings: list[dict] = []

    console.print("[dim]EDGAR: querying SEC for S-1 filings…[/dim]")

    for display_name, legal_variants in TRACKED_COMPANIES:
        company_filings: list[dict] = []
        for variant in legal_variants:
            filings = _query_edgar_for_company(variant, start_dt, today)
            # Avoid duplicate accessions across variants
            existing_adsh = {f["accession_no"] for f in company_filings}
            for f in filings:
                if f["accession_no"] not in existing_adsh:
                    f["tracked_as"] = display_name
                    company_filings.append(f)
                    existing_adsh.add(f["accession_no"])

        for filing in company_filings:
            all_found.append(filing)
            if filing["accession_no"] not in seen_set:
                filing["priority"] = "MAX_ALERT"
                filing["ji_moment"] = True
                new_filings.append(filing)
                seen_set.add(filing["accession_no"])

    # ── Persist ────────────────────────────────────────────────────────────────
    _save_json(EDGAR_CACHE_FILE, {
        "timestamp":    _now_utc().isoformat(),
        "lookback_days": lookback_days,
        "all_found":    all_found,
        "new_filings":  new_filings,
    })
    _save_json(EDGAR_SEEN_FILE, {
        "seen_accessions": sorted(seen_set),
        "last_updated":    _now_utc().isoformat(),
    })

    return new_filings


def get_edgar_status() -> dict:
    """
    Return a summary dict for the dashboard:
    {last_check, tracked_companies, new_filings, alert_active}
    """
    new_filings = check_edgar_for_s1_filings()
    cache       = _load_json(EDGAR_CACHE_FILE) or {}
    return {
        "last_check":        cache.get("timestamp", "never"),
        "lookback_days":     cache.get("lookback_days", 7),
        "tracked_companies": [name for name, _ in TRACKED_COMPANIES],
        "all_found":         cache.get("all_found", []),
        "new_filings":       new_filings,
        "alert_active":      len(new_filings) > 0,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 — SPACEX FORGE PRICE
# ══════════════════════════════════════════════════════════════════════════════

def get_spacex_forge_price() -> dict:
    """
    Attempt to fetch the current SpaceX secondary market price from
    forgeglobal.com/spacex_stock/.

    Forge blocks automated access — this scrape will typically fail.
    On failure, surfaces the last cached reading with stale=True.

    Returns:
        current_price, last_updated, change_from_last, pct_change,
        alert (>10% move), stale (bool), source
    """
    history: list[dict] = _load_json(FORGE_HISTORY_FILE) or []

    last_entry    = history[-1] if history else None
    last_price    = float(last_entry["price"]) if last_entry else None
    last_updated  = last_entry.get("timestamp") if last_entry else None

    scraped_price: float | None = None
    scrape_error:  str | None   = None

    # ── Attempt scrape ─────────────────────────────────────────────────────────
    try:
        r = requests.get(
            "https://forgeglobal.com/spacex_stock/",
            headers=BROWSER_HEADERS,
            timeout=15,
        )
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            # Try common price selectors
            candidates = []
            for sel in [
                {"class": lambda c: c and "price" in " ".join(c).lower()},
                {"class": lambda c: c and "forge-price" in " ".join(c).lower()},
            ]:
                candidates += soup.find_all(True, attrs=sel)

            for tag in candidates:
                text = tag.get_text(strip=True).replace(",", "").replace("$", "")
                try:
                    val = float(text)
                    if 10 < val < 100_000:   # sanity range for a stock price
                        scraped_price = val
                        break
                except ValueError:
                    continue

            if scraped_price is None:
                scrape_error = f"HTTP 200 but no price found in HTML (site may be JS-rendered)"
        else:
            scrape_error = f"HTTP {r.status_code} — site blocks automated access"
    except Exception as e:
        scrape_error = str(e)

    # ── Update history if we got a fresh price ─────────────────────────────────
    if scraped_price is not None:
        entry = {
            "timestamp": _now_utc().isoformat(),
            "price":     scraped_price,
            "source":    "forgeglobal.com",
        }
        history.append(entry)
        _save_json(FORGE_HISTORY_FILE, history)
        current_price  = scraped_price
        current_ts     = entry["timestamp"]
        stale          = False
    else:
        log.error(f"Forge scrape failed: {scrape_error}")
        current_price  = last_price
        current_ts     = last_updated
        stale          = True

    # ── Delta vs previous reading ──────────────────────────────────────────────
    prev_price = float(history[-2]["price"]) if len(history) >= 2 else None
    if current_price and prev_price:
        change_abs = current_price - prev_price
        change_pct = change_abs / prev_price * 100
    else:
        change_abs = change_pct = None

    alert = (
        change_pct is not None
        and abs(change_pct) >= FORGE_ALERT_THRESHOLD * 100
    )

    return {
        "current_price":    current_price,
        "last_updated":     current_ts,
        "change_from_last": change_abs,
        "pct_change":       change_pct,
        "alert":            alert,
        "stale":            stale,
        "stale_reason":     scrape_error if stale else None,
        "history_length":   len(history),
        "source":           "cached_manual" if stale else "forgeglobal.com",
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 — ANTHROPIC STATUS (manual data)
# ══════════════════════════════════════════════════════════════════════════════

def get_anthropic_status() -> dict:
    """
    Load Anthropic fundamentals from data/anthropic_status.json.
    Flags needs_update=True if last_verified is more than ANTHROPIC_STALE_DAYS old.
    Edit the JSON file directly to update values without code changes.
    """
    data = _load_json(ANTHROPIC_FILE)
    if not data:
        return {
            "error":        "anthropic_status.json not found",
            "needs_update": True,
        }

    try:
        verified_dt  = datetime.strptime(data["last_verified"], "%Y-%m-%d").date()
        age_days     = (date.today() - verified_dt).days
        needs_update = age_days > ANTHROPIC_STALE_DAYS
    except Exception:
        age_days     = None
        needs_update = True

    return {
        **data,
        "age_days":    age_days,
        "needs_update": needs_update,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 — IPO SIGNAL LOGIC
# ══════════════════════════════════════════════════════════════════════════════

def check_ji_moment(edgar_status: dict) -> bool:
    """
    Ji Moment: any tracked company has filed an S-1.
    This is the IPO mandate's equivalent of CONVERGENCE — the irreversible threshold.
    """
    return edgar_status.get("alert_active", False)


def check_wu_wei_entry(
    public_ticker: str,
    ipo_open_price: float,
    data_dict: dict | None = None,
) -> bool:
    """
    Wu Wei Entry: price is >20% below the IPO opening price.
    The Facebook-at-$18 pattern — maximum pessimism after a failed IPO.

    Args:
        public_ticker:  yfinance-compatible ticker (e.g. 'SPCE')
        ipo_open_price: the IPO day opening price
        data_dict:      optional pre-fetched data_feed dict; if None, fetches live

    Returns True when Wu Wei conditions are met.
    Currently returns False for all companies since none have IPO'd yet.
    Built and ready for activation when SpaceX/Anthropic list.
    """
    try:
        if data_dict and public_ticker in data_dict:
            df = data_dict[public_ticker]
            current = float(df["Close"].iloc[-1])
        else:
            import yfinance as yf
            ticker = yf.Ticker(public_ticker)
            hist   = ticker.history(period="1d")
            if hist.empty:
                return False
            current = float(hist["Close"].iloc[-1])

        drawdown_pct = (ipo_open_price - current) / ipo_open_price * 100
        return drawdown_pct > 20.0

    except Exception as e:
        log.error(f"Wu Wei check failed for {public_ticker}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# PART 5 — UNIFIED OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def get_venture_signals(data_dict: dict | None = None) -> dict:
    """
    Collect all venture intelligence into a single dict.

    Args:
        data_dict: optional pre-fetched {ticker: DataFrame} from data_feed.
                   Used for Wu Wei checks on listed companies.

    Returns:
        {edgar, spacex_forge, anthropic, ji_moment, wu_wei_entry, alerts}
    """
    edgar     = get_edgar_status()
    forge     = get_spacex_forge_price()
    anthropic = get_anthropic_status()

    ji = check_ji_moment(edgar)

    # Wu Wei: only relevant post-IPO — all return False until listing day
    # Add entries here as companies go public:
    # wu_wei_entries = [t for t, price in [("SPCE", 0.00)] if check_wu_wei_entry(t, price, data_dict)]
    wu_wei_entries: list[str] = []

    # Collate all alerts
    alerts: list[dict] = []
    for filing in edgar.get("new_filings", []):
        alerts.append({
            "priority":    "MAX_ALERT",
            "type":        "JI_MOMENT",
            "message":     f"S-1 FILED: {filing.get('tracked_as')} ({filing.get('company_name')})",
            "detail":      f"Filed {filing.get('file_date')} | {filing.get('filing_url')}",
            "accession_no": filing.get("accession_no"),
        })
    for company in wu_wei_entries:
        alerts.append({
            "priority": "HIGH",
            "type":     "WU_WEI_ENTRY",
            "message":  f"WU WEI ENTRY: {company} >20% below IPO open",
        })
    if forge.get("alert"):
        alerts.append({
            "priority": "MEDIUM",
            "type":     "FORGE_MOVE",
            "message":  f"SpaceX Forge price moved {forge.get('pct_change',0):+.1f}%",
        })

    return {
        "edgar":         edgar,
        "spacex_forge":  forge,
        "anthropic":     anthropic,
        "ji_moment":     ji,
        "wu_wei_entry":  wu_wei_entries,
        "alerts":        alerts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# RICH DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

def print_venture_summary(signals: dict) -> None:
    """Render full venture intelligence panel in the terminal."""

    # ── Ji Moment alert (top priority) ────────────────────────────────────────
    if signals["ji_moment"]:
        for filing in signals["edgar"]["new_filings"]:
            console.print(Panel(
                f"[bold white]⚡ JI MOMENT: {filing.get('tracked_as', '').upper()} — "
                f"S-1 FILED WITH SEC[/bold white]\n\n"
                f"  Company    : {filing.get('company_name')}\n"
                f"  Filed      : {filing.get('file_date')}\n"
                f"  Accession  : {filing.get('accession_no')}\n"
                f"  Filing URL : [link]{filing.get('filing_url')}[/link]\n\n"
                f"[bold]This is the irreversible threshold. Review S-1 immediately.[/bold]",
                title="[bold white on red]  ⚡  JI MOMENT — IPO FILING DETECTED — ACT NOW  ⚡  [/bold white on red]",
                border_style="bold red",
                padding=(1, 3),
            ))

    # ── Pipeline status table ──────────────────────────────────────────────────
    all_found_adsh = {
        f["accession_no"]: f
        for f in signals["edgar"].get("all_found", [])
    }
    # Map tracked_as → most recent filing
    company_filings: dict[str, dict] = {}
    for f in signals["edgar"].get("all_found", []):
        name = f.get("tracked_as", "")
        existing = company_filings.get(name)
        if not existing or f.get("file_date", "") > existing.get("file_date", ""):
            company_filings[name] = f

    pipeline = Table(
        title="[bold]Pre-IPO Pipeline — Tracked Companies[/bold]",
        show_lines=True, header_style="bold", border_style="dim",
        box=None, pad_edge=False,
    )
    pipeline.add_column("Company",     style="bold",  no_wrap=True,  min_width=12)
    pipeline.add_column("S-1 Status",  justify="center",             min_width=20)
    pipeline.add_column("Filed",       justify="center",             min_width=12)
    pipeline.add_column("Form",        justify="center",             min_width=8)
    pipeline.add_column("Accession",   style="dim",                  min_width=22)
    pipeline.add_column("Link",        style="dim",                  min_width=10)

    last_check = signals["edgar"].get("last_check", "unknown")
    lookback   = signals["edgar"].get("lookback_days", 7)

    for display_name, _ in TRACKED_COMPANIES:
        filing = company_filings.get(display_name)
        if filing:
            is_new = any(
                f.get("accession_no") == filing.get("accession_no")
                for f in signals["edgar"].get("new_filings", [])
            )
            status_str = "[bold red]⚡ NEW — JI MOMENT[/bold red]" if is_new else "[yellow]FILED (seen)[/yellow]"
            pipeline.add_row(
                display_name,
                status_str,
                filing.get("file_date", "—"),
                filing.get("form_type", "—"),
                filing.get("accession_no", "—"),
                filing.get("filing_url", "—"),
            )
        else:
            pipeline.add_row(
                display_name,
                "[dim]No S-1 detected[/dim]",
                "—", "—", "—", "—",
            )

    console.print()
    console.print(Panel(
        pipeline,
        title="[bold cyan]LAYER 3 — VENTURE / IPO SIGNALS[/bold cyan]",
        border_style="cyan", padding=(0, 1),
    ))
    console.print(
        f"[dim]  EDGAR lookback: {lookback} days | "
        f"Last check: {last_check} | "
        f"Cache TTL: {EDGAR_CACHE_TTL_HOURS}h[/dim]"
    )

    # ── SpaceX Forge price ─────────────────────────────────────────────────────
    forge = signals["spacex_forge"]
    price = forge.get("current_price")
    price_str  = f"${price:,.2f}" if price else "N/A"
    change_str = ""
    if forge.get("pct_change") is not None:
        chg = forge["pct_change"]
        col = "red" if forge["alert"] else ("green" if chg >= 0 else "dim")
        change_str = f" [{col}]({chg:+.1f}% vs prev)[/{col}]"

    stale_tag = (
        f"\n  [bold yellow]⚠ STALE — manual update required[/bold yellow]"
        f"\n  [dim]Reason: {forge.get('stale_reason', 'unknown')}[/dim]"
        if forge.get("stale") else ""
    )

    forge_body = (
        f"  [bold]SpaceX (SPCE — pre-IPO)[/bold]\n"
        f"  Forge secondary price : [bold green]{price_str}[/bold green]{change_str}\n"
        f"  Last updated          : {forge.get('last_updated', 'N/A')}\n"
        f"  Source                : {forge.get('source', 'N/A')}\n"
        f"  History entries       : {forge.get('history_length', 0)}"
        f"{stale_tag}"
    )
    alert_border = "bold yellow" if forge.get("alert") else "dim"
    console.print(Panel(
        forge_body,
        title="[bold]SpaceX Forge Secondary Market Price[/bold]",
        border_style=alert_border, padding=(0, 2),
    ))

    # ── Anthropic status ───────────────────────────────────────────────────────
    ant = signals["anthropic"]
    if "error" not in ant:
        arr_b       = ant.get("last_known_arr", 0) / 1e9
        cc_arr_b    = ant.get("claude_code_arr", 0) / 1e9
        stale_warn  = (
            f"\n  [bold yellow]⚠ DATA IS {ant.get('age_days')} DAYS OLD — update anthropic_status.json[/bold yellow]"
            if ant.get("needs_update") else ""
        )
        ant_body = (
            f"  Hiive secondary price : [bold]${ant.get('last_known_hiive_price', 0):,.2f}[/bold] per share\n"
            f"  Total ARR             : [bold]${arr_b:.1f}B[/bold]\n"
            f"  Claude Code ARR       : [bold]${cc_arr_b:.1f}B[/bold]\n"
            f"  Valuation range       : [bold]{ant.get('valuation_range', 'N/A')}[/bold]\n"
            f"  IPO target window     : {ant.get('ipo_target', 'N/A')}\n"
            f"  Last verified         : {ant.get('last_verified', 'N/A')} "
            f"[dim]({ant.get('age_days', '?')} days ago)[/dim]"
            f"{stale_warn}"
        )
    else:
        ant_body = f"[red]{ant['error']}[/red]"

    ant_border = "bold yellow" if ant.get("needs_update") else "dim"
    console.print(Panel(
        ant_body,
        title="[bold]Anthropic — Pre-IPO Fundamentals (Manual)[/bold]",
        border_style=ant_border, padding=(0, 2),
    ))

    # ── Wu Wei entry (post-IPO monitor) ───────────────────────────────────────
    wu_entries = signals.get("wu_wei_entry", [])
    wu_body = (
        "[dim]No companies currently listed. Wu Wei monitor activates on IPO day.\n"
        "Logic: entry fires when price drops >20% below IPO opening price (the 'Facebook at $18' moment).[/dim]"
        if not wu_entries
        else "\n".join(
            f"[bold green]★ WU WEI ENTRY: {c}[/bold green]" for c in wu_entries
        )
    )
    console.print(Panel(
        wu_body,
        title="[bold]Wu Wei Entry Monitor (Post-IPO)[/bold]",
        border_style="dim", padding=(0, 2),
    ))


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold cyan]Taoist CIO — Layer 3: Venture / IPO Signals[/bold cyan]\n")

    signals = get_venture_signals()
    print_venture_summary(signals)

    # ── EDGAR validation test ──────────────────────────────────────────────────
    console.print()
    console.print("[bold]EDGAR validation test — querying for Figma S-1 (filed 2025-07-01)[/bold]")
    console.print("[dim]This confirms the filing detection logic is working correctly.[/dim]")

    from datetime import date as _date
    test_filings = _query_edgar_for_company(
        "Figma, Inc.",
        _date(2025, 6, 28),
        _date(2025, 7, 5),
    )
    if test_filings:
        f = test_filings[0]
        console.print(f"[green]✓ Figma S-1 detected:[/green]")
        console.print(f"  Filed      : {f['file_date']}")
        console.print(f"  Form       : {f['form_type']}")
        console.print(f"  Accession  : {f['accession_no']}")
        console.print(f"  Filer      : {f['company_name']}")
        console.print(f"  Filing URL : {f['filing_url']}")
        console.print(
            f"\n[green]✓ EDGAR query pattern confirmed working.[/green] "
            f"When SpaceX files, the system will detect it within "
            f"{EDGAR_CACHE_TTL_HOURS}h of the query cache expiring."
        )
    else:
        console.print("[red]✗ Figma test failed — check EDGAR connectivity[/red]")
