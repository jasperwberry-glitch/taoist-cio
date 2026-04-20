"""
data_feed.py — Market data layer for the Taoist CIO dashboard.

Pulls 1-year daily OHLCV history for the full market universe via yfinance.
Results are cached in data/market_cache.json with a 15-minute TTL.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
LOGS_DIR = ROOT / "logs"
CACHE_FILE = DATA_DIR / "market_cache.json"
CACHE_TTL_MINUTES = 15

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    filename=LOGS_DIR / "data_feed.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

console = Console()

# ── Ticker Universe ────────────────────────────────────────────────────────────
TICKERS: dict[str, list[str]] = {
    "US_EQUITY":        ["SPY", "QQQ", "DIA"],
    "CRYPTO":           ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"],
    "GOLD_PHYSICAL":    ["GLD", "IAU"],
    "GOLD_ROYALTY":     ["FNV", "WPM"],
    "GOLD_MINER":       ["AEM", "NEM"],
    "SILVER_PHYSICAL":  ["PSLV", "SLV"],
    "COPPER_PHYSICAL":  ["CPER"],
    "COPPER_MINER":     ["FCX"],
    "URANIUM_PHYSICAL": ["SRUUF"],
    "URANIUM_ETF":      ["URA"],
    "URANIUM_MINER":    ["CCJ"],
    "WATER_UTILITY":    ["AWK"],
    "WATER_ETF":        ["CGW"],
    "INTERNATIONAL":    ["EFA"],
    "CREDIT":           ["HYG", "LQD"],
    "MACRO":            ["^TNX", "DX-Y.NYB", "^VIX", "BZ=F", "GC=F", "SI=F", "HG=F"],
    "SENTIMENT":        ["^PCALL"],
    "IPO_PROXY":        ["ARKVX", "XOVR", "AMZN", "GOOGL"],
}

ALL_TICKERS: list[str] = [t for tickers in TICKERS.values() for t in tickers]

# ── Cache helpers ──────────────────────────────────────────────────────────────

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _load_cache() -> dict | None:
    """Return parsed cache dict, or None if missing / unreadable."""
    try:
        if not CACHE_FILE.exists():
            return None
        with CACHE_FILE.open() as f:
            return json.load(f)
    except Exception as e:
        log.error(f"Cache read failed: {e}")
        return None


def _cache_is_fresh(cache: dict) -> bool:
    """True if cache timestamp is within TTL."""
    try:
        ts = datetime.fromisoformat(cache["timestamp"])
        age_minutes = (_now_utc() - ts).total_seconds() / 60
        return age_minutes < CACHE_TTL_MINUTES
    except Exception:
        return False


def _serialize_ohlcv(data: dict[str, pd.DataFrame]) -> dict:
    """Convert {ticker: DataFrame} → JSON-safe nested dict."""
    out = {}
    for ticker, df in data.items():
        df_copy = df.copy()
        df_copy.index = df_copy.index.strftime("%Y-%m-%d")
        out[ticker] = df_copy.to_dict(orient="index")
    return out


def _deserialize_ohlcv(raw: dict) -> dict[str, pd.DataFrame]:
    """Restore {ticker: DataFrame} from cached JSON."""
    result = {}
    for ticker, records in raw.items():
        df = pd.DataFrame.from_dict(records, orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "Date"
        result[ticker] = df
    return result


def _save_cache(ohlcv: dict[str, pd.DataFrame]) -> None:
    """Write OHLCV data + timestamp to CACHE_FILE."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp": _now_utc().isoformat(),
            "ohlcv": _serialize_ohlcv(ohlcv),
        }
        with CACHE_FILE.open("w") as f:
            json.dump(payload, f)
    except Exception as e:
        log.error(f"Cache write failed: {e}")


# ── Download logic ─────────────────────────────────────────────────────────────

def _extract_ticker_df(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Pull a single-ticker DataFrame from a multi-ticker yfinance download.
    raw has MultiIndex columns: (PriceType, Ticker).
    """
    mask = raw.columns.get_level_values(1) == ticker
    df = raw.loc[:, mask].copy()
    df.columns = df.columns.get_level_values(0)          # flatten to Open/High/…
    df = df.dropna(how="all")
    return df


def _fetch_individual(ticker: str) -> pd.DataFrame | None:
    """Fallback: download a single ticker via Ticker.history()."""
    try:
        df = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="all")
        if df.empty:
            raise ValueError("empty result")
        return df
    except Exception as e:
        log.error(f"Individual fetch failed for {ticker}: {e}")
        return None


def _fetch_all_from_yfinance() -> dict[str, pd.DataFrame]:
    """
    Bulk-download all tickers; fall back to individual fetch for any that fail.
    """
    console.print("[dim]Downloading market data from yfinance…[/dim]")
    result: dict[str, pd.DataFrame] = {}

    try:
        raw = yf.download(
            ALL_TICKERS,
            period="1y",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            multi_level_index=True,
        )
    except Exception as e:
        log.error(f"Bulk download failed: {e}")
        raw = pd.DataFrame()

    for ticker in ALL_TICKERS:
        try:
            if raw.empty:
                raise ValueError("bulk download returned nothing")
            df = _extract_ticker_df(raw, ticker)
            if df.empty or len(df) < 2:
                raise ValueError("insufficient rows after extraction")
            result[ticker] = df
        except Exception as bulk_err:
            # Fallback to individual download
            df = _fetch_individual(ticker)
            if df is not None:
                result[ticker] = df
            else:
                log.error(f"All fetch attempts failed for {ticker}: {bulk_err}")

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def get_all_data(force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    """
    Return {ticker: DataFrame} with 1-year daily OHLCV history.

    Serves from cache if it is < CACHE_TTL_MINUTES old, unless force_refresh=True.
    """
    if not force_refresh:
        cache = _load_cache()
        if cache and _cache_is_fresh(cache):
            age = (_now_utc() - datetime.fromisoformat(cache["timestamp"])).total_seconds() / 60
            console.print(f"[dim]Using cached data ({age:.1f} min old)[/dim]")
            return _deserialize_ohlcv(cache["ohlcv"])

    data = _fetch_all_from_yfinance()
    _save_cache(data)
    return data


def get_current_prices(
    data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, dict]:
    """
    Return a summary dict: {ticker: {price, change_pct, last_updated, status}}.

    Pass in a pre-fetched `data` dict to avoid a second yfinance call.
    """
    if data is None:
        data = get_all_data()

    result: dict[str, dict] = {}

    for ticker in ALL_TICKERS:
        if ticker not in data or data[ticker].empty:
            log.error(f"No data available for {ticker}")
            result[ticker] = {
                "price": None,
                "change_pct": None,
                "last_updated": None,
                "status": "ERROR",
            }
            continue

        try:
            df = data[ticker]
            if len(df) < 2:
                raise ValueError("need at least 2 rows to compute change")
            last_close = float(df["Close"].iloc[-1])
            prev_close = float(df["Close"].iloc[-2])
            change_pct = (last_close - prev_close) / prev_close * 100
            last_date = df.index[-1]
            if hasattr(last_date, "strftime"):
                last_date = last_date.strftime("%Y-%m-%d")

            result[ticker] = {
                "price": round(last_close, 4),
                "change_pct": round(change_pct, 2),
                "last_updated": str(last_date),
                "status": "OK",
            }
        except Exception as e:
            log.error(f"Price computation failed for {ticker}: {e}")
            result[ticker] = {
                "price": None,
                "change_pct": None,
                "last_updated": None,
                "status": "ERROR",
            }

    return result


# ── Rich display ───────────────────────────────────────────────────────────────

def print_market_table(prices: dict[str, dict] | None = None) -> None:
    """Print a Rich table: Ticker | Category | Price | Change % | Status."""
    if prices is None:
        prices = get_current_prices()

    # Build reverse lookup: ticker → category
    ticker_category = {
        t: cat for cat, tickers in TICKERS.items() for t in tickers
    }

    table = Table(
        title="Taoist CIO — Market Universe",
        show_lines=False,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Ticker",   style="cyan bold", no_wrap=True)
    table.add_column("Category", style="dim")
    table.add_column("Price",    justify="right")
    table.add_column("Change %", justify="right")
    table.add_column("Updated",  justify="center", style="dim")
    table.add_column("Status",   justify="center")

    ok_count = err_count = 0
    for ticker in ALL_TICKERS:
        info = prices.get(ticker, {"status": "ERROR"})
        category = ticker_category.get(ticker, "—")

        if info["status"] == "OK":
            ok_count += 1
            chg = info["change_pct"]
            chg_color = "green" if chg >= 0 else "red"
            chg_str = f"[{chg_color}]{chg:+.2f}%[/{chg_color}]"

            # Format price: crypto and metals have larger values
            price_val = info["price"]
            price_str = (
                f"{price_val:,.2f}" if price_val >= 1
                else f"{price_val:.6f}"
            )

            table.add_row(
                ticker,
                category,
                price_str,
                chg_str,
                info.get("last_updated", "—"),
                "[green]OK[/green]",
            )
        else:
            err_count += 1
            table.add_row(
                ticker,
                category,
                "—",
                "—",
                "—",
                "[red]ERROR[/red]",
            )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] [green]{ok_count} OK[/green]  "
        f"[red]{err_count} ERROR[/red]  "
        f"| [dim]Cache TTL: {CACHE_TTL_MINUTES} min[/dim]"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--refresh" in sys.argv
    data = get_all_data(force_refresh=force)
    prices = get_current_prices(data=data)
    print_market_table(prices)
