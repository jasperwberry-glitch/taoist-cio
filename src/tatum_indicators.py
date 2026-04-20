"""
tatum_indicators.py — Technical indicators layer (Layer 1) for the Taoist CIO dashboard.

Takes {ticker: DataFrame} from data_feed.py and produces a full technical
reading for every ticker: moving averages, RSI, MACD, Bollinger Bands, ATR,
VIX classification, and a composite technical posture score.
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import pandas_ta as ta
from rich.console import Console
from rich.table import Table

# ── Paths / logging ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
logging.basicConfig(
    filename=ROOT / "logs" / "indicators.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

# ── Scoring maps ───────────────────────────────────────────────────────────────
# Each classification maps to a momentum score: +1 bullish, 0 neutral, -1 bearish

_MA_ALIGN_SCORE = {
    "FULLY BULLISH": 1,
    "MIXED":         0,
    "FULLY BEARISH": -1,
}

_RSI_SCORE = {
    "OVERBOUGHT": 1,    # strong momentum
    "HIGH":       1,
    "NEUTRAL":    0,
    "LOW":       -1,
    "OVERSOLD":  -1,
}

_MACD_SCORE = {
    "BULLISH CROSSOVER":  1,
    "BULLISH":            1,
    "NEUTRAL":            0,
    "BEARISH":           -1,
    "BEARISH CROSSOVER": -1,
}

_BB_SCORE = {
    "ABOVE UPPER":  1,
    "UPPER THIRD":  1,
    "MIDDLE":       0,
    "LOWER THIRD": -1,
    "BELOW LOWER": -1,
}

_POSTURE_THRESHOLDS = [
    (0.5,  "BULLISH"),
    (0.1,  "NEUTRAL-BULLISH"),
    (-0.1, "NEUTRAL"),
    (-0.5, "NEUTRAL-BEARISH"),
]

_POSTURE_COLORS = {
    "BULLISH":          "bold green",
    "NEUTRAL-BULLISH":  "green",
    "NEUTRAL":          "yellow",
    "NEUTRAL-BEARISH":  "red",
    "BEARISH":          "bold red",
}

# ── Low-level helpers ──────────────────────────────────────────────────────────

def _last(series: pd.Series) -> float | None:
    """Return the last non-NaN value, or None."""
    clean = series.dropna()
    return float(clean.iloc[-1]) if not clean.empty else None


def _prev(series: pd.Series, n: int = 1) -> float | None:
    """Return the nth-from-last non-NaN value (n=1 → second-to-last)."""
    clean = series.dropna()
    return float(clean.iloc[-(n + 1)]) if len(clean) > n else None


def _last_n(series: pd.Series, n: int) -> pd.Series:
    """Return last n non-NaN values."""
    return series.dropna().iloc[-n:]


# ── Classification functions ───────────────────────────────────────────────────

def _classify_ma_position(price: float, ma: float | None) -> str:
    if ma is None:
        return "N/A"
    if price > ma * 1.01:
        return "ABOVE"
    if price < ma * 0.99:
        return "BELOW"
    return "AT"


def _classify_ma_alignment(
    price: float,
    ma20: float | None,
    ma50: float | None,
    ma200: float | None,
) -> str:
    if any(v is None for v in (ma20, ma50, ma200)):
        return "MIXED"
    if price > ma20 > ma50 > ma200:
        return "FULLY BULLISH"
    if price < ma20 < ma50 < ma200:
        return "FULLY BEARISH"
    return "MIXED"


def _classify_rsi(rsi_val: float | None) -> str:
    if rsi_val is None:
        return "NEUTRAL"
    if rsi_val > 70:
        return "OVERBOUGHT"
    if rsi_val > 60:
        return "HIGH"
    if rsi_val >= 40:
        return "NEUTRAL"
    if rsi_val >= 30:
        return "LOW"
    return "OVERSOLD"


def _classify_macd(
    macd: pd.Series,
    signal: pd.Series,
    lookback: int = 3,
) -> str:
    cur_macd   = _last(macd)
    cur_signal = _last(signal)
    if cur_macd is None or cur_signal is None:
        return "NEUTRAL"

    # Check for recent crossover within last `lookback` bars
    recent_macd   = _last_n(macd,   lookback + 1)
    recent_signal = _last_n(signal, lookback + 1)

    if len(recent_macd) > 1 and len(recent_signal) > 1:
        # Align on common index
        aligned = pd.concat(
            [recent_macd.rename("m"), recent_signal.rename("s")], axis=1
        ).dropna()
        if len(aligned) >= 2:
            was_below = aligned["m"].iloc[0] < aligned["s"].iloc[0]
            is_above  = aligned["m"].iloc[-1] > aligned["s"].iloc[-1]
            was_above = aligned["m"].iloc[0] > aligned["s"].iloc[0]
            is_below  = aligned["m"].iloc[-1] < aligned["s"].iloc[-1]

            if was_below and is_above:
                return "BULLISH CROSSOVER"
            if was_above and is_below:
                return "BEARISH CROSSOVER"

    if cur_macd > cur_signal:
        return "BULLISH"
    if cur_macd < cur_signal:
        return "BEARISH"
    return "NEUTRAL"


def _classify_bbands(
    price: float,
    bbl: float | None,
    bbm: float | None,
    bbu: float | None,
) -> str:
    if any(v is None for v in (bbl, bbm, bbu)):
        return "MIDDLE"
    if price > bbu:
        return "ABOVE UPPER"
    if price < bbl:
        return "BELOW LOWER"
    upper_third = bbl + (bbu - bbl) * 2 / 3
    lower_third = bbl + (bbu - bbl) * 1 / 3
    if price >= upper_third:
        return "UPPER THIRD"
    if price <= lower_third:
        return "LOWER THIRD"
    return "MIDDLE"


def _classify_vix(vix_val: float | None) -> str:
    if vix_val is None:
        return "UNKNOWN"
    if vix_val > 30:
        return "EXTREME FEAR"
    if vix_val > 20:
        return "FEAR"
    if vix_val >= 15:
        return "NEUTRAL"
    return "COMPLACENCY"


def _composite_posture(scores: list[float]) -> str:
    if not scores:
        return "NEUTRAL"
    avg = sum(scores) / len(scores)
    for threshold, label in _POSTURE_THRESHOLDS:
        if avg >= threshold:
            return label
    return "BEARISH"


# ── Core analysis ──────────────────────────────────────────────────────────────

def analyze_ticker(ticker: str, df: pd.DataFrame) -> dict:
    """
    Run all technical indicators on a single ticker DataFrame.
    Returns a dict with every reading and classification.
    """
    result: dict = {"ticker": ticker, "error": None}

    try:
        if df is None or len(df) < 30:
            raise ValueError(f"Insufficient data: {len(df) if df is not None else 0} rows")

        price = _last(df["Close"])
        if price is None:
            raise ValueError("No close price available")

        result["price"] = price

        # ── Moving Averages ────────────────────────────────────────────────────
        ma20  = _last(df.ta.sma(length=20))
        ma50  = _last(df.ta.sma(length=50))
        ma200 = _last(df.ta.sma(length=200))

        result["ma"] = {
            "ma20":       ma20,
            "ma50":       ma50,
            "ma200":      ma200,
            "vs_20":      _classify_ma_position(price, ma20),
            "vs_50":      _classify_ma_position(price, ma50),
            "vs_200":     _classify_ma_position(price, ma200),
            "alignment":  _classify_ma_alignment(price, ma20, ma50, ma200),
        }

        # ── RSI ────────────────────────────────────────────────────────────────
        rsi_val = _last(df.ta.rsi(length=14))
        rsi_class = _classify_rsi(rsi_val)
        result["rsi"] = {"value": rsi_val, "classification": rsi_class}

        # ── MACD ───────────────────────────────────────────────────────────────
        macd_df   = df.ta.macd(fast=12, slow=26, signal=9)
        macd_line = macd_df["MACD_12_26_9"]
        sig_line  = macd_df["MACDs_12_26_9"]
        macd_class = _classify_macd(macd_line, sig_line)
        result["macd"] = {
            "macd":           _last(macd_line),
            "signal":         _last(sig_line),
            "histogram":      _last(macd_df["MACDh_12_26_9"]),
            "classification": macd_class,
        }

        # ── Bollinger Bands ────────────────────────────────────────────────────
        bb_df = df.ta.bbands(length=20, std=2)
        bbl   = _last(bb_df["BBL_20_2.0_2.0"])
        bbm   = _last(bb_df["BBM_20_2.0_2.0"])
        bbu   = _last(bb_df["BBU_20_2.0_2.0"])
        bbp   = _last(bb_df["BBP_20_2.0_2.0"])   # percent-B
        bb_class = _classify_bbands(price, bbl, bbm, bbu)
        result["bbands"] = {
            "lower":          bbl,
            "mid":            bbm,
            "upper":          bbu,
            "percent_b":      bbp,
            "classification": bb_class,
        }

        # ── ATR ────────────────────────────────────────────────────────────────
        atr_val = _last(df.ta.atr(length=14))
        result["atr"] = {
            "value":   atr_val,
            "pct":     round(atr_val / price * 100, 2) if atr_val and price else None,
        }

        # ── VIX-specific ───────────────────────────────────────────────────────
        if ticker == "^VIX":
            result["vix_classification"] = _classify_vix(price)

        # ── Composite posture ──────────────────────────────────────────────────
        scores = [
            _MA_ALIGN_SCORE.get(result["ma"]["alignment"],  0),
            _RSI_SCORE     .get(rsi_class,                  0),
            _MACD_SCORE    .get(macd_class,                 0),
            _BB_SCORE      .get(bb_class,                   0),
        ]
        result["posture_scores"] = scores
        result["posture"]        = _composite_posture(scores)

    except Exception as e:
        log.error(f"analyze_ticker({ticker}): {e}")
        result["error"] = str(e)
        result.setdefault("posture", "N/A")

    return result


def analyze_all(data_dict: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """
    Run analyze_ticker() on every ticker in data_dict.
    Returns {ticker: analysis_dict}.
    """
    results: dict[str, dict] = {}
    for ticker, df in data_dict.items():
        results[ticker] = analyze_ticker(ticker, df)
    return results


# ── Rich display ───────────────────────────────────────────────────────────────

def print_technical_summary(results: dict[str, dict]) -> None:
    """
    Print a Rich summary table:
    Ticker | Price | vs 200-MA | RSI | MACD | BB Position | Posture
    """
    table = Table(
        title="Taoist CIO — Layer 1: Technical Indicators",
        show_lines=False,
        header_style="bold cyan",
        border_style="dim",
    )
    table.add_column("Ticker",      style="cyan bold", no_wrap=True)
    table.add_column("Price",       justify="right")
    table.add_column("vs 200-MA",   justify="center")
    table.add_column("RSI",         justify="center")
    table.add_column("MACD",        justify="center")
    table.add_column("BB Position", justify="center")
    table.add_column("Posture",     justify="center")

    # Colour maps
    vs_ma_colors = {"ABOVE": "green", "AT": "yellow", "BELOW": "red", "N/A": "dim"}
    rsi_colors   = {
        "OVERBOUGHT": "bold green",
        "HIGH":       "green",
        "NEUTRAL":    "yellow",
        "LOW":        "red",
        "OVERSOLD":   "bold red",
    }
    macd_colors  = {
        "BULLISH CROSSOVER":  "bold green",
        "BULLISH":            "green",
        "NEUTRAL":            "yellow",
        "BEARISH":            "red",
        "BEARISH CROSSOVER":  "bold red",
    }
    bb_colors = {
        "ABOVE UPPER":  "bold green",
        "UPPER THIRD":  "green",
        "MIDDLE":       "yellow",
        "LOWER THIRD":  "red",
        "BELOW LOWER":  "bold red",
    }

    ok = err = 0
    for ticker, r in results.items():
        if r.get("error") and not r.get("price"):
            err += 1
            table.add_row(
                ticker, "—", "—", "—", "—", "—",
                "[dim]ERROR[/dim]",
            )
            continue

        ok += 1
        price = r.get("price")
        price_str = (
            f"{price:,.2f}" if price and price >= 1 else f"{price:.6f}" if price else "—"
        )

        ma      = r.get("ma", {})
        rsi     = r.get("rsi", {})
        macd    = r.get("macd", {})
        bb      = r.get("bbands", {})
        posture = r.get("posture", "N/A")

        vs200      = ma.get("vs_200", "N/A")
        rsi_class  = rsi.get("classification", "N/A")
        macd_class = macd.get("classification", "N/A")
        bb_class   = bb.get("classification", "MIDDLE")
        rsi_val    = rsi.get("value")

        # RSI label includes raw value
        rsi_str = (
            f"[{rsi_colors.get(rsi_class, 'white')}]{rsi_class}"
            + (f" ({rsi_val:.0f})" if rsi_val is not None else "")
            + f"[/{rsi_colors.get(rsi_class, 'white')}]"
        )

        # Special VIX label
        if ticker == "^VIX":
            posture = r.get("vix_classification", posture)

        table.add_row(
            ticker,
            price_str,
            f"[{vs_ma_colors.get(vs200, 'white')}]{vs200}[/{vs_ma_colors.get(vs200, 'white')}]",
            rsi_str,
            f"[{macd_colors.get(macd_class, 'white')}]{macd_class}[/{macd_colors.get(macd_class, 'white')}]",
            f"[{bb_colors.get(bb_class, 'yellow')}]{bb_class}[/{bb_colors.get(bb_class, 'yellow')}]",
            f"[{_POSTURE_COLORS.get(posture, 'white')}]{posture}[/{_POSTURE_COLORS.get(posture, 'white')}]",
        )

    console.print()
    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] [green]{ok} analysed[/green]  "
        f"[red]{err} error[/red]"
    )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import here to keep module importable without side-effects
    sys.path.insert(0, str(ROOT / "src"))
    from data_feed import get_all_data

    console.print("[bold cyan]Taoist CIO — Technical Analysis Layer[/bold cyan]")
    data    = get_all_data()
    results = analyze_all(data)
    print_technical_summary(results)
