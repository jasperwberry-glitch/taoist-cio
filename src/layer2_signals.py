"""
layer2_signals.py — Non-traditional signals layer (Layer 2) for the Taoist CIO dashboard.

Derives macro and cross-asset signals from data already in data_feed.py.
No additional API calls needed (except SPUT which requires manual input).

Signals:
  1. Copper/Gold Ratio       — growth vs defensive
  2. Silver/Gold Ratio       — risk-on vs risk-off
  3. Gold/Silver Ratio       — historical cheap/expensive signal
  4. HY Credit Spread Proxy  — HYG/LQD 30-day price ratio
  5. Put/Call Ratio          — ^PCALL (currently unavailable from Yahoo)
  6. Gold/Oil Ratio          — McClellan forward-looking indicator
  7. SPUT NAV Premium/Disc.  — manual input required
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

# ── Paths / logging ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
logging.basicConfig(
    filename=ROOT / "logs" / "layer2.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

# ── Status colours ─────────────────────────────────────────────────────────────
STATUS_STYLE = {
    "GREEN":   "bold green",
    "AMBER":   "bold yellow",
    "RED":     "bold red",
    "NEUTRAL": "dim",
    "N/A":     "dim",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _close(data: dict[str, pd.DataFrame], ticker: str) -> pd.Series | None:
    """Return the Close series for a ticker, or None if unavailable."""
    df = data.get(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df["Close"].sort_index().dropna()


def _ratio_series(data: dict, numerator: str, denominator: str) -> pd.Series | None:
    """Align two Close series and return their ratio."""
    num = _close(data, numerator)
    den = _close(data, denominator)
    if num is None or den is None:
        return None
    aligned = pd.concat([num.rename("n"), den.rename("d")], axis=1).dropna()
    if len(aligned) < 2:
        return None
    return aligned["n"] / aligned["d"]


def _pct_change_n_days(series: pd.Series, n: int) -> float | None:
    """Percentage change over last n rows."""
    if series is None or len(series) < n + 1:
        return None
    old = series.iloc[-(n + 1)]
    new = series.iloc[-1]
    if old == 0:
        return None
    return (new - old) / abs(old) * 100


def _make_signal(
    name: str,
    current_reading,
    threshold: str,
    status: str,
    interpretation: str,
    extra: dict | None = None,
) -> dict:
    sig = {
        "name":            name,
        "current_reading": current_reading,
        "threshold":       threshold,
        "status":          status,
        "interpretation":  interpretation,
    }
    if extra:
        sig.update(extra)
    return sig


# ── Individual signal functions ────────────────────────────────────────────────

def _copper_gold_ratio(data: dict) -> dict:
    """
    Copper/Gold Ratio = HG=F / GC=F
    Rising (current > 50-day avg) → GROWTH (GREEN)
    Falling                        → DEFENSIVE (RED)
    """
    name = "Copper/Gold Ratio"
    ratio = _ratio_series(data, "HG=F", "GC=F")
    if ratio is None or len(ratio) < 2:
        return _make_signal(name, "N/A", "current > 50-day avg", "N/A", "Data unavailable")

    current = float(ratio.iloc[-1])
    ma50    = float(ratio.tail(50).mean()) if len(ratio) >= 50 else float(ratio.mean())
    pct_vs_avg = (current - ma50) / ma50 * 100

    if current > ma50:
        status  = "GREEN"
        interp  = f"Rising — GROWTH signal (current {pct_vs_avg:+.2f}% above 50-day avg)"
        reading = "GROWTH"
    else:
        status  = "RED"
        interp  = f"Falling — DEFENSIVE signal (current {pct_vs_avg:+.2f}% below 50-day avg)"
        reading = "DEFENSIVE"

    return _make_signal(
        name, reading,
        "current vs 50-day avg of ratio",
        status, interp,
        {"ratio": round(current, 6), "ma50": round(ma50, 6)},
    )


def _silver_gold_ratio(data: dict) -> dict:
    """
    Silver/Gold Ratio = SI=F / GC=F
    Rising → RISK-ON (GREEN)
    Falling → RISK-OFF (RED)
    """
    name  = "Silver/Gold Ratio"
    ratio = _ratio_series(data, "SI=F", "GC=F")
    if ratio is None or len(ratio) < 2:
        return _make_signal(name, "N/A", "trend direction", "N/A", "Data unavailable")

    current  = float(ratio.iloc[-1])
    ma20     = float(ratio.tail(20).mean()) if len(ratio) >= 20 else float(ratio.mean())
    chg_30d  = _pct_change_n_days(ratio, 30)

    if chg_30d is None:
        trend = "UNKNOWN"
    elif chg_30d > 0:
        trend = "RISING"
    else:
        trend = "FALLING"

    if trend == "RISING":
        status = "GREEN"
        reading = "RISK-ON"
        interp = f"Silver outperforming gold (+{chg_30d:.2f}% over 30 days) — risk appetite present"
    elif trend == "FALLING":
        status = "RED"
        reading = "RISK-OFF"
        interp = f"Gold outperforming silver ({chg_30d:.2f}% over 30 days) — defensive rotation"
    else:
        status = "NEUTRAL"
        reading = "NEUTRAL"
        interp = "Trend indeterminate"

    return _make_signal(
        name, reading,
        "30-day trend direction",
        status, interp,
        {"ratio": round(current, 6), "30d_chg_pct": round(chg_30d, 2) if chg_30d else None},
    )


def _gold_silver_ratio(data: dict) -> dict:
    """
    Gold/Silver Ratio = GC=F / SI=F
    >80  → RED   (silver historically cheap — potential entry signal)
    <50  → GREEN (silver historically expensive)
    else → AMBER
    """
    name  = "Gold/Silver Ratio"
    ratio = _ratio_series(data, "GC=F", "SI=F")
    if ratio is None or len(ratio) < 1:
        return _make_signal(name, "N/A", ">80 RED | 50-80 AMBER | <50 GREEN", "N/A", "Data unavailable")

    current = float(ratio.iloc[-1])

    if current > 80:
        status = "RED"
        interp = (
            f"Ratio at {current:.1f} — ABOVE 80. Silver historically cheap vs gold. "
            "Potential silver entry signal — watch for mean reversion."
        )
    elif current < 50:
        status = "GREEN"
        interp = f"Ratio at {current:.1f} — below 50. Silver historically expensive relative to gold."
    else:
        status = "AMBER"
        interp = f"Ratio at {current:.1f} — within normal range (50–80)."

    return _make_signal(
        name, round(current, 2),
        ">80 RED (silver cheap) | 50–80 AMBER | <50 GREEN",
        status, interp,
    )


def _hy_credit_spread_proxy(data: dict) -> dict:
    """
    HY Credit Spread Proxy: HYG/LQD price ratio, 30-day change.
    Declining >2%  → RED   (credit stress)
    Declining 1–2% → AMBER
    Stable/rising  → GREEN
    """
    name  = "HY Credit Spread Proxy"
    ratio = _ratio_series(data, "HYG", "LQD")
    if ratio is None or len(ratio) < 31:
        return _make_signal(
            name, "N/A",
            "30-day HYG/LQD change: decline >2% RED | 1-2% AMBER | stable GREEN",
            "N/A", "Insufficient data (need 31+ days)",
        )

    current = float(ratio.iloc[-1])
    chg_30d = _pct_change_n_days(ratio, 30)

    if chg_30d is None:
        return _make_signal(name, "N/A", "", "N/A", "Cannot compute 30-day change")

    if chg_30d < -2.0:
        status = "RED"
        interp = (
            f"HYG/LQD ratio fell {chg_30d:.2f}% over 30 days — "
            "credit spreads likely widening. STRESS signal."
        )
    elif chg_30d < -1.0:
        status = "AMBER"
        interp = (
            f"HYG/LQD ratio fell {chg_30d:.2f}% over 30 days — "
            "mild spread widening. Monitor for deterioration."
        )
    else:
        status = "GREEN"
        interp = (
            f"HYG/LQD ratio {chg_30d:+.2f}% over 30 days — "
            "credit conditions stable or tightening."
        )

    return _make_signal(
        name, f"{chg_30d:+.2f}% (30d)",
        "30-day HYG/LQD change: <−2% RED | −1–2% AMBER | stable/rising GREEN",
        status, interp,
        {"current_ratio": round(current, 6), "30d_chg_pct": round(chg_30d, 3)},
    )


def _put_call_ratio(data: dict) -> dict:
    """
    Put/Call Ratio via ^PCALL.
    >1.2  → RED   (extreme fear / potential bottom)
    0.9–1.2 → AMBER
    0.7–0.9 → NEUTRAL
    <0.7  → GREEN (complacency / caution)

    NOTE: ^PCALL is currently unavailable from Yahoo Finance.
    Signal returns N/A with explanation.
    """
    name = "Put/Call Ratio (^PCALL)"
    series = _close(data, "^PCALL")

    if series is None or series.empty:
        return _make_signal(
            name, "N/A",
            ">1.2 RED | 0.9–1.2 AMBER | 0.7–0.9 NEUTRAL | <0.7 GREEN",
            "N/A",
            "^PCALL delisted from Yahoo Finance. "
            "Source manually from CBOE (cboe.com/data/volatility-indexes) "
            "or replace with ^SKEW / CPCE.",
        )

    current = float(series.iloc[-1])
    if current > 1.2:
        status = "RED"
        interp = f"P/C ratio {current:.2f} — extreme fear. Potential contrarian buy signal."
    elif current >= 0.9:
        status = "AMBER"
        interp = f"P/C ratio {current:.2f} — elevated caution."
    elif current >= 0.7:
        status = "NEUTRAL"
        interp = f"P/C ratio {current:.2f} — normal range."
    else:
        status = "GREEN"
        interp = f"P/C ratio {current:.2f} — complacency. Monitor for reversal risk."

    return _make_signal(
        name, round(current, 3),
        ">1.2 RED | 0.9–1.2 AMBER | 0.7–0.9 NEUTRAL | <0.7 GREEN",
        status, interp,
    )


def _gold_oil_ratio(data: dict) -> dict:
    """
    Gold/Oil Ratio = GC=F / BZ=F
    McClellan insight: gold leads oil by 9–12 months.
    Report current ratio + 52-week high/low context + trend direction.
    """
    name  = "Gold/Oil Ratio"
    ratio = _ratio_series(data, "GC=F", "BZ=F")
    if ratio is None or len(ratio) < 10:
        return _make_signal(name, "N/A", "52-week range context", "N/A", "Data unavailable")

    current  = float(ratio.iloc[-1])
    high_52w = float(ratio.max())
    low_52w  = float(ratio.min())
    pct_from_high = (current - high_52w) / high_52w * 100
    pct_from_low  = (current - low_52w)  / low_52w  * 100
    chg_30d  = _pct_change_n_days(ratio, 30)
    trend    = "RISING" if (chg_30d or 0) > 0 else "FALLING"

    # Near 52-week high = gold very expensive vs oil → status reflects that
    if pct_from_high > -3:
        status = "RED"   # at 52-week high — gold priced for significant oil rally ahead
        range_note = "AT 52-WEEK HIGH"
    elif pct_from_low < 3:
        status = "GREEN"
        range_note = "AT 52-WEEK LOW"
    else:
        status = "AMBER"
        range_note = f"{pct_from_low:.1f}% off 52w low | {pct_from_high:.1f}% off 52w high"

    interp = (
        f"Ratio {current:.2f} — {trend} ({chg_30d:+.2f}% over 30d). "
        f"McClellan: elevated gold/oil ratio historically precedes oil outperformance "
        f"by 9–12 months. 52w range: {low_52w:.2f}–{high_52w:.2f}."
    )

    return _make_signal(
        name, round(current, 2),
        "52-week range context + 30-day trend",
        status, interp,
        {
            "52w_high": round(high_52w, 2),
            "52w_low":  round(low_52w,  2),
            "range_note": range_note,
            "trend":    trend,
            "30d_chg":  round(chg_30d, 2) if chg_30d else None,
        },
    )


def _sput_nav_premium(premium_pct: float | None) -> dict:
    """
    SPUT (Sprott Physical Uranium Trust) NAV premium/discount.
    Requires manual input — not derivable from price data alone.

    premium_pct: positive = trading at premium, negative = trading at discount.
    >+5%  → RED   (overpriced, avoid)
    <−2%  → GREEN (buying below NAV — attractive entry)
    else  → AMBER
    None  → MANUAL INPUT REQUIRED
    """
    name = "SPUT NAV Premium/Discount"
    threshold = "<−2% GREEN (discount) | −2%–+5% AMBER | >+5% RED (premium)"

    if premium_pct is None:
        return _make_signal(
            name, "MANUAL INPUT REQUIRED",
            threshold, "N/A",
            "Pass sput_nav_premium=<float> to get_layer2_signals(). "
            "Source from Sprott website or Bloomberg (SRUUF vs SPUT NAV).",
        )

    if premium_pct < -2.0:
        status = "GREEN"
        interp = (
            f"SPUT at {premium_pct:+.2f}% discount to NAV — "
            "physical uranium available below trust value. Attractive entry."
        )
    elif premium_pct > 5.0:
        status = "RED"
        interp = (
            f"SPUT at {premium_pct:+.2f}% premium to NAV — "
            "overpriced relative to physical uranium. Avoid chasing."
        )
    else:
        status = "AMBER"
        interp = (
            f"SPUT at {premium_pct:+.2f}% vs NAV — "
            "within normal range. Monitor for discount entry opportunity."
        )

    return _make_signal(
        name, f"{premium_pct:+.2f}%", threshold, status, interp,
        {"premium_pct": premium_pct},
    )


# ── Public API ─────────────────────────────────────────────────────────────────

def get_layer2_signals(
    data: dict[str, pd.DataFrame],
    sput_nav_premium: float | None = None,
) -> dict[str, dict]:
    """
    Compute all Layer 2 signals from a data_feed.get_all_data() result.

    Args:
        data:             {ticker: DataFrame} from data_feed.get_all_data()
        sput_nav_premium: SPUT premium to NAV as a float percentage
                          (positive = premium, negative = discount).
                          Pass None to mark as manual input required.

    Returns:
        {signal_key: signal_dict} where each signal_dict contains:
        name, current_reading, threshold, status, interpretation
    """
    signals: dict[str, dict] = {}
    runners = [
        ("copper_gold",       lambda: _copper_gold_ratio(data)),
        ("silver_gold",       lambda: _silver_gold_ratio(data)),
        ("gold_silver",       lambda: _gold_silver_ratio(data)),
        ("credit_spread",     lambda: _hy_credit_spread_proxy(data)),
        ("put_call",          lambda: _put_call_ratio(data)),
        ("gold_oil",          lambda: _gold_oil_ratio(data)),
        ("sput_nav",          lambda: _sput_nav_premium(sput_nav_premium)),
    ]
    for key, fn in runners:
        try:
            signals[key] = fn()
        except Exception as e:
            log.error(f"Layer2 signal '{key}' failed: {e}")
            signals[key] = _make_signal(
                key, "ERROR", "", "N/A", str(e)
            )
    return signals


# ── Rich display ───────────────────────────────────────────────────────────────

def print_layer2_summary(signals: dict[str, dict]) -> None:
    """Print a colour-coded Rich table of all Layer 2 signals."""
    table = Table(
        title="Taoist CIO — Layer 2: Non-Traditional Signals",
        show_lines=True,
        header_style="bold cyan",
        border_style="dim",
        min_width=90,
    )
    table.add_column("Signal",          style="bold", no_wrap=True, min_width=26)
    table.add_column("Reading",         justify="center", min_width=16)
    table.add_column("Status",          justify="center", min_width=8)
    table.add_column("Interpretation",  min_width=38)

    for sig in signals.values():
        status   = sig.get("status", "N/A")
        s_style  = STATUS_STYLE.get(status, "white")
        reading  = str(sig.get("current_reading", "—"))
        interp   = sig.get("interpretation", "")

        # Truncate long interpretations for the table; full text in dict
        if len(interp) > 120:
            interp = interp[:117] + "…"

        table.add_row(
            sig.get("name", "—"),
            f"[{s_style}]{reading}[/{s_style}]",
            f"[{s_style}]{status}[/{s_style}]",
            interp,
        )

    console.print()
    console.print(table)

    # Highlight any critical flags
    alerts = [
        s for s in signals.values()
        if s.get("status") == "RED" and s.get("status") != "N/A"
    ]
    if alerts:
        console.print("\n[bold red]⚑ Active RED signals:[/bold red]")
        for a in alerts:
            console.print(f"  [red]•[/red] {a['name']}: {a['interpretation'][:100]}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from data_feed import get_all_data

    console.print("[bold cyan]Taoist CIO — Layer 2: Non-Traditional Signals[/bold cyan]")
    data    = get_all_data()
    signals = get_layer2_signals(data)
    print_layer2_summary(signals)

    # Print full interpretation for each signal (no truncation)
    console.print("\n[bold]Full signal detail:[/bold]")
    for key, sig in signals.items():
        console.print(f"\n[cyan]{sig['name']}[/cyan]")
        console.print(f"  Reading:   {sig['current_reading']}")
        console.print(f"  Status:    [{STATUS_STYLE.get(sig['status'], 'white')}]{sig['status']}[/{STATUS_STYLE.get(sig['status'], 'white')}]")
        console.print(f"  Threshold: {sig['threshold']}")
        console.print(f"  Detail:    {sig['interpretation']}")
