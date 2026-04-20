"""
mason_signals.py — Fundamental threshold signals (Layer 1B) for the Taoist CIO dashboard.

Checks live prices against Gregory's Signal Tracker v2 thresholds.
Each signal returns: asset, signal description, threshold, current value,
status (GREEN / AMBER / RED / NO SIGNAL), and an action note.

Sections:
  Natural Resources — Gold, Silver, Copper, Uranium, Water
  Equity           — SPY 200MA, VIX panic level
  Crypto           — BTC 200MA
  Macro            — 10-year yield, DXY
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

try:
    import pandas_ta as _pta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

# ── Paths / logging ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
logging.basicConfig(
    filename=ROOT / "logs" / "mason_signals.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

# ── Status colours ─────────────────────────────────────────────────────────────
STATUS_STYLE = {
    "GREEN":     "bold green",
    "AMBER":     "bold yellow",
    "RED":       "bold red",
    "NO SIGNAL": "dim",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _close(data: dict, ticker: str) -> pd.Series | None:
    df = data.get(ticker)
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return df["Close"].sort_index().dropna()


def _last(s: pd.Series | None) -> float | None:
    if s is None or s.empty:
        return None
    return float(s.iloc[-1])


def _52w_high(s: pd.Series | None) -> float | None:
    if s is None or s.empty:
        return None
    return float(s.max())


def _pullback_pct(current: float, high: float) -> float:
    """How far current is below the 52-week high, as a positive percentage."""
    if high == 0:
        return 0.0
    return (high - current) / high * 100


def _sma(s: pd.Series, length: int) -> float | None:
    if s is None or len(s) < length:
        return None
    if PANDAS_TA_AVAILABLE:
        result = _pta.sma(s, length=length)
    else:
        result = s.rolling(window=length).mean()
    if result is None or result.dropna().empty:
        return None
    return float(result.dropna().iloc[-1])


def _rsi(s: pd.Series, length: int = 14) -> float | None:
    if s is None or len(s) < length + 1:
        return None
    if PANDAS_TA_AVAILABLE:
        result = _pta.rsi(s, length=length)
    else:
        delta    = s.diff()
        gain     = delta.clip(lower=0)
        loss     = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
        rs       = avg_gain / avg_loss.replace(0, np.nan)
        result   = 100 - (100 / (1 + rs))
    if result is None or result.dropna().empty:
        return None
    return float(result.dropna().iloc[-1])


def _sig(
    asset: str,
    signal: str,
    threshold_value: str,
    current_value: str,
    status: str,
    action_note: str,
) -> dict:
    return {
        "asset":             asset,
        "signal":            signal,
        "threshold_value":   threshold_value,
        "current_value":     current_value,
        "status":            status,
        "action_note":       action_note,
    }


def _unavailable(asset: str, signal: str) -> dict:
    return _sig(asset, signal, "—", "N/A", "NO SIGNAL", "Data unavailable")


# ── Natural Resources ──────────────────────────────────────────────────────────

def _gold_signals(data: dict) -> list[dict]:
    signals = []
    s = _close(data, "GC=F")
    if s is None:
        return [_unavailable("Gold", "All gold signals")]

    price   = _last(s)
    high_52 = _52w_high(s)
    pb_pct  = _pullback_pct(price, high_52) if (price and high_52) else None
    rsi_val = _rsi(s)

    # 1. Price below $4,200
    if price is not None:
        if price < 4200:
            signals.append(_sig(
                "Gold", "Spot price entry zone",
                "< $4,200", f"${price:,.0f}",
                "GREEN",
                "Below $4,200 threshold — core entry zone active.",
            ))
        else:
            signals.append(_sig(
                "Gold", "Spot price entry zone",
                "< $4,200", f"${price:,.0f}",
                "NO SIGNAL",
                f"${price:,.0f} is above $4,200 entry threshold.",
            ))

    # 2. Pullback 8–15% from 52-week high
    if pb_pct is not None and high_52 is not None:
        if 8 <= pb_pct <= 15:
            signals.append(_sig(
                "Gold", "Pullback from 52w high",
                "8–15% pullback", f"{pb_pct:.1f}% off ${high_52:,.0f}",
                "GREEN",
                f"Down {pb_pct:.1f}% from 52w high (${high_52:,.0f}) — thesis-intact pullback entry.",
            ))
        elif pb_pct > 15:
            signals.append(_sig(
                "Gold", "Pullback from 52w high",
                "8–15% pullback", f"{pb_pct:.1f}% off ${high_52:,.0f}",
                "AMBER",
                f"Down {pb_pct:.1f}% from 52w high — exceeds normal pullback range, verify thesis.",
            ))
        else:
            signals.append(_sig(
                "Gold", "Pullback from 52w high",
                "8–15% pullback", f"{pb_pct:.1f}% off ${high_52:,.0f}",
                "NO SIGNAL",
                f"Only {pb_pct:.1f}% off 52w high — no meaningful pullback yet.",
            ))

    # 3. RSI oversold
    if rsi_val is not None:
        if rsi_val < 30:
            signals.append(_sig(
                "Gold", "RSI oversold",
                "RSI < 30", f"RSI {rsi_val:.1f}",
                "GREEN",
                f"RSI {rsi_val:.1f} — oversold. Combine with price levels for entry.",
            ))
        else:
            signals.append(_sig(
                "Gold", "RSI oversold",
                "RSI < 30", f"RSI {rsi_val:.1f}",
                "NO SIGNAL",
                f"RSI {rsi_val:.1f} — not oversold.",
            ))

    return signals


def _silver_signals(data: dict) -> list[dict]:
    signals = []
    s_silver = _close(data, "SI=F")
    s_gold   = _close(data, "GC=F")

    if s_silver is None or s_gold is None:
        return [_unavailable("Silver", "All silver signals")]

    silver_price = _last(s_silver)
    gold_price   = _last(s_gold)
    rsi_val      = _rsi(s_silver)

    # 1. Gold/Silver ratio above 80
    if silver_price and gold_price and silver_price > 0:
        gs_ratio = gold_price / silver_price
        if gs_ratio > 80:
            signals.append(_sig(
                "Silver", "Gold/Silver ratio",
                "> 80 (silver cheap)", f"{gs_ratio:.1f}",
                "GREEN",
                f"G/S ratio {gs_ratio:.1f} — silver historically cheap vs gold. Entry signal.",
            ))
        else:
            signals.append(_sig(
                "Silver", "Gold/Silver ratio",
                "> 80 (silver cheap)", f"{gs_ratio:.1f}",
                "NO SIGNAL",
                f"G/S ratio {gs_ratio:.1f} — within normal range (below 80 threshold).",
            ))

    # 2. RSI extreme oversold < 25
    if rsi_val is not None:
        if rsi_val < 25:
            signals.append(_sig(
                "Silver", "RSI extreme oversold",
                "RSI < 25", f"RSI {rsi_val:.1f}",
                "GREEN",
                f"RSI {rsi_val:.1f} — extreme oversold. High-conviction entry signal.",
            ))
        else:
            signals.append(_sig(
                "Silver", "RSI extreme oversold",
                "RSI < 25", f"RSI {rsi_val:.1f}",
                "NO SIGNAL",
                f"RSI {rsi_val:.1f} — not at extreme oversold levels.",
            ))

    return signals


def _copper_signals(data: dict) -> list[dict]:
    s = _close(data, "HG=F")
    if s is None:
        return [_unavailable("Copper", "Price entry levels")]

    price = _last(s)
    if price is None:
        return [_unavailable("Copper", "Price entry levels")]

    if price < 4.50:
        return [_sig(
            "Copper", "Price accelerated entry",
            "< $4.50/lb", f"${price:.4f}",
            "GREEN",
            f"${price:.4f}/lb — below $4.50 accelerated entry level. Strong buy zone.",
        )]
    elif price < 5.00:
        return [_sig(
            "Copper", "Price entry caution",
            "< $5.00/lb", f"${price:.4f}",
            "AMBER",
            f"${price:.4f}/lb — between $4.50–$5.00. Entry caution zone; scale in carefully.",
        )]
    else:
        return [_sig(
            "Copper", "Price entry levels",
            "< $5.00 AMBER | < $4.50 GREEN", f"${price:.4f}",
            "NO SIGNAL",
            f"${price:.4f}/lb — above $5.00 threshold. No entry signal.",
        )]


def _uranium_signals(sput_nav_premium: float | None) -> list[dict]:
    if sput_nav_premium is None:
        return [_sig(
            "Uranium", "SPUT NAV premium/discount",
            "Discount > 2% = GREEN",
            "MANUAL INPUT REQUIRED",
            "NO SIGNAL",
            "Pass sput_nav_premium=<float> to get_fundamental_signals(). "
            "Source from Sprott website.",
        )]

    if sput_nav_premium < -2.0:
        return [_sig(
            "Uranium", "SPUT NAV premium/discount",
            "Discount > 2% = GREEN",
            f"{sput_nav_premium:+.2f}%",
            "GREEN",
            f"SPUT at {sput_nav_premium:+.2f}% discount to NAV — physical uranium entry signal.",
        )]
    else:
        return [_sig(
            "Uranium", "SPUT NAV premium/discount",
            "Discount > 2% = GREEN",
            f"{sput_nav_premium:+.2f}%",
            "NO SIGNAL",
            f"SPUT at {sput_nav_premium:+.2f}% vs NAV — no meaningful discount to exploit.",
        )]


def _water_signals(data: dict) -> list[dict]:
    signals = []
    s_awk = _close(data, "AWK")
    s_tnx = _close(data, "^TNX")

    if s_awk is None or s_tnx is None:
        return [_unavailable("Water", "AWK rate-driven entry")]

    awk_price = _last(s_awk)
    awk_high  = _52w_high(s_awk)
    tnx_val   = _last(s_tnx)

    if awk_price is None or awk_high is None or tnx_val is None:
        return [_unavailable("Water", "AWK rate-driven entry")]

    awk_pb = _pullback_pct(awk_price, awk_high)

    # 1. AWK down 10%+ from 52w high AND yield > 5.0%
    if awk_pb >= 10 and tnx_val > 5.0:
        signals.append(_sig(
            "Water (AWK)", "Rate-driven entry",
            "AWK −10%+ from 52w high AND ^TNX > 5%",
            f"AWK −{awk_pb:.1f}% | ^TNX {tnx_val:.2f}%",
            "GREEN",
            f"AWK off {awk_pb:.1f}% from 52w high (${awk_high:.2f}) with 10y yield at "
            f"{tnx_val:.2f}%. Classic rate-driven utility entry signal.",
        ))
    else:
        missing = []
        if awk_pb < 10:
            missing.append(f"AWK only {awk_pb:.1f}% below 52w high (need 10%+)")
        if tnx_val <= 5.0:
            missing.append(f"^TNX at {tnx_val:.2f}% (need > 5.00%)")
        signals.append(_sig(
            "Water (AWK)", "Rate-driven entry",
            "AWK −10%+ from 52w high AND ^TNX > 5%",
            f"AWK −{awk_pb:.1f}% | ^TNX {tnx_val:.2f}%",
            "NO SIGNAL",
            "Conditions not met: " + "; ".join(missing) + ".",
        ))

    # 2. Yield above 4.75% — rate headwind watch
    if tnx_val > 5.0:
        signals.append(_sig(
            "Water (AWK)", "Rate headwind — RED",
            "^TNX > 5.00%", f"{tnx_val:.2f}%",
            "RED",
            f"10-year yield {tnx_val:.2f}% — above 5%. Significant rate headwind for utilities.",
        ))
    elif tnx_val > 4.75:
        signals.append(_sig(
            "Water (AWK)", "Rate headwind — watch",
            "^TNX > 4.75%", f"{tnx_val:.2f}%",
            "AMBER",
            f"10-year yield {tnx_val:.2f}% — above 4.75% watch level. Rate headwind building.",
        ))
    else:
        signals.append(_sig(
            "Water (AWK)", "Rate headwind — watch",
            "^TNX > 4.75%", f"{tnx_val:.2f}%",
            "NO SIGNAL",
            f"10-year yield {tnx_val:.2f}% — below 4.75% watch level. Rate backdrop benign.",
        ))

    return signals


# ── Equity ─────────────────────────────────────────────────────────────────────

def _equity_signals(data: dict) -> list[dict]:
    signals = []

    # SPY vs 200-day MA
    s_spy = _close(data, "SPY")
    if s_spy is not None:
        price  = _last(s_spy)
        ma200  = _sma(s_spy, 200)
        if price is not None and ma200 is not None:
            pct_vs = (price - ma200) / ma200 * 100
            if price < ma200:
                signals.append(_sig(
                    "S&P 500 (SPY)", "200-day MA bear warning",
                    "SPY < 200MA = RED",
                    f"${price:.2f} vs MA ${ma200:.2f} ({pct_vs:+.1f}%)",
                    "RED",
                    f"SPY {pct_vs:.1f}% below 200-day MA (${ma200:.2f}). Bear market signal — "
                    "reduce risk, avoid new longs.",
                ))
            else:
                signals.append(_sig(
                    "S&P 500 (SPY)", "200-day MA bear warning",
                    "SPY < 200MA = RED",
                    f"${price:.2f} vs MA ${ma200:.2f} ({pct_vs:+.1f}%)",
                    "NO SIGNAL",
                    f"SPY {pct_vs:+.1f}% above 200-day MA. Uptrend intact.",
                ))
    else:
        signals.append(_unavailable("S&P 500 (SPY)", "200-day MA bear warning"))

    # VIX above 30
    s_vix = _close(data, "^VIX")
    if s_vix is not None:
        vix = _last(s_vix)
        if vix is not None:
            if vix > 30:
                signals.append(_sig(
                    "VIX", "Panic / potential low",
                    "^VIX > 30 = RED",
                    f"{vix:.2f}",
                    "RED",
                    f"VIX {vix:.2f} — above 30. Panic conditions. Historically near-term "
                    "market lows. Contrarian entry watch.",
                ))
            else:
                signals.append(_sig(
                    "VIX", "Panic / potential low",
                    "^VIX > 30 = RED",
                    f"{vix:.2f}",
                    "NO SIGNAL",
                    f"VIX {vix:.2f} — below 30 panic threshold.",
                ))
    else:
        signals.append(_unavailable("VIX", "Panic level"))

    return signals


# ── Crypto ─────────────────────────────────────────────────────────────────────

def _crypto_signals(data: dict) -> list[dict]:
    s_btc = _close(data, "BTC-USD")
    if s_btc is None:
        return [_unavailable("BTC", "200-day MA bear warning")]

    price = _last(s_btc)
    ma200 = _sma(s_btc, 200)
    if price is None or ma200 is None:
        return [_unavailable("BTC", "200-day MA bear warning")]

    pct_vs = (price - ma200) / ma200 * 100
    if price < ma200:
        return [_sig(
            "BTC", "200-day MA crypto bear warning",
            "BTC < 200MA = RED",
            f"${price:,.0f} vs MA ${ma200:,.0f} ({pct_vs:+.1f}%)",
            "RED",
            f"BTC {pct_vs:.1f}% below 200-day MA (${ma200:,.0f}). Crypto bear signal — "
            "avoid new entries; size accordingly.",
        )]
    else:
        return [_sig(
            "BTC", "200-day MA crypto bear warning",
            "BTC < 200MA = RED",
            f"${price:,.0f} vs MA ${ma200:,.0f} ({pct_vs:+.1f}%)",
            "NO SIGNAL",
            f"BTC {pct_vs:+.1f}% above 200-day MA. Crypto uptrend intact.",
        )]


# ── Macro ──────────────────────────────────────────────────────────────────────

def _macro_signals(data: dict) -> list[dict]:
    signals = []

    # 10-year yield
    s_tnx = _close(data, "^TNX")
    if s_tnx is not None:
        tnx = _last(s_tnx)
        if tnx is not None:
            if tnx > 5.0:
                signals.append(_sig(
                    "10-Year Yield (^TNX)", "Rate pressure — RED",
                    "> 5.00% = RED",
                    f"{tnx:.2f}%",
                    "RED",
                    f"10-year yield {tnx:.2f}% — above 5%. High rate pressure on equities, "
                    "real estate, and growth assets.",
                ))
            elif tnx > 4.50:
                signals.append(_sig(
                    "10-Year Yield (^TNX)", "Rate pressure — AMBER",
                    "> 4.50% = AMBER",
                    f"{tnx:.2f}%",
                    "AMBER",
                    f"10-year yield {tnx:.2f}% — above 4.50% caution zone. Rate pressure "
                    "building; watch for impact on rate-sensitive assets.",
                ))
            else:
                signals.append(_sig(
                    "10-Year Yield (^TNX)", "Rate pressure",
                    "> 4.50% AMBER | > 5.00% RED",
                    f"{tnx:.2f}%",
                    "NO SIGNAL",
                    f"10-year yield {tnx:.2f}% — below 4.50% threshold. Benign rate backdrop.",
                ))
    else:
        signals.append(_unavailable("10-Year Yield (^TNX)", "Rate pressure"))

    # DXY above 105
    s_dxy = _close(data, "DX-Y.NYB")
    if s_dxy is not None:
        dxy = _last(s_dxy)
        if dxy is not None:
            if dxy > 105:
                signals.append(_sig(
                    "DXY (DX-Y.NYB)", "Dollar strength headwind",
                    "> 105 = AMBER",
                    f"{dxy:.2f}",
                    "AMBER",
                    f"DXY {dxy:.2f} — above 105. Strong dollar creates headwind for "
                    "commodities, EM, and gold priced in non-USD currencies.",
                ))
            else:
                signals.append(_sig(
                    "DXY (DX-Y.NYB)", "Dollar strength headwind",
                    "> 105 = AMBER",
                    f"{dxy:.2f}",
                    "NO SIGNAL",
                    f"DXY {dxy:.2f} — below 105. No dollar headwind signal.",
                ))
    else:
        signals.append(_unavailable("DXY (DX-Y.NYB)", "Dollar strength headwind"))

    return signals


def _manual_macro_signals(
    ted_spread_bps: float | None = None,
    gs_risk_appetite: float | None = None,
) -> list[dict]:
    """
    Manual-input macro signals that require external data sources.
    Pass values directly; default to N/A (like SPUT NAV).
    """
    signals = []

    # ── TED Spread (3-month LIBOR minus 3-month T-bill) ───────────────────────
    # Source: FRED TEDRATE or Bloomberg. Update manually.
    threshold_ted = "> 50bps AMBER | > 100bps RED"
    if ted_spread_bps is None:
        signals.append(_sig(
            "TED Spread", "Credit stress indicator",
            threshold_ted, "MANUAL INPUT REQUIRED", "NO SIGNAL",
            "Pass ted_spread_bps=<float> to get_fundamental_signals(). "
            "Source from FRED (fred.stlouisfed.org/series/TEDRATE) or Bloomberg.",
        ))
    elif ted_spread_bps > 100:
        signals.append(_sig(
            "TED Spread", "Credit stress indicator",
            threshold_ted, f"{ted_spread_bps:.0f}bps", "RED",
            f"TED Spread {ted_spread_bps:.0f}bps — above 100bps. "
            "Significant interbank stress. Risk-off signal.",
        ))
    elif ted_spread_bps > 50:
        signals.append(_sig(
            "TED Spread", "Credit stress indicator",
            threshold_ted, f"{ted_spread_bps:.0f}bps", "AMBER",
            f"TED Spread {ted_spread_bps:.0f}bps — above 50bps caution level. "
            "Mild credit stress emerging. Monitor.",
        ))
    else:
        signals.append(_sig(
            "TED Spread", "Credit stress indicator",
            threshold_ted, f"{ted_spread_bps:.0f}bps", "NO SIGNAL",
            f"TED Spread {ted_spread_bps:.0f}bps — benign. No credit stress signal.",
        ))

    # ── Goldman Sachs Risk Appetite Indicator ─────────────────────────────────
    # Source: GS Global Investment Research (manual / subscription).
    # Convention: < -2.0 = extreme low (fear); > +2.0 = extreme high (greed).
    threshold_gs = "Extreme low (< -2.0) = AMBER"
    if gs_risk_appetite is None:
        signals.append(_sig(
            "GS Risk Appetite", "Sentiment extreme indicator",
            threshold_gs, "MANUAL INPUT REQUIRED", "NO SIGNAL",
            "Pass gs_risk_appetite=<float> to get_fundamental_signals(). "
            "Source from Goldman Sachs Global Investment Research.",
        ))
    elif gs_risk_appetite < -2.0:
        signals.append(_sig(
            "GS Risk Appetite", "Sentiment extreme indicator",
            threshold_gs, f"{gs_risk_appetite:.2f}", "AMBER",
            f"GS Risk Appetite {gs_risk_appetite:.2f} — extreme low. "
            "Broad fear/risk aversion. Historically near medium-term lows.",
        ))
    else:
        signals.append(_sig(
            "GS Risk Appetite", "Sentiment extreme indicator",
            threshold_gs, f"{gs_risk_appetite:.2f}", "NO SIGNAL",
            f"GS Risk Appetite {gs_risk_appetite:.2f} — not at extreme low.",
        ))

    return signals


# ── Public API ─────────────────────────────────────────────────────────────────

def get_fundamental_signals(
    data: dict[str, pd.DataFrame],
    sput_nav_premium: float | None = None,
    ted_spread_bps: float | None = None,
    gs_risk_appetite: float | None = None,
) -> list[dict]:
    """
    Run all Signal Tracker v5 threshold checks against live data.

    Args:
        data:             {ticker: DataFrame} from data_feed.get_all_data()
        sput_nav_premium: SPUT NAV premium/discount as a float percentage.
                          Negative = discount. Pass None to mark as manual input required.
        ted_spread_bps:   TED Spread in basis points. Pass None for manual input.
        gs_risk_appetite: GS Risk Appetite Indicator value. Pass None for manual input.

    Returns:
        List of signal dicts, each with:
        asset, signal, threshold_value, current_value, status, action_note
    """
    all_signals: list[dict] = []
    runners = [
        lambda: _gold_signals(data),
        lambda: _silver_signals(data),
        lambda: _copper_signals(data),
        lambda: _uranium_signals(sput_nav_premium),
        lambda: _water_signals(data),
        lambda: _equity_signals(data),
        lambda: _crypto_signals(data),
        lambda: _macro_signals(data),
        lambda: _manual_macro_signals(ted_spread_bps, gs_risk_appetite),
    ]
    for fn in runners:
        try:
            all_signals.extend(fn())
        except Exception as e:
            log.error(f"Signal function failed: {e}")

    return all_signals


# ── Rich display ───────────────────────────────────────────────────────────────

def print_fundamental_summary(signals: list[dict]) -> None:
    """Print colour-coded Rich table of all fundamental threshold signals."""
    table = Table(
        title="Taoist CIO — Layer 1B: Fundamental Threshold Signals",
        show_lines=True,
        header_style="bold cyan",
        border_style="dim",
        min_width=110,
    )
    table.add_column("Asset",           style="bold", no_wrap=True,  min_width=20)
    table.add_column("Signal",          no_wrap=False,               min_width=24)
    table.add_column("Current",         justify="right",             min_width=14)
    table.add_column("Threshold",       justify="center",            min_width=18)
    table.add_column("Status",          justify="center",            min_width=10)
    table.add_column("Action Note",                                  min_width=30)

    counts = {"GREEN": 0, "AMBER": 0, "RED": 0, "NO SIGNAL": 0}

    for s in signals:
        status   = s.get("status", "NO SIGNAL")
        s_style  = STATUS_STYLE.get(status, "dim")
        note     = s.get("action_note", "")
        counts[status] = counts.get(status, 0) + 1

        table.add_row(
            s.get("asset",           "—"),
            s.get("signal",          "—"),
            s.get("current_value",   "—"),
            s.get("threshold_value", "—"),
            f"[{s_style}]{status}[/{s_style}]",
            note,
        )

    console.print()
    console.print(table)

    # Summary counts
    console.print(
        f"\n[bold]Signal count:[/bold]  "
        f"[bold green]{counts.get('GREEN', 0)} GREEN[/bold green]  "
        f"[bold yellow]{counts.get('AMBER', 0)} AMBER[/bold yellow]  "
        f"[bold red]{counts.get('RED', 0)} RED[/bold red]  "
        f"[dim]{counts.get('NO SIGNAL', 0)} NO SIGNAL[/dim]"
    )

    # Active entry signals (GREEN)
    active_green = [s for s in signals if s.get("status") == "GREEN"]
    if active_green:
        console.print("\n[bold green]Active entry signals (GREEN):[/bold green]")
        for s in active_green:
            console.print(f"  [green]✓[/green] [{s['asset']}] {s['signal']}: {s['action_note']}")

    # Active warnings (RED)
    active_red = [s for s in signals if s.get("status") == "RED"]
    if active_red:
        console.print("\n[bold red]Active warnings (RED):[/bold red]")
        for s in active_red:
            console.print(f"  [red]⚑[/red] [{s['asset']}] {s['signal']}: {s['action_note']}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from data_feed import get_all_data

    console.print("[bold cyan]Taoist CIO — Layer 1B: Fundamental Threshold Signals[/bold cyan]")
    data    = get_all_data()
    signals = get_fundamental_signals(data)
    print_fundamental_summary(signals)
