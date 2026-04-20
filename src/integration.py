"""
integration.py — Confirmation Verdict layer for the Taoist CIO dashboard.

Combines tatum_indicators (technical posture) and mason_signals (fundamental
threshold signals) to produce Confirmation Verdicts for each mapped asset.

Verdict logic (applied only where a fundamental GREEN signal exists):
  CONFIRMS    — fundamental GREEN + technical BULLISH or NEUTRAL-BULLISH
  LEADS       — fundamental GREEN + technical NEUTRAL
  CONTRADICTS — fundamental GREEN + technical NEUTRAL-BEARISH or BEARISH
  NO SIGNAL   — no GREEN fundamental (threshold not crossed, RED warning, or
                manual input required)

CONVERGENCE (highest-conviction) — fundamental GREEN + technical BULLISH +
  RSI < 50. Flagged prominently in a Rich Panel below the main table.

Asset → ticker mapping:
  Gold      : fundamentals from GC=F thresholds → technicals from GLD
  Silver    : fundamentals from gold/silver ratio → technicals from PSLV
  Copper    : fundamentals from HG=F thresholds  → technicals from CPER
  Uranium   : fundamentals from SPUT NAV         → technicals from URA
  Water     : fundamentals from AWK thresholds   → technicals from AWK
  S&P 500   : fundamentals from SPY 200MA signal → technicals from SPY
  Bitcoin   : fundamentals from BTC 200MA signal → technicals from BTC-USD
"""

import logging
import sys
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths / logging ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
logging.basicConfig(
    filename=ROOT / "logs" / "integration.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)
console = Console()

# ── Asset map ──────────────────────────────────────────────────────────────────
# asset_label → (tech_ticker, set of asset names used in mason_signals output)
ASSET_MAP: dict[str, tuple[str, set[str]]] = {
    "Gold":    ("GLD",     {"Gold"}),
    "Silver":  ("PSLV",    {"Silver"}),
    "Copper":  ("CPER",    {"Copper"}),
    "Uranium": ("URA",     {"Uranium"}),
    "Water":   ("AWK",     {"Water (AWK)"}),
    "S&P 500": ("SPY",     {"S&P 500 (SPY)"}),
    "Bitcoin": ("BTC-USD", {"BTC"}),
}

# ── Technical posture ordering ─────────────────────────────────────────────────
_POSTURE_RANK = {
    "BULLISH":          5,
    "NEUTRAL-BULLISH":  4,
    "NEUTRAL":          3,
    "NEUTRAL-BEARISH":  2,
    "BEARISH":          1,
    "N/A":              0,
}

# ── Verdict styling ────────────────────────────────────────────────────────────
VERDICT_STYLE = {
    "CONFIRMS":    "bold green",
    "LEADS":       "bold yellow",
    "CONTRADICTS": "bold red",
    "NO SIGNAL":   "dim",
    "CONVERGENCE": "bold white on green",
}

POSTURE_STYLE = {
    "BULLISH":          "bold green",
    "NEUTRAL-BULLISH":  "green",
    "NEUTRAL":          "yellow",
    "NEUTRAL-BEARISH":  "red",
    "BEARISH":          "bold red",
    "N/A":              "dim",
}

FUND_STYLE = {
    "GREEN":     "bold green",
    "AMBER":     "bold yellow",
    "RED":       "bold red",
    "NO SIGNAL": "dim",
    "N/A":       "dim",
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def _best_fundamental_status(
    fundamental_signals: list[dict],
    asset_names: set[str],
) -> tuple[str, list[str]]:
    """
    From a list of mason_signals dicts, return the best status for the given
    asset names and the list of signal descriptions that fired.

    Priority: GREEN > AMBER > RED > NO SIGNAL.
    Only GREEN fundamentals can drive a CONFIRMS / LEADS / CONTRADICTS verdict.
    """
    priority = {"GREEN": 4, "AMBER": 3, "RED": 2, "NO SIGNAL": 1}
    best_status = "NO SIGNAL"
    fired: list[str] = []

    for sig in fundamental_signals:
        if sig.get("asset") not in asset_names:
            continue
        status = sig.get("status", "NO SIGNAL")
        if priority.get(status, 0) > priority.get(best_status, 0):
            best_status = status
        if status == "GREEN":
            fired.append(sig.get("signal", ""))

    return best_status, fired


def _determine_verdict(
    fund_status: str,
    posture: str,
    rsi: float | None,
) -> tuple[str, str]:
    """
    Return (verdict, rationale) based on fundamental status, technical
    posture, and RSI.
    """
    rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"

    if fund_status != "GREEN":
        if fund_status == "RED":
            rationale = (
                "Bear warning signal active — fundamental RED overrides "
                "entry logic. Monitor for reversal before re-entry."
            )
        elif fund_status == "AMBER":
            rationale = "Threshold at caution level — no entry signal yet."
        else:
            rationale = "No fundamental threshold crossed for this asset."
        return "NO SIGNAL", rationale

    rank = _POSTURE_RANK.get(posture, 0)

    # CONVERGENCE: GREEN + BULLISH + RSI < 50
    if rank >= 5 and rsi is not None and rsi < 50:
        rationale = (
            f"HIGHEST CONVICTION — fundamental entry signal confirmed by "
            f"BULLISH technical posture with RSI {rsi:.1f} (below 50, not extended)."
        )
        return "CONVERGENCE", rationale

    # CONFIRMS: GREEN + BULLISH or NEUTRAL-BULLISH
    if rank >= 4:
        rationale = (
            f"Fundamental entry signal confirmed by {posture} technical posture. "
            f"RSI {rsi_str}. Thesis aligned with price action — deploy capital."
        )
        return "CONFIRMS", rationale

    # LEADS: GREEN + NEUTRAL
    if rank == 3:
        rationale = (
            f"Fundamental entry signal present but technicals are NEUTRAL. "
            f"RSI {rsi_str}. Thesis is ahead of price — "
            "small starter position; wait for technical improvement."
        )
        return "LEADS", rationale

    # CONTRADICTS: GREEN + NEUTRAL-BEARISH or BEARISH
    rationale = (
        f"Fundamental entry signal present but technicals are {posture}. "
        f"RSI {rsi_str}. Price action contradicts thesis — "
        "hold off; wait for technical confirmation before committing."
    )
    return "CONTRADICTS", rationale


# ── Public API ─────────────────────────────────────────────────────────────────

def get_confirmation_verdicts(
    technical_results: dict[str, dict],
    fundamental_signals: list[dict],
    data_dict: dict[str, pd.DataFrame],
) -> list[dict]:
    """
    Produce a Confirmation Verdict for each mapped asset.

    Args:
        technical_results:   output of tatum_indicators.analyze_all()
        fundamental_signals: output of mason_signals.get_fundamental_signals()
        data_dict:           raw {ticker: DataFrame} from data_feed.get_all_data()
                             (reserved for future use / additional lookups)

    Returns:
        List of verdict dicts, one per asset.
    """
    verdicts: list[dict] = []

    for asset_label, (tech_ticker, fund_asset_names) in ASSET_MAP.items():
        try:
            # ── Fundamental side ───────────────────────────────────────────────
            fund_status, fired_signals = _best_fundamental_status(
                fundamental_signals, fund_asset_names
            )

            # ── Technical side ─────────────────────────────────────────────────
            tech = technical_results.get(tech_ticker, {})
            posture = tech.get("posture", "N/A")
            rsi_val = tech.get("rsi", {}).get("value")
            ma_align = tech.get("ma", {}).get("alignment", "N/A")
            macd_class = tech.get("macd", {}).get("classification", "N/A")
            vs_200 = tech.get("ma", {}).get("vs_200", "N/A")

            # ── Verdict ────────────────────────────────────────────────────────
            verdict, rationale = _determine_verdict(fund_status, posture, rsi_val)

            verdicts.append({
                "asset":              asset_label,
                "tech_ticker":        tech_ticker,
                "fundamental_status": fund_status,
                "fired_signals":      fired_signals,
                "technical_posture":  posture,
                "rsi":                rsi_val,
                "ma_alignment":       ma_align,
                "macd":               macd_class,
                "vs_200ma":           vs_200,
                "verdict":            verdict,
                "rationale":          rationale,
            })

        except Exception as e:
            log.error(f"Verdict failed for {asset_label}: {e}")
            verdicts.append({
                "asset":              asset_label,
                "tech_ticker":        ASSET_MAP[asset_label][0],
                "fundamental_status": "N/A",
                "fired_signals":      [],
                "technical_posture":  "N/A",
                "rsi":                None,
                "ma_alignment":       "N/A",
                "macd":               "N/A",
                "vs_200ma":           "N/A",
                "verdict":            "NO SIGNAL",
                "rationale":          f"Error: {e}",
            })

    return verdicts


def check_convergence(verdicts: list[dict]) -> list[dict]:
    """Return only those verdicts that are in CONVERGENCE state."""
    return [v for v in verdicts if v.get("verdict") == "CONVERGENCE"]


# ── Rich display ───────────────────────────────────────────────────────────────

def print_integration_summary(verdicts: list[dict]) -> None:
    """
    Print the Confirmation Verdict table and, if any exist, a prominent
    CONVERGENCE panel below it.
    """
    # ── Main table ─────────────────────────────────────────────────────────────
    table = Table(
        title="Taoist CIO — Integration: Confirmation Verdicts",
        show_lines=True,
        header_style="bold cyan",
        border_style="dim",
        min_width=100,
    )
    table.add_column("Asset",       style="bold",   no_wrap=True,  min_width=10)
    table.add_column("Ticker",      style="dim",    no_wrap=True,  min_width=8)
    table.add_column("Fundamental", justify="center",              min_width=12)
    table.add_column("Technical",   justify="center",              min_width=18)
    table.add_column("RSI",         justify="center",              min_width=8)
    table.add_column("vs 200MA",    justify="center",              min_width=8)
    table.add_column("MACD",        justify="center",              min_width=20)
    table.add_column("Verdict",     justify="center",              min_width=14)

    convergence_list = check_convergence(verdicts)

    for v in verdicts:
        fund_s   = v["fundamental_status"]
        posture  = v["technical_posture"]
        verdict  = v["verdict"]
        rsi_val  = v["rsi"]
        vs200    = v["vs_200ma"]
        macd     = v["macd"]

        fund_style    = FUND_STYLE.get(fund_s, "white")
        posture_style = POSTURE_STYLE.get(posture, "white")
        verdict_style = VERDICT_STYLE.get(verdict, "white")

        vs200_color = {"ABOVE": "green", "AT": "yellow", "BELOW": "red"}.get(vs200, "dim")
        macd_color  = {
            "BULLISH CROSSOVER":  "bold green",
            "BULLISH":            "green",
            "NEUTRAL":            "yellow",
            "BEARISH":            "red",
            "BEARISH CROSSOVER":  "bold red",
        }.get(macd, "dim")

        rsi_str = f"{rsi_val:.0f}" if rsi_val is not None else "—"
        # Colour RSI relative to 50 (below = not extended, green; above = yellow/red)
        if rsi_val is not None:
            rsi_color = "green" if rsi_val < 50 else "yellow" if rsi_val < 60 else "red"
            rsi_str = f"[{rsi_color}]{rsi_str}[/{rsi_color}]"

        table.add_row(
            v["asset"],
            v["tech_ticker"],
            f"[{fund_style}]{fund_s}[/{fund_style}]",
            f"[{posture_style}]{posture}[/{posture_style}]",
            rsi_str,
            f"[{vs200_color}]{vs200}[/{vs200_color}]",
            f"[{macd_color}]{macd}[/{macd_color}]",
            f"[{verdict_style}]{verdict}[/{verdict_style}]",
        )

    console.print()
    console.print(table)

    # ── Verdict summary counts ─────────────────────────────────────────────────
    from collections import Counter
    counts = Counter(v["verdict"] for v in verdicts)
    console.print(
        f"\n[bold]Verdicts:[/bold]  "
        + "  ".join(
            f"[{VERDICT_STYLE.get(k, 'white')}]{v} {k}[/{VERDICT_STYLE.get(k, 'white')}]"
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        )
    )

    # ── Rationale for actionable verdicts ──────────────────────────────────────
    actionable = [v for v in verdicts if v["verdict"] in ("CONFIRMS", "LEADS", "CONTRADICTS", "CONVERGENCE")]
    if actionable:
        console.print("\n[bold]Actionable verdicts — detail:[/bold]")
        for v in actionable:
            fired_str = (
                f"  [dim]Signals:[/dim] {'; '.join(v['fired_signals'])}" if v["fired_signals"] else ""
            )
            style = VERDICT_STYLE.get(v["verdict"], "white")
            console.print(
                f"  [{style}]{v['verdict']}[/{style}]  "
                f"[bold]{v['asset']}[/bold] ({v['tech_ticker']})"
            )
            if fired_str:
                console.print(fired_str)
            console.print(f"  [dim]{v['rationale']}[/dim]")

    # ── CONVERGENCE panel ──────────────────────────────────────────────────────
    console.print()
    if convergence_list:
        lines = []
        for v in convergence_list:
            lines.append(
                f"[bold white]{v['asset']}[/bold white] ({v['tech_ticker']})\n"
                f"  Fundamental : {v['fundamental_status']}  |  "
                f"Technical : {v['technical_posture']}  |  "
                f"RSI : {v['rsi']:.1f}\n"
                f"  Fired       : {'; '.join(v['fired_signals'])}\n"
                f"  Rationale   : {v['rationale']}"
            )
        panel_body = "\n\n".join(lines)
        console.print(
            Panel(
                panel_body,
                title="[bold white on green]  ★  CONVERGENCE — HIGHEST CONVICTION SIGNAL  ★  [/bold white on green]",
                border_style="bold green",
                padding=(1, 3),
            )
        )
    else:
        console.print(
            Panel(
                "[dim]No CONVERGENCE signals active today.\n\n"
                "CONVERGENCE requires: Fundamental GREEN  +  Technical BULLISH  +  RSI < 50.\n"
                "Current environment: entry signals present but technical posture has not "
                "yet reached full BULLISH alignment.\n"
                "Watch for posture upgrades — particularly in [bold]Gold (GLD)[/bold] "
                "if RSI remains sub-50 and MA alignment tightens.[/dim]",
                title="[bold dim]  ◯  CONVERGENCE MONITOR  [/bold dim]",
                border_style="dim",
                padding=(1, 3),
            )
        )


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    from data_feed import get_all_data
    from tatum_indicators import analyze_all
    from mason_signals import get_fundamental_signals

    console.print("[bold cyan]Taoist CIO — Integration Layer: Confirmation Verdicts[/bold cyan]")

    data       = get_all_data()
    tech       = analyze_all(data)
    fund_sigs  = get_fundamental_signals(data)
    verdicts   = get_confirmation_verdicts(tech, fund_sigs, data)

    print_integration_summary(verdicts)
