"""
dashboard.py — Taoist CIO terminal intelligence dashboard.

Orchestrates all five layers into a single rich terminal display.
Run with: python src/dashboard.py
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_feed import get_all_data
from tatum_indicators import analyze_all, analyze_ticker
from layer2_signals import get_layer2_signals
from mason_signals import get_fundamental_signals
from integration import get_confirmation_verdicts, check_convergence, ASSET_MAP
from venture_signals import get_venture_signals, TRACKED_COMPANIES

# ── Console — force colour even when piped ─────────────────────────────────────
console = Console(force_terminal=True, highlight=False, width=120)

# ── Shared style maps ──────────────────────────────────────────────────────────
POSTURE_STYLE = {
    "BULLISH":          "bold green",
    "NEUTRAL-BULLISH":  "green",
    "NEUTRAL":          "yellow",
    "NEUTRAL-BEARISH":  "red",
    "BEARISH":          "bold red",
    "N/A":              "dim",
}
VERDICT_STYLE = {
    "CONVERGENCE": "bold white on green",
    "CONFIRMS":    "bold green",
    "LEADS":       "bold yellow",
    "CONTRADICTS": "bold red",
    "NO SIGNAL":   "dim",
}
STATUS_STYLE = {
    "GREEN":     "bold green",
    "AMBER":     "bold yellow",
    "RED":       "bold red",
    "NEUTRAL":   "dim",
    "NO SIGNAL": "dim",
    "N/A":       "dim",
}
VS_MA_COLOR  = {"ABOVE": "green", "AT": "yellow", "BELOW": "red", "N/A": "dim"}
MACD_COLOR   = {
    "BULLISH CROSSOVER":  "bold green",
    "BULLISH":            "green",
    "NEUTRAL":            "yellow",
    "BEARISH":            "red",
    "BEARISH CROSSOVER":  "bold red",
    "N/A":                "dim",
}


# ── Section builders ───────────────────────────────────────────────────────────

def _header() -> Panel:
    now   = datetime.now(ZoneInfo("America/New_York"))
    date  = now.strftime("%A, %B %-d, %Y")
    time_ = now.strftime("%-I:%M %p ET")
    title = Text("TAOIST CIO — MARKET INTELLIGENCE DASHBOARD", style="bold white")
    sub   = Text(f"{date}  |  {time_}", style="dim")
    body  = Text.assemble(title, "\n", sub, justify="center")
    return Panel(body, style="bold cyan", padding=(0, 2))


def _macro_bar(tech: dict[str, dict]) -> Panel:
    """One-line macro context: ^TNX, DXY, VIX, Brent."""
    items = [
        ("^TNX",      "10Y Yield", "%",   lambda v: "red" if v > 5 else "yellow" if v > 4.5 else "green"),
        ("DX-Y.NYB",  "DXY",       "",    lambda v: "yellow" if v > 105 else "green"),
        ("^VIX",      "VIX",       "",    lambda v: "red" if v > 30 else "yellow" if v > 20 else "green"),
        ("BZ=F",      "Brent",     "$/b", lambda v: "white"),
    ]
    parts: list[Text] = []
    for ticker, label, suffix, colour_fn in items:
        r      = tech.get(ticker, {})
        price  = r.get("price")
        chg    = r.get("change_pct")
        if price is None:
            val_str = "N/A"
            color   = "dim"
        else:
            color   = colour_fn(price)
            val_str = f"{price:.2f}{suffix}"
        chg_str = f" ({chg:+.2f}%)" if chg is not None else ""
        seg = Text()
        seg.append(f"{label}: ", style="bold")
        seg.append(val_str, style=color)
        seg.append(chg_str, style="dim")
        parts.append(seg)

    row = Text("   |   ", style="dim").join(parts)
    return Panel(row, title="[bold]MACRO CONTEXT[/bold]", padding=(0, 2), border_style="dim")


def _layer1_table(tech_results: dict, prices: dict) -> Panel:
    """Technical posture for key markets."""
    TICKERS = [
        ("SPY",      "S&P 500"),
        ("QQQ",      "Nasdaq"),
        ("DIA",      "Dow"),
        ("BTC-USD",  "Bitcoin"),
        ("ETH-USD",  "Ethereum"),
        ("GLD",      "Gold ETF"),
        ("PSLV",     "Silver ETF"),
        ("CPER",     "Copper ETF"),
        ("URA",      "Uranium ETF"),
        ("AWK",      "Water Util."),
        ("EFA",      "Intl Equity"),
    ]

    t = Table(show_header=True, header_style="bold", show_lines=False,
              border_style="dim", box=None, pad_edge=False)
    t.add_column("Market",   style="bold", no_wrap=True, min_width=12)
    t.add_column("Price",    justify="right", min_width=10)
    t.add_column("vs 200MA", justify="center", min_width=9)
    t.add_column("RSI",      justify="center", min_width=8)
    t.add_column("MACD",     justify="center", min_width=20)
    t.add_column("Posture",  justify="center", min_width=18)

    for ticker, label in TICKERS:
        r       = tech_results.get(ticker, {})
        pr      = prices.get(ticker, {})
        price   = pr.get("price")
        chg     = pr.get("change_pct")
        posture = r.get("posture", "N/A")
        rsi_val = r.get("rsi", {}).get("value")
        vs200   = r.get("ma", {}).get("vs_200", "N/A")
        macd    = r.get("macd", {}).get("classification", "N/A")

        price_str = (
            f"{price:,.2f}" if price and price >= 1
            else f"{price:.6f}" if price else "—"
        )
        chg_color  = "green" if (chg or 0) >= 0 else "red"
        chg_str    = f" [{chg_color}]{chg:+.2f}%[/{chg_color}]" if chg is not None else ""
        rsi_color  = "green" if rsi_val and rsi_val < 40 else "yellow" if rsi_val and rsi_val < 60 else "red" if rsi_val else "dim"
        rsi_str    = f"[{rsi_color}]{rsi_val:.0f}[/{rsi_color}]" if rsi_val else "—"

        t.add_row(
            label,
            f"{price_str}{chg_str}",
            f"[{VS_MA_COLOR.get(vs200,'dim')}]{vs200}[/{VS_MA_COLOR.get(vs200,'dim')}]",
            rsi_str,
            f"[{MACD_COLOR.get(macd,'dim')}]{macd}[/{MACD_COLOR.get(macd,'dim')}]",
            f"[{POSTURE_STYLE.get(posture,'white')}]{posture}[/{POSTURE_STYLE.get(posture,'white')}]",
        )

    return Panel(t, title="[bold]LAYER 1 — TECHNICAL POSTURE  (Tatum)[/bold]",
                 border_style="cyan", padding=(0, 1))


def _layer2_table(signals: dict) -> Panel:
    t = Table(show_header=True, header_style="bold", show_lines=True,
              border_style="dim", box=None, pad_edge=False)
    t.add_column("Signal",         style="bold", no_wrap=True, min_width=26)
    t.add_column("Reading",        justify="center", min_width=18)
    t.add_column("St.",            justify="center", min_width=7)
    t.add_column("Interpretation", min_width=55)

    for sig in signals.values():
        status   = sig.get("status", "N/A")
        s_style  = STATUS_STYLE.get(status, "white")
        interp   = sig.get("interpretation", "")
        if len(interp) > 80:
            interp = interp[:77] + "…"

        # Append range note for gold/oil if present
        range_note = sig.get("range_note", "")
        reading = str(sig.get("current_reading", "—"))
        if range_note:
            reading = f"{reading}  [{range_note}]"

        t.add_row(
            sig.get("name", "—"),
            f"[{s_style}]{reading}[/{s_style}]",
            f"[{s_style}]{status}[/{s_style}]",
            interp,
        )

    return Panel(t, title="[bold]LAYER 2 — NON-TRADITIONAL SIGNALS[/bold]",
                 border_style="cyan", padding=(0, 1))


def _mason_table(signals: list[dict]) -> Panel:
    active = [s for s in signals if s.get("status") in ("GREEN", "RED")]

    if not active:
        body = Text("No active fundamental signals.", style="dim italic")
        return Panel(body, title="[bold]FUNDAMENTAL SIGNALS  (Mason)[/bold]",
                     border_style="cyan", padding=(0, 2))

    t = Table(show_header=True, header_style="bold", show_lines=True,
              border_style="dim", box=None, pad_edge=False)
    t.add_column("Asset",     style="bold", no_wrap=True, min_width=20)
    t.add_column("Signal",    no_wrap=False, min_width=26)
    t.add_column("Current",   justify="right", min_width=14)
    t.add_column("Threshold", justify="center", min_width=20)
    t.add_column("St.",       justify="center", min_width=7)
    t.add_column("Action",    min_width=30)

    for s in active:
        status  = s.get("status", "NO SIGNAL")
        s_style = STATUS_STYLE.get(status, "white")
        note    = s.get("action_note", "")
        if len(note) > 60:
            note = note[:57] + "…"

        t.add_row(
            s.get("asset", "—"),
            s.get("signal", "—"),
            s.get("current_value", "—"),
            s.get("threshold_value", "—"),
            f"[{s_style}]{status}[/{s_style}]",
            note,
        )

    return Panel(t, title="[bold]FUNDAMENTAL SIGNALS  (Mason)[/bold]",
                 border_style="cyan", padding=(0, 1))


def _verdicts_table(verdicts: list[dict]) -> Panel:
    t = Table(show_header=True, header_style="bold", show_lines=False,
              border_style="dim", box=None, pad_edge=False)
    t.add_column("Asset",       style="bold", no_wrap=True, min_width=10)
    t.add_column("Tkr",         style="dim", no_wrap=True, min_width=8)
    t.add_column("Fundamental", justify="center", min_width=12)
    t.add_column("Technical",   justify="center", min_width=18)
    t.add_column("RSI",         justify="center", min_width=5)
    t.add_column("MACD",        justify="center", min_width=20)
    t.add_column("Verdict",     justify="center", min_width=14)

    for v in verdicts:
        fund_s  = v.get("fundamental_status", "N/A")
        posture = v.get("technical_posture", "N/A")
        verdict = v.get("verdict", "NO SIGNAL")
        rsi_val = v.get("rsi")
        macd    = v.get("macd", "N/A")

        rsi_color  = "green" if rsi_val and rsi_val < 50 else "yellow" if rsi_val else "dim"
        rsi_str    = f"[{rsi_color}]{rsi_val:.0f}[/{rsi_color}]" if rsi_val else "—"

        t.add_row(
            v.get("asset", "—"),
            v.get("tech_ticker", "—"),
            f"[{STATUS_STYLE.get(fund_s,'dim')}]{fund_s}[/{STATUS_STYLE.get(fund_s,'dim')}]",
            f"[{POSTURE_STYLE.get(posture,'white')}]{posture}[/{POSTURE_STYLE.get(posture,'white')}]",
            rsi_str,
            f"[{MACD_COLOR.get(macd,'dim')}]{macd}[/{MACD_COLOR.get(macd,'dim')}]",
            f"[{VERDICT_STYLE.get(verdict,'dim')}]{verdict}[/{VERDICT_STYLE.get(verdict,'dim')}]",
        )

    return Panel(t, title="[bold]CONFIRMATION VERDICTS[/bold]",
                 border_style="cyan", padding=(0, 1))


def _convergence_panel(verdicts: list[dict]) -> Panel:
    hits = check_convergence(verdicts)

    if not hits:
        body = Text(
            "No convergence signals active.\n"
            "Convergence = Fundamental GREEN  +  Technical BULLISH  +  RSI < 50\n"
            "System is monitoring all mapped assets for alignment.",
            style="dim",
        )
        return Panel(body,
                     title="[dim]  ◯  CONVERGENCE MONITOR  [/dim]",
                     border_style="dim", padding=(0, 2))

    lines = []
    for v in hits:
        lines.append(
            f"[bold white]★  {v['asset']}[/bold white]  ({v['tech_ticker']})\n"
            f"   Fundamental: [bold green]{v['fundamental_status']}[/bold green]   "
            f"Technical: [bold green]{v['technical_posture']}[/bold green]   "
            f"RSI: {v['rsi']:.1f}\n"
            f"   Signals: {'; '.join(v['fired_signals'])}\n"
            f"   [italic]{v['rationale']}[/italic]"
        )

    body = "\n\n".join(lines)
    return Panel(
        body,
        title="[bold white on green]  ⚡ CONVERGENCE DETECTED — HIGHEST CONVICTION  ⚡  [/bold white on green]",
        border_style="bold green",
        padding=(1, 3),
    )


def _ipo_tracker_panel(venture: dict) -> Panel:
    """Section 6 — IPO Tracker (Venture signals)."""
    forge    = venture.get("spacex_forge", {})
    ant      = venture.get("anthropic", {})
    edgar    = venture.get("edgar", {})
    ji       = venture.get("ji_moment", False)
    wu_list  = venture.get("wu_wei_entry", [])

    # ── Ji Moment banner ───────────────────────────────────────────────────────
    if ji:
        for filing in edgar.get("new_filings", []):
            company = filing.get("tracked_as", "Unknown")
            console.print(Panel(
                f"[bold white]S-1 FILED: {company.upper()}\n"
                f"Accession: {filing.get('accession_no')}  |  "
                f"Filed: {filing.get('file_date')}\n"
                f"URL: {filing.get('filing_url')}[/bold white]",
                title="[bold white on red]  ⚡  JI MOMENT — S-1 FILED — ACT NOW  ⚡  [/bold white on red]",
                border_style="bold red", padding=(0, 2),
            ))

    # ── Wu Wei banner ──────────────────────────────────────────────────────────
    if wu_list:
        console.print(Panel(
            "[bold white]" + "\n".join(f"★  {c}" for c in wu_list) + "[/bold white]\n"
            "[dim]Price >20% below IPO open — Wu Wei entry conditions met.[/dim]",
            title="[bold white on green]  ★  WU WEI ENTRY SIGNAL  ★  [/bold white on green]",
            border_style="bold green", padding=(0, 2),
        ))

    # ── Pipeline table ─────────────────────────────────────────────────────────
    company_filings = {
        f.get("tracked_as"): f
        for f in reversed(edgar.get("all_found", []))
    }

    pipeline = Table(show_header=True, header_style="bold", show_lines=False,
                     border_style="dim", box=None, pad_edge=False)
    pipeline.add_column("Company",   style="bold", no_wrap=True, min_width=12)
    pipeline.add_column("S-1",       justify="center",           min_width=18)
    pipeline.add_column("Filed",     justify="center",           min_width=12)
    pipeline.add_column("Accession", style="dim",                min_width=24)

    for display_name, _ in TRACKED_COMPANIES:
        filing = company_filings.get(display_name)
        if filing:
            is_new = any(f.get("accession_no") == filing.get("accession_no")
                         for f in edgar.get("new_filings", []))
            s1_str = "[bold red]⚡ NEW — JI MOMENT[/bold red]" if is_new else "[yellow]FILED[/yellow]"
            pipeline.add_row(display_name, s1_str,
                             filing.get("file_date", "—"),
                             filing.get("accession_no", "—"))
        else:
            pipeline.add_row(display_name, "[dim]No S-1 detected[/dim]", "—", "—")

    # ── SpaceX Forge ───────────────────────────────────────────────────────────
    forge_price = forge.get("current_price")
    forge_str   = f"${forge_price:,.2f}" if forge_price else "N/A"
    stale_tag   = " [bold yellow]⚠ STALE[/bold yellow]" if forge.get("stale") else ""
    forge_line  = f"Forge price: [bold]{forge_str}[/bold]{stale_tag}  [dim](updated {forge.get('last_updated','?')})[/dim]"

    # ── Anthropic ─────────────────────────────────────────────────────────────
    hiive  = ant.get("last_known_hiive_price")
    arr_b  = (ant.get("last_known_arr") or 0) / 1e9
    stale_ant = " [bold yellow]⚠ STALE[/bold yellow]" if ant.get("needs_update") else ""
    ant_line  = (
        f"Hiive: [bold]${hiive:,.2f}[/bold]  ARR: [bold]${arr_b:.0f}B[/bold]"
        f"  IPO: {ant.get('ipo_target','?')}"
        f"  [dim]verified {ant.get('last_verified','?')}[/dim]{stale_ant}"
        if hiive else "Anthropic data unavailable"
    )

    # ── EDGAR status ───────────────────────────────────────────────────────────
    edgar_line = (
        f"EDGAR last check: [dim]{edgar.get('last_check','?')}[/dim]  "
        f"Lookback: {edgar.get('lookback_days','?')}d  "
        f"New filings: [{'bold red' if ji else 'dim'}]{len(edgar.get('new_filings',[]))}[/{'bold red' if ji else 'dim'}]"
    )

    body = Text.assemble(
        Text("SpaceX    — ", style="bold"), Text.from_markup(forge_line), "\n",
        Text("Anthropic — ", style="bold"), Text.from_markup(ant_line),   "\n",
        Text("EDGAR     — ", style="bold"), Text.from_markup(edgar_line), "\n\n",
    )

    from rich.console import Group
    return Panel(
        Group(body, pipeline),
        title="[bold cyan]SECTION 6 — IPO TRACKER  (Venture)[/bold cyan]",
        border_style="cyan", padding=(0, 1),
    )


def _outlook_panel() -> Panel:
    body = Text(
        "Weekly outlook: [To be written by Gregory]\n\n"
        "This field is reserved for the advisor's qualitative macro view, "
        "thesis updates, and forward positioning notes.",
        style="dim italic",
    )
    return Panel(body, title="[bold]LAYER 5 — ADVISOR OUTLOOK[/bold]",
                 border_style="dim", padding=(1, 2))


def _footer(timestamp: str) -> Text:
    return Text(
        f" Last updated: {timestamp}  |  Data source: yfinance  |  "
        "⚠  VERIFY ALL READINGS BEFORE ACTING ",
        style="dim",
        justify="center",
    )


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run_dashboard() -> None:
    # ── Data loading ───────────────────────────────────────────────────────────
    console.print(
        Panel("[dim]Loading market data…[/dim]", border_style="dim", padding=(0, 2))
    )

    data        = get_all_data()
    tech        = analyze_all(data)
    l2_signals  = get_layer2_signals(data)
    fund_sigs   = get_fundamental_signals(data)
    verdicts    = get_confirmation_verdicts(tech, fund_sigs, data)
    venture     = get_venture_signals(data)

    # Flatten technical results to include price + change for macro bar
    from data_feed import get_current_prices
    prices = get_current_prices(data=data)

    # Add price/change into tech results for easy access in macro bar
    for ticker, info in prices.items():
        if ticker in tech:
            tech[ticker]["price"]      = info.get("price")
            tech[ticker]["change_pct"] = info.get("change_pct")
        else:
            tech[ticker] = {"price": info.get("price"), "change_pct": info.get("change_pct")}

    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d  %-I:%M:%S %p ET")

    # ── Render ─────────────────────────────────────────────────────────────────
    console.print()
    console.print(_header())
    console.print()
    console.print(_macro_bar(tech))
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(_layer1_table(tech, prices))
    console.print()
    console.print(_layer2_table(l2_signals))
    console.print()
    console.print(_mason_table(fund_sigs))
    console.print()
    console.print(Rule(style="dim"))
    console.print()
    console.print(_verdicts_table(verdicts))
    console.print()
    console.print(_convergence_panel(verdicts))
    console.print()
    console.print(_ipo_tracker_panel(venture))
    console.print()
    console.print(_outlook_panel())
    console.print()
    console.print(Rule(style="dim"))
    console.print(_footer(timestamp))
    console.print()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_dashboard()
