"""
dashboard_web.py — Streamlit web dashboard for the Taoist CIO system.

Web-based equivalent of dashboard.py. Run with:
    streamlit run src/dashboard_web.py
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_feed import get_all_data, get_current_prices, TICKERS as TICKER_MAP
from tatum_indicators import analyze_all
from layer2_signals import get_layer2_signals
from mason_signals import get_fundamental_signals
from integration import get_confirmation_verdicts, check_convergence
from venture_signals import get_venture_signals, TRACKED_COMPANIES

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Taoist CIO",
    page_icon="☯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Style constants ────────────────────────────────────────────────────────────

POSTURE_COLOR = {
    "BULLISH":          "#00c853",
    "NEUTRAL-BULLISH":  "#69f0ae",
    "NEUTRAL":          "#9e9e9e",
    "NEUTRAL-BEARISH":  "#ff6d00",
    "BEARISH":          "#d50000",
    "N/A":              "#616161",
}

STATUS_COLOR = {
    "GREEN":     "#00c853",
    "AMBER":     "#ffd600",
    "RED":       "#d50000",
    "NEUTRAL":   "#9e9e9e",
    "NO SIGNAL": "#616161",
    "N/A":       "#616161",
}

VERDICT_COLOR = {
    "CONVERGENCE": "#00e676",
    "CONFIRMS":    "#69f0ae",
    "LEADS":       "#ffd600",
    "CONTRADICTS": "#ff5252",
    "NO SIGNAL":   "#9e9e9e",
}

OUTLOOK_FILE = ROOT / "data" / "layer5_outlook.txt"

# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(ttl=900)
def load_all_data():
    data      = get_all_data()
    prices    = get_current_prices(data=data)
    tech      = analyze_all(data)
    # Merge price info into tech for convenience
    for ticker, info in prices.items():
        if ticker in tech:
            tech[ticker]["price"]      = info.get("price")
            tech[ticker]["change_pct"] = info.get("change_pct")
        else:
            tech[ticker] = {"price": info.get("price"), "change_pct": info.get("change_pct")}
    l2       = get_layer2_signals(data)
    fund     = get_fundamental_signals(data)
    verdicts = get_confirmation_verdicts(tech, fund, data)
    venture  = get_venture_signals(data)
    return data, prices, tech, l2, fund, verdicts, venture


# ── Colour helpers ─────────────────────────────────────────────────────────────

def _colored_badge(text: str, color: str) -> str:
    """Return an HTML badge span."""
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em">{text}</span>'
    )


def _posture_badge(posture: str) -> str:
    return _colored_badge(posture, POSTURE_COLOR.get(posture, "#9e9e9e"))


def _status_badge(status: str) -> str:
    return _colored_badge(status, STATUS_COLOR.get(status, "#9e9e9e"))


def _verdict_badge(verdict: str) -> str:
    return _colored_badge(verdict, VERDICT_COLOR.get(verdict, "#9e9e9e"))


# ── Section renderers ──────────────────────────────────────────────────────────

def render_header(timestamp: str):
    st.markdown(
        """
        <div style="text-align:center;padding:16px 0 4px">
            <h1 style="margin:0;letter-spacing:2px">TAOIST CIO — MARKET INTELLIGENCE DASHBOARD</h1>
            <p style="margin:4px 0 0;color:#9e9e9e;font-size:1.05em">
                Conscious Capital Wealth Management
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_ts, col_btn = st.columns([5, 1])
    col_ts.caption(f"Last updated: {timestamp}  |  Cache TTL: 15 min")
    if col_btn.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    st.divider()


def render_sidebar(tech: dict, l2: dict, fund: list, verdicts: list):
    with st.sidebar:
        st.markdown("## ☯ Taoist CIO")
        st.markdown("**Navigation**")
        section = st.radio(
            "Go to section",
            [
                "1 — Macro Context",
                "2 — Technical Posture",
                "3 — Price Charts",
                "4 — Layer 2 Signals",
                "5 — Fundamental & Verdicts",
                "6 — IPO Tracker",
                "7 — Outlook",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("**Signal Summary**")
        green_count = sum(1 for s in fund if s.get("status") == "GREEN")
        amber_count = sum(1 for s in fund if s.get("status") == "AMBER")
        red_count   = sum(1 for s in fund if s.get("status") == "RED")
        l2_green = sum(1 for s in l2.values() if s.get("status") == "GREEN")
        l2_red   = sum(1 for s in l2.values() if s.get("status") == "RED")

        col1, col2, col3 = st.columns(3)
        col1.metric("GREEN", green_count + l2_green, delta=None)
        col2.metric("AMBER", amber_count, delta=None)
        col3.metric("RED", red_count + l2_red, delta=None)

        st.divider()
        st.markdown("**Gold RSI**")
        gld_rsi = tech.get("GLD", {}).get("rsi", {}).get("value")
        if gld_rsi is not None:
            st.metric(
                "GLD RSI (14)",
                f"{gld_rsi:.1f}",
                delta="↑ One tick from CONVERGENCE" if gld_rsi < 52 else None,
                delta_color="normal",
            )
            if gld_rsi < 52:
                st.caption("🟡 RSI approaching sub-50 — watch for CONVERGENCE signal")
        else:
            st.metric("GLD RSI (14)", "N/A")

        st.divider()
        conv_hits = check_convergence(verdicts)
        if conv_hits:
            st.success(f"⚡ CONVERGENCE: {', '.join(v['asset'] for v in conv_hits)}")
        else:
            st.info("◯ No convergence active")

    return section


def render_macro(tech: dict, data: dict):
    st.markdown("## 1 — Macro Context")
    items = [
        ("^TNX",     "10Y Yield",  "%",   lambda v: "GREEN" if v < 4.5 else "AMBER" if v < 5 else "RED"),
        ("DX-Y.NYB", "DXY",        "",    lambda v: "GREEN" if v < 100 else "AMBER" if v < 105 else "RED"),
        ("^VIX",     "VIX",        "",    lambda v: "GREEN" if v < 20 else "AMBER" if v < 30 else "RED"),
        ("BZ=F",     "Brent Crude","$/b", lambda v: "AMBER"),
    ]
    cols = st.columns(4)
    for col, (ticker, label, suffix, status_fn) in zip(cols, items):
        r       = tech.get(ticker, {})
        price   = r.get("price")
        chg     = r.get("change_pct")
        status  = status_fn(price) if price is not None else "N/A"
        color   = STATUS_COLOR.get(status, "#9e9e9e")

        with col:
            if price is not None:
                st.metric(
                    label=f"{label}",
                    value=f"{price:.2f}{suffix}",
                    delta=f"{chg:+.2f}%" if chg is not None else None,
                    delta_color="normal",
                )
            else:
                st.metric(label=label, value="N/A")
            st.markdown(
                _colored_badge(status, color),
                unsafe_allow_html=True,
            )
            # Mini 30-day sparkline
            df = data.get(ticker)
            if df is not None and len(df) >= 5:
                mini = df["Close"].tail(30)
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(len(mini))),
                    y=mini.values,
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    showlegend=False,
                ))
                fig.update_layout(
                    height=60,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_technical(tech: dict, prices: dict):
    st.markdown("## 2 — Technical Posture (Tatum)")
    LAYER1_TICKERS = [
        ("SPY",     "S&P 500"),
        ("QQQ",     "Nasdaq"),
        ("DIA",     "Dow"),
        ("BTC-USD", "Bitcoin"),
        ("ETH-USD", "Ethereum"),
        ("GLD",     "Gold ETF"),
        ("PSLV",    "Silver ETF"),
        ("CPER",    "Copper ETF"),
        ("URA",     "Uranium ETF"),
        ("AWK",     "Water Util."),
        ("EFA",     "Intl Equity"),
    ]

    rows = []
    for ticker, label in LAYER1_TICKERS:
        r       = tech.get(ticker, {})
        pr      = prices.get(ticker, {})
        price   = pr.get("price")
        chg     = pr.get("change_pct")
        posture = r.get("posture", "N/A")
        rsi_val = r.get("rsi", {}).get("value")
        vs200   = r.get("ma", {}).get("vs_200", "N/A")
        macd    = r.get("macd", {}).get("classification", "N/A")

        price_str = f"{price:,.2f}" if price and price >= 1 else f"{price:.6f}" if price else "—"
        chg_str   = f"{chg:+.2f}%" if chg is not None else "—"
        rsi_str   = f"{rsi_val:.0f}" if rsi_val else "—"

        rows.append({
            "Market":  label,
            "Ticker":  ticker,
            "Price":   price_str,
            "Chg %":   chg_str,
            "vs 200MA": vs200,
            "RSI":     rsi_str,
            "MACD":    macd,
            "_posture": posture,
            "_chg":    chg or 0,
            "_rsi":    rsi_val or 0,
            "_vs200":  vs200,
        })

    df_display = pd.DataFrame(rows)

    def style_posture(val):
        color = POSTURE_COLOR.get(val, "#9e9e9e")
        return f"background-color:{color}20;color:{color};font-weight:bold"

    def style_vs200(val):
        c = {"ABOVE": "#00c853", "AT": "#ffd600", "BELOW": "#d50000"}.get(val, "#9e9e9e")
        return f"color:{c};font-weight:bold"

    def style_chg(val):
        if val == "—":
            return "color:#9e9e9e"
        try:
            return "color:#00c853" if float(val.replace("%","").replace("+","")) >= 0 else "color:#d50000"
        except Exception:
            return ""

    styled = (
        df_display[["Market","Ticker","Price","Chg %","vs 200MA","RSI","MACD","_posture"]]
        .rename(columns={"_posture": "Posture"})
        .style
        .map(style_posture, subset=["Posture"])
        .map(style_vs200,  subset=["vs 200MA"])
        .map(style_chg,    subset=["Chg %"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_charts(data: dict, tech: dict):
    st.markdown("## 3 — Price Charts")
    all_tickers = sorted([t for tickers in TICKER_MAP.values() for t in tickers])
    default_idx = all_tickers.index("GLD") if "GLD" in all_tickers else 0
    ticker = st.selectbox("Select ticker", all_tickers, index=default_idx)

    df = data.get(ticker)
    if df is None or df.empty:
        st.warning(f"No data available for {ticker}")
        return

    df = df.tail(90).copy()
    closes = df["Close"]

    # Moving averages
    ma20  = closes.rolling(20).mean()
    ma50  = closes.rolling(50).mean()
    ma200 = closes.rolling(200).mean()

    # RSI
    delta  = closes.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss.replace(0, float("nan"))
    rsi    = 100 - (100 / (1 + rs))

    # ── Main price chart ───────────────────────────────────────────────────────
    has_ohlc = all(c in df.columns for c in ["Open", "High", "Low", "Close"])
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.03,
    )

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=ticker,
            increasing_line_color="#00c853",
            decreasing_line_color="#d50000",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=closes,
            mode="lines", name=ticker,
            line=dict(color="#69f0ae", width=1.5),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=ma20, mode="lines", name="MA20",
        line=dict(color="#2196f3", width=1), opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ma50, mode="lines", name="MA50",
        line=dict(color="#ff9800", width=1), opacity=0.8,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=ma200, mode="lines", name="MA200",
        line=dict(color="#f44336", width=1), opacity=0.8,
    ), row=1, col=1)

    # Volume bars
    if "Volume" in df.columns and df["Volume"].sum() > 0:
        vol_colors = [
            "#00c853" if c >= o else "#d50000"
            for c, o in zip(df["Close"], df["Open"] if "Open" in df.columns else df["Close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.6,
            showlegend=False,
        ), row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig.add_annotation(
            text="Volume data unavailable",
            xref="paper", yref="paper",
            x=0.5, y=0.1, showarrow=False,
            font=dict(color="#9e9e9e"),
            row=2, col=1,
        )

    fig.update_layout(
        title=f"{ticker} — 90 Day",
        height=520,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_xaxes(gridcolor="#1e2533", showgrid=True)
    fig.update_yaxes(gridcolor="#1e2533", showgrid=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── RSI chart ──────────────────────────────────────────────────────────────
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(
        x=df.index, y=rsi, mode="lines", name="RSI(14)",
        line=dict(color="#ce93d8", width=1.5),
    ))
    fig_rsi.add_hline(y=70, line=dict(color="#d50000", dash="dash", width=1), annotation_text="70")
    fig_rsi.add_hline(y=30, line=dict(color="#00c853", dash="dash", width=1), annotation_text="30")
    fig_rsi.add_hline(y=50, line=dict(color="#9e9e9e", dash="dot",  width=1), annotation_text="50")
    fig_rsi.update_layout(
        title=f"{ticker} — RSI (14)",
        height=200,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        margin=dict(l=0, r=0, t=36, b=0),
        yaxis=dict(range=[0, 100], gridcolor="#1e2533"),
        xaxis=dict(gridcolor="#1e2533"),
        showlegend=False,
    )
    st.plotly_chart(fig_rsi, use_container_width=True)


def render_layer2(l2: dict):
    st.markdown("## 4 — Layer 2 Non-Traditional Signals")
    rows = []
    for sig in l2.values():
        status  = sig.get("status", "N/A")
        reading = str(sig.get("current_reading", "—"))
        if sig.get("range_note"):
            reading += f"  [{sig['range_note']}]"
        rows.append({
            "Signal":         sig.get("name", "—"),
            "Reading":        reading,
            "_status":        status,
            "Interpretation": sig.get("interpretation", ""),
        })

    df = pd.DataFrame(rows)

    def style_status(val):
        c = STATUS_COLOR.get(val, "#9e9e9e")
        return f"background-color:{c}20;color:{c};font-weight:bold"

    styled = (
        df.rename(columns={"_status": "Status"})
        .style.map(style_status, subset=["Status"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)


def render_fundamental_verdicts(fund: list, verdicts: list):
    st.markdown("## 5 — Fundamental Signals & Verdicts")
    col_fund, col_verd = st.columns(2)

    with col_fund:
        st.markdown("### Active Fundamental Signals")
        active = [s for s in fund if s.get("status") in ("GREEN", "RED")]
        if not active:
            st.info("No active GREEN or RED fundamental signals.")
        else:
            rows = []
            for s in active:
                rows.append({
                    "Asset":     s.get("asset", "—"),
                    "Signal":    s.get("signal", "—"),
                    "Current":   s.get("current_value", "—"),
                    "Threshold": s.get("threshold_value", "—"),
                    "_status":   s.get("status", "NO SIGNAL"),
                    "Action":    s.get("action_note", ""),
                })
            df = pd.DataFrame(rows)

            def style_s(val):
                c = STATUS_COLOR.get(val, "#9e9e9e")
                return f"background-color:{c}20;color:{c};font-weight:bold"

            st.dataframe(
                df.rename(columns={"_status": "Status"})
                  .style.map(style_s, subset=["Status"]),
                use_container_width=True, hide_index=True,
            )

    with col_verd:
        st.markdown("### Confirmation Verdicts")
        rows = []
        for v in verdicts:
            rows.append({
                "Asset":       v.get("asset", "—"),
                "Ticker":      v.get("tech_ticker", "—"),
                "Fundamental": v.get("fundamental_status", "N/A"),
                "Technical":   v.get("technical_posture", "N/A"),
                "RSI":         f"{v['rsi']:.0f}" if v.get("rsi") else "—",
                "MACD":        v.get("macd", "N/A"),
                "_verdict":    v.get("verdict", "NO SIGNAL"),
            })
        df = pd.DataFrame(rows)

        def style_fund(val):
            c = STATUS_COLOR.get(val, "#9e9e9e")
            return f"color:{c};font-weight:bold"

        def style_posture(val):
            c = POSTURE_COLOR.get(val, "#9e9e9e")
            return f"color:{c};font-weight:bold"

        def style_verdict(val):
            c = VERDICT_COLOR.get(val, "#9e9e9e")
            return f"background-color:{c}20;color:{c};font-weight:bold"

        st.dataframe(
            df.rename(columns={"_verdict": "Verdict"})
              .style
              .map(style_fund,    subset=["Fundamental"])
              .map(style_posture, subset=["Technical"])
              .map(style_verdict, subset=["Verdict"]),
            use_container_width=True, hide_index=True,
        )

    # ── CONVERGENCE alert ──────────────────────────────────────────────────────
    st.markdown("### Convergence Monitor")
    conv_hits = check_convergence(verdicts)
    if conv_hits:
        for v in conv_hits:
            st.success(
                f"⚡ **CONVERGENCE — HIGHEST CONVICTION**\n\n"
                f"**{v['asset']}** ({v['tech_ticker']})\n\n"
                f"Fundamental: {v['fundamental_status']}  |  "
                f"Technical: {v['technical_posture']}  |  "
                f"RSI: {v['rsi']:.1f}\n\n"
                f"Signals fired: {'; '.join(v['fired_signals'])}\n\n"
                f"_{v['rationale']}_"
            )
    else:
        # Find closest-to-convergence asset (highest posture rank + lowest RSI)
        def _conv_score(v):
            posture_rank = {
                "BULLISH": 5, "NEUTRAL-BULLISH": 4, "NEUTRAL": 3,
                "NEUTRAL-BEARISH": 2, "BEARISH": 1, "N/A": 0,
            }
            rsi = v.get("rsi") or 100
            return posture_rank.get(v.get("technical_posture", "N/A"), 0) - rsi / 100

        best = max(verdicts, key=_conv_score) if verdicts else None
        gld_rsi = None
        for v in verdicts:
            if v.get("asset") == "Gold":
                gld_rsi = v.get("rsi")

        msg = "No CONVERGENCE signals active.\n\nConvergence = Fundamental GREEN + Technical BULLISH + RSI < 50\n\n"
        if gld_rsi is not None:
            msg += f"**Closest: Gold (GLD) — RSI {gld_rsi:.1f}** — one tick from CONVERGENCE threshold (needs RSI < 50 + BULLISH posture)"
        elif best:
            msg += f"Closest: **{best['asset']}** ({best['tech_ticker']}) — posture: {best.get('technical_posture','N/A')}, RSI: {best.get('rsi','N/A')}"
        st.info(msg)


def render_ipo_tracker(venture: dict):
    st.markdown("## 6 — IPO Tracker (Venture)")

    edgar  = venture.get("edgar", {})
    forge  = venture.get("spacex_forge", {})
    ant    = venture.get("anthropic", {})
    ji     = venture.get("ji_moment", False)
    wu_list = venture.get("wu_wei_entry", [])

    # Ji Moment banner
    if ji:
        for filing in edgar.get("new_filings", []):
            st.error(
                f"## ⚡ JI MOMENT — S-1 FILED\n\n"
                f"**{filing.get('tracked_as','').upper()}** filed an S-1 with the SEC.\n\n"
                f"Filed: {filing.get('file_date')}  |  Accession: {filing.get('accession_no')}\n\n"
                f"URL: {filing.get('filing_url')}\n\n"
                f"**This is the irreversible threshold. Review S-1 immediately.**"
            )

    # Wu Wei banner
    if wu_list:
        st.success(
            "## ★ WU WEI ENTRY SIGNAL\n\n"
            + "\n".join(f"★ {c}" for c in wu_list)
            + "\n\nPrice >20% below IPO open — Wu Wei entry conditions met."
        )

    # SpaceX and Anthropic side-by-side
    col_sx, col_ant = st.columns(2)

    with col_sx:
        st.markdown("### SpaceX (Forge)")
        forge_price = forge.get("current_price")
        price_str   = f"${forge_price:,.2f}" if forge_price else "N/A"
        st.metric("Forge Secondary Price", price_str,
                  delta=f"{forge.get('pct_change',0):+.1f}% vs prev" if forge.get("pct_change") else None)
        if forge.get("stale"):
            st.warning(f"⚠ STALE — manual update required\nReason: {forge.get('stale_reason','unknown')}")
        else:
            st.caption(f"Updated: {forge.get('last_updated','?')}  |  Source: {forge.get('source','?')}")

    with col_ant:
        st.markdown("### Anthropic")
        if "error" not in ant:
            hiive = ant.get("last_known_hiive_price")
            arr_b = (ant.get("last_known_arr") or 0) / 1e9
            st.metric("Hiive Price", f"${hiive:,.2f}" if hiive else "N/A")
            st.metric("ARR", f"${arr_b:.1f}B")
            st.caption(
                f"Valuation: {ant.get('valuation_range','N/A')}  |  "
                f"IPO target: {ant.get('ipo_target','?')}  |  "
                f"Verified: {ant.get('last_verified','?')} ({ant.get('age_days','?')} days ago)"
            )
            if ant.get("needs_update"):
                st.warning(f"⚠ DATA IS {ant.get('age_days')} DAYS OLD — update anthropic_status.json")
        else:
            st.error(ant.get("error", "Unknown error"))

    # Pipeline table
    st.markdown("### Pre-IPO Pipeline")
    company_filings = {}
    for f in edgar.get("all_found", []):
        name = f.get("tracked_as", "")
        existing = company_filings.get(name)
        if not existing or f.get("file_date", "") > existing.get("file_date", ""):
            company_filings[name] = f

    pipeline_rows = []
    for display_name, _ in TRACKED_COMPANIES:
        filing = company_filings.get(display_name)
        if filing:
            is_new = any(
                f.get("accession_no") == filing.get("accession_no")
                for f in edgar.get("new_filings", [])
            )
            s1_status = "⚡ NEW — JI MOMENT" if is_new else "FILED (seen)"
            pipeline_rows.append({
                "Company":    display_name,
                "S-1 Status": s1_status,
                "Filed":      filing.get("file_date", "—"),
                "Accession":  filing.get("accession_no", "—"),
            })
        else:
            pipeline_rows.append({
                "Company":    display_name,
                "S-1 Status": "No S-1 detected",
                "Filed":      "—",
                "Accession":  "—",
            })

    st.dataframe(pd.DataFrame(pipeline_rows), use_container_width=True, hide_index=True)

    # EDGAR status
    last_check = edgar.get("last_check", "never")
    if last_check and last_check != "never":
        try:
            from datetime import datetime as _dt, timezone as _tz
            ts = _dt.fromisoformat(last_check)
            age_days = (_dt.now(_tz.utc) - ts).days
        except Exception:
            age_days = "?"
    else:
        age_days = "?"
    st.caption(
        f"EDGAR last check: {last_check}  |  "
        f"Days since check: {age_days}  |  "
        f"Lookback: {edgar.get('lookback_days','?')} days"
    )


def render_outlook():
    st.markdown("## 7 — Outlook")

    existing = ""
    try:
        if OUTLOOK_FILE.exists():
            existing = OUTLOOK_FILE.read_text().strip()
    except (OSError, PermissionError):
        pass

    if existing:
        st.markdown("**Current saved outlook:**")
        st.markdown(existing)
        st.divider()

    st.markdown("**Edit weekly Layer 5 outlook:**")
    st.markdown(
        "Include three posture lines: **OVERALL MARKET** | **NATURAL RESOURCES SLEEVE** | **IPO/PRE-IPO**"
    )
    new_text = st.text_area(
        "Weekly outlook",
        value=existing,
        height=220,
        label_visibility="collapsed",
    )

    if st.button("💾 Save Outlook"):
        try:
            OUTLOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
            OUTLOOK_FILE.write_text(new_text.strip())
            st.success("Outlook saved to data/layer5_outlook.txt")
        except (OSError, PermissionError):
            st.warning("Cannot save to disk on this environment — copy your text locally to preserve it.")


def render_footer():
    st.divider()
    st.markdown(
        "<div style='text-align:center;color:#9e9e9e;font-size:0.85em'>"
        "Data source: yfinance &nbsp;|&nbsp; "
        "<strong>⚠ VERIFY ALL READINGS BEFORE ACTING</strong><br>"
        "Dashboard v2.0 — Taoist CIO Team — Conscious Capital Wealth Management"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d  %-I:%M:%S %p ET")

    with st.spinner("Loading market data…"):
        data, prices, tech, l2, fund, verdicts, venture = load_all_data()

    render_header(timestamp)
    section = render_sidebar(tech, l2, fund, verdicts)

    # Anchor mapping — render all sections but jump via query param
    sec_num = section.split("—")[0].strip()

    if sec_num == "1":
        render_macro(tech, data)
    elif sec_num == "2":
        render_technical(tech, prices)
    elif sec_num == "3":
        render_charts(data, tech)
    elif sec_num == "4":
        render_layer2(l2)
    elif sec_num == "5":
        render_fundamental_verdicts(fund, verdicts)
    elif sec_num == "6":
        render_ipo_tracker(venture)
    elif sec_num == "7":
        render_outlook()

    render_footer()


if __name__ == "__main__":
    main()
