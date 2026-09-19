"""Look and feel.

Visual idea: an aeronautical chart on the desk (cool pale-blue paper, navy ink),
signage typography (Barlow Condensed for readouts, Barlow for text), and delay
risk graded with the same four colours pilots know from METAR flight
categories. The boarding-pass ticket is the one loud element; the rest is quiet.
"""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

from . import config, modeling

INK = config.INK
PAPER = config.PAPER
AMBER = config.AMBER
FONT = "Barlow, 'Helvetica Neue', Arial, sans-serif"
HEAD = "'Barlow Condensed', 'Arial Narrow', Arial, sans-serif"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap');

html, body, .stApp, [class*="css"] { font-family: Barlow, 'Helvetica Neue', Arial, sans-serif; }
.stApp { background: #E8EEF2; color: #12233F; }
.block-container { padding-top: 1.8rem; max-width: 1240px; }
h1, h2, h3, h4 { font-family: 'Barlow Condensed', 'Arial Narrow', sans-serif !important; font-weight: 600 !important; color: #12233F; letter-spacing: 0.01em; }
h3 { font-size: 1.7rem !important; }

/* ---------- masthead ---------- */
.mast { display: flex; justify-content: space-between; align-items: flex-end; gap: 28px; flex-wrap: wrap;
        border-bottom: 3px solid #12233F; padding-bottom: 14px; margin-bottom: 8px; }
.mast .title { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 3.4rem; line-height: 0.95; }
.mast .tag { margin: 8px 0 0; font-size: 1.08rem; color: #3B4D69; max-width: 44ch; }
.readouts { display: flex; gap: 30px; flex-wrap: wrap; }
.readout .n { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 2.1rem; line-height: 1;
              font-variant-numeric: tabular-nums; }
.readout .l { font-size: 0.9rem; color: #4A5B76; margin-top: 2px; }

/* ---------- tabs ---------- */
button[data-baseweb="tab"] { font-family: 'Barlow Condensed', sans-serif; font-size: 1.2rem; font-weight: 600; }
div[data-baseweb="tab-highlight"] { background-color: #C9861A !important; height: 3px !important; }

/* ---------- boarding pass ---------- */
.ticket { position: relative; display: flex; background: #FBFCFD; margin: 6px 0 22px;
          box-shadow: 0 1px 0 #B9C6D2, 0 10px 24px -14px rgba(18,35,63,0.45); }
.ticket::before, .ticket::after { content: ""; position: absolute; left: 33%; width: 22px; height: 22px;
          border-radius: 50%; background: #E8EEF2; transform: translateX(-50%); }
.ticket::before { top: -11px; }
.ticket::after { bottom: -11px; }
.ticket .stub { width: 33%; padding: 20px 22px; background: #F0F4F7; border-right: 2px dashed #B9C6D2; }
.ticket .main { flex: 1; padding: 20px 28px 22px; }
.ticket .k { font-size: 0.86rem; color: #5B6B84; margin-top: 14px; }
.ticket .k:first-child { margin-top: 0; }
.ticket .v { font-family: 'Barlow Condensed', sans-serif; font-weight: 600; font-size: 1.4rem; line-height: 1.12; overflow-wrap: anywhere; }
.ticket .top { display: flex; align-items: center; gap: 14px; }
.ticket .code { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.9rem; color: #fff; padding: 1px 14px 2px; }
.ticket .risk { font-size: 1.1rem; font-weight: 500; }
.ticket .pct { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 5.6rem; line-height: 0.95;
               font-variant-numeric: tabular-nums; margin-top: 10px; }
.ticket .cap { color: #3B4D69; margin-top: 2px; }
.ticket .msg { margin-top: 14px; font-size: 1.04rem; max-width: 50ch; }
.gauge { position: relative; display: flex; height: 12px; margin-top: 22px; }
.gauge span { display: block; height: 100%; }
.gauge .mark { position: absolute; top: -7px; width: 4px; height: 26px; background: #12233F; transform: translateX(-2px); }
.scale { display: flex; justify-content: space-between; font-size: 0.82rem; color: #5B6B84; margin-top: 6px; }

/* ---------- flight strips (legend) ---------- */
.strips { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; margin: 8px 0 18px; }
.strip { background: #F6F9FB; border-left: 8px solid var(--c); padding: 8px 12px 9px; }
.strip .s-code { font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.35rem; color: var(--c); line-height: 1.1; }
.strip .s-rng { font-size: 0.92rem; color: #3B4D69; }

.note { color: #4A5B76; font-size: 0.92rem; }
.demo-flag { background: #FBEFD5; border-left: 6px solid #C9861A; padding: 10px 14px; margin: 10px 0 6px; }

@media (max-width: 760px) {
  .mast .title { font-size: 2.5rem; }
  .ticket { flex-direction: column; }
  .ticket::before, .ticket::after { display: none; }
  .ticket .stub { width: auto; border-right: 0; border-bottom: 2px dashed #B9C6D2; }
  .ticket .pct { font-size: 4.4rem; }
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def e(text) -> str:
    return html.escape(str(text))


# --------------------------------------------------------------------------
# Blocks of HTML (kept on one line each so Markdown never treats them as code)
# --------------------------------------------------------------------------


def masthead(readouts: list[tuple[str, str]]) -> None:
    items = "".join(
        f'<div class="readout"><div class="n">{e(n)}</div><div class="l">{e(label)}</div></div>'
        for n, label in readouts
    )
    st.markdown(
        f'<div class="mast"><div><div class="title">{e(config.APP_NAME)}</div>'
        f'<p class="tag">{e(config.APP_TAGLINE)}</p></div><div class="readouts">{items}</div></div>',
        unsafe_allow_html=True,
    )


def demo_flag() -> None:
    st.markdown(
        '<div class="demo-flag"><b>Demo data.</b> These artifacts were built from synthetic flights, so every '
        "chart and forecast here is made up. Run <code>python train_model.py</code> on the real "
        "train.csv to replace them.</div>",
        unsafe_allow_html=True,
    )


def ticket(carrier: str, airport: str, block: str, when: str, p: float, base_rate: float) -> None:
    cat = modeling.flight_category(p, base_rate)
    bounds = modeling.category_bounds(base_rate)
    edges = [0.0] + bounds + [1.0]
    zones = "".join(
        f'<span style="width:{(edges[i + 1] - edges[i]) * 100:.2f}%;background:{c["color"]}"></span>'
        for i, c in enumerate(config.CATEGORIES)
    )
    mark = f'<div class="mark" style="left:{min(max(p, 0.0), 1.0) * 100:.2f}%"></div>'
    html_ticket = (
        '<div class="ticket">'
        '<div class="stub">'
        f'<div class="k">Airline</div><div class="v">{e(carrier)}</div>'
        f'<div class="k">Departing</div><div class="v">{e(airport)}</div>'
        f'<div class="k">Slot</div><div class="v">{e(block)}</div>'
        f'<div class="k">When</div><div class="v">{e(when)}</div>'
        "</div>"
        '<div class="main">'
        f'<div class="top"><span class="code" style="background:{cat["color"]}">{cat["code"]}</span>'
        f'<span class="risk">{e(cat["label"])}</span></div>'
        f'<div class="pct" style="color:{cat["color"]}">{p * 100:.0f}%</div>'
        '<div class="cap">chance of leaving more than 15 minutes late</div>'
        f'<div class="gauge">{zones}{mark}</div>'
        f'<div class="scale"><span>0%</span><span>Average departure: {base_rate * 100:.0f}%</span><span>100%</span></div>'
        f'<div class="msg">{e(cat["message"])}</div>'
        "</div></div>"
    )
    st.markdown(html_ticket, unsafe_allow_html=True)


def legend(base_rate: float) -> None:
    bounds = modeling.category_bounds(base_rate)
    ranges = [
        f"Under {bounds[0] * 100:.0f}%",
        f"{bounds[0] * 100:.0f}% to {bounds[1] * 100:.0f}%",
        f"{bounds[1] * 100:.0f}% to {bounds[2] * 100:.0f}%",
        f"Over {bounds[2] * 100:.0f}%",
    ]
    cards = "".join(
        f'<div class="strip" style="--c:{c["color"]}"><div class="s-code">{c["code"]}</div>'
        f'<div class="s-rng">{e(c["label"])}, {e(r)}</div></div>'
        for c, r in zip(config.CATEGORIES, ranges)
    )
    st.markdown(f'<div class="strips">{cards}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Plotly
# --------------------------------------------------------------------------


def style_fig(fig: go.Figure, height: int = 360, legend: bool = False) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=14),
        title=dict(font=dict(family=HEAD, size=22, color=INK), x=0.0, xanchor="left"),
        showlegend=legend,
        hoverlabel=dict(font_family=FONT),
    )
    fig.update_xaxes(gridcolor="#CFD9E1", linecolor=INK, zeroline=False)
    fig.update_yaxes(gridcolor="#CFD9E1", zeroline=False)
    return fig


def rate_bars(records: list[dict], base_rate: float, title: str, label_fn=None, horizontal=False, height=340) -> go.Figure:
    """Bar chart of late-departure rate, each bar coloured by its flight category."""
    labels = [label_fn(r["label"]) if label_fn else str(r["label"]) for r in records]
    rates = [r["rate"] for r in records]
    colors = [modeling.flight_category(v, base_rate)["color"] for v in rates]
    hover = [f"{lab}<br>{v * 100:.1f}% late<br>{int(r['n']):,} departures" for lab, v, r in zip(labels, rates, records)]

    if horizontal:
        fig = go.Figure(go.Bar(y=labels, x=rates, orientation="h", marker_color=colors, hovertext=hover, hoverinfo="text"))
        fig.add_vline(x=base_rate, line_dash="dot", line_color=INK)
        fig.update_xaxes(tickformat=".0%")
    else:
        fig = go.Figure(go.Bar(x=labels, y=rates, marker_color=colors, hovertext=hover, hoverinfo="text"))
        fig.add_hline(y=base_rate, line_dash="dot", line_color=INK)
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title=title)
    return style_fig(fig, height=height)
