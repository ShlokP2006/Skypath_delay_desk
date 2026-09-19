"""SkyPath Delay Desk: will this departure push back on time?

Run with:  streamlit run app.py
(Train first with:  python train_model.py)
"""
from __future__ import annotations

import datetime as dt
import json
import math

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, modeling, ui

st.set_page_config(page_title=config.APP_NAME, page_icon="✈️", layout="wide")
ui.inject_css()

_NEW_API = tuple(int(x) for x in st.__version__.split(".")[:2]) >= (1, 50)


def chart(fig: go.Figure) -> None:
    if _NEW_API:
        st.plotly_chart(fig, width="stretch", theme=None)
    else:
        st.plotly_chart(fig, use_container_width=True, theme=None)


def table(df: pd.DataFrame, **kwargs) -> None:
    if _NEW_API:
        st.dataframe(df, hide_index=True, width="stretch", **kwargs)
    else:
        st.dataframe(df, hide_index=True, use_container_width=True, **kwargs)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
ARTIFACTS = [config.MODEL_PATH, config.LOOKUPS_PATH, config.METRICS_PATH, config.EDA_PATH]


@st.cache_resource(show_spinner="Loading the model...")
def load_artifacts(stamp: tuple):
    bundle = joblib.load(config.MODEL_PATH)
    lookups = json.loads(config.LOOKUPS_PATH.read_text())
    metrics = json.loads(config.METRICS_PATH.read_text())
    eda = json.loads(config.EDA_PATH.read_text())
    return bundle, lookups, metrics, eda


if not all(p.exists() for p in ARTIFACTS):
    ui.masthead([])
    st.info("No trained model found yet. Train one, then refresh this page.")
    st.code(
        "# with the real data unzipped into data/\n"
        "python train_model.py --sample-frac 0.4\n\n"
        "# or a 1-minute smoke test on synthetic data\n"
        "python train_model.py --demo",
        language="bash",
    )
    st.stop()

bundle, lk, metrics, eda = load_artifacts(tuple(p.stat().st_mtime for p in ARTIFACTS))
BASE = metrics["base_rate_train"]


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def snap(value: float, step: float, lo: float, hi: float) -> float:
    value = round(value / step) * step
    return float(min(max(value, lo), hi))


def typical_traffic(airport: str, block: str, fallback: int) -> int:
    value = lk["concurrent"].get(airport, {}).get(block)
    return int(round(value)) if value is not None else fallback


def weather_slider(label: str, key: str, defaults: dict, step: float, help_text: str | None = None) -> float:
    lo, hi = lk["weather_range"][key]
    lo, hi = math.floor(lo), math.ceil(hi)
    if hi <= lo:
        hi = lo + 1
    default = defaults.get(key)
    default = snap(default if default is not None else (lo + hi) / 2, step, lo, hi)
    return st.slider(label, float(lo), float(hi), default, step=step, help=help_text)


# --------------------------------------------------------------------------
# Masthead
# --------------------------------------------------------------------------
ui.masthead(
    [
        (f"{BASE:.0%}", "of departures leave 15+ min late"),
        (f"{metrics['roc_auc']:.2f}", "ROC-AUC on held-out flights"),
        (f"{metrics['n_train']:,}", "departures studied"),
    ]
)
if metrics.get("source") == "demo":
    ui.demo_flag()

tabs = st.tabs(
    ["Pre-flight briefing", "Airport radar", "Airlines and fleet", "Sky and clock", "Model bay", "Field notes"]
)


# --------------------------------------------------------------------------
# 1. Pre-flight briefing
# --------------------------------------------------------------------------
with tabs[0]:
    left, right = st.columns([5, 7], gap="large")

    with left:
        st.subheader("Flight plan")
        carrier = st.selectbox("Airline", sorted(lk["carriers"]))
        airport = st.selectbox("Departing airport", sorted(lk["airports"]))
        previous = st.selectbox(
            "Aircraft arrived from",
            lk["previous_airports"],
            help=f"Pick {config.FIRST_LEG} if this is the aircraft's first flight of the day.",
        )

        today = dt.date.today()
        c1, c2 = st.columns(2)
        month = c1.selectbox(
            "Month", list(range(1, 13)), index=today.month - 1, format_func=lambda m: config.MONTHS[m - 1]
        )
        dow = c2.selectbox(
            "Day", list(range(1, 8)), index=today.isoweekday() - 1, format_func=lambda d: config.DAYS[d - 1]
        )

        blocks = lk["dep_blocks"]
        default_block = blocks.index("0800-0859") if "0800-0859" in blocks else len(blocks) // 3
        block = st.selectbox("Departure slot (local time)", blocks, index=default_block)
        distance = st.selectbox(
            "Trip length", list(config.DISTANCE_GROUPS), index=3, format_func=lambda g: config.DISTANCE_GROUPS[g]
        )

        if previous == config.FIRST_LEG:
            segment = 1
            st.caption("First leg of the day for this aircraft.")
        else:
            segment = st.slider(
                "Leg of the day for this aircraft",
                2, 10, 2,
                help="How many flights this tail number has already flown today, plus one.",
            )

        st.subheader("Aircraft and airport")
        cr = lk["carriers"].get(carrier, {})
        seat_lo, seat_hi = (int(v) for v in lk["seat_range"])
        default_seats = int(cr.get("seats") or (seat_lo + seat_hi) // 2)
        seats = st.number_input("Seats on the aircraft", 10, max(seat_hi, default_seats, 60), default_seats, step=1)
        max_age = int(max(lk["max_plane_age"], 5))
        default_age = int(min(cr.get("plane_age") or 10, max_age))
        age = st.slider("Aircraft age (years)", 0, max_age, default_age)

        default_conc = typical_traffic(airport, block, 10)
        max_conc = int(max(lk["max_concurrent"], default_conc + 5))
        concurrent = st.slider(
            "Flights leaving in the same slot",
            1, max_conc, int(min(max(default_conc, 1), max_conc)),
            help="Departures from this airport in the same time block. Starts at the typical value for the slot.",
        )

        with st.expander("Conditions at the departure airport"):
            wx = lk["weather"].get(airport, {}).get(str(month), {})
            st.caption(
                "Starts at the typical day for this airport and month. Rain and snow are in inches; "
                "temperature and wind use the dataset's own units."
            )
            tmax = weather_slider("Max temperature", "TMAX", wx, 1.0)
            prcp = weather_slider("Rain", "PRCP", wx, 0.05)
            snow = weather_slider("Snowfall", "SNOW", wx, 0.1)
            snwd = weather_slider("Snow on the ground", "SNWD", wx, 0.5)
            awnd = weather_slider("Wind", "AWND", wx, 0.5)

    inputs = dict(
        MONTH=month, DAY_OF_WEEK=dow, DISTANCE_GROUP=distance, DEP_BLOCK=block,
        SEGMENT_NUMBER=segment, CONCURRENT_FLIGHTS=concurrent, NUMBER_OF_SEATS=seats,
        CARRIER_NAME=carrier, DEPARTING_AIRPORT=airport, PREVIOUS_AIRPORT=previous,
        PLANE_AGE=age, PRCP=prcp, SNOW=snow, SNWD=snwd, TMAX=tmax, AWND=awnd,
    )
    row = modeling.build_row(inputs, lk)
    p = float(modeling.predict(bundle, [row])[0])

    with right:
        st.subheader("Briefing")
        ui.ticket(carrier, airport, block, f"{config.DAYS[dow - 1]}, {config.MONTHS[month - 1]}", p, BASE)

        # Same flight, every slot of the day.
        slot_rows = []
        for b in blocks:
            r = dict(row)
            r["DEP_BLOCK"] = b
            r["CONCURRENT_FLIGHTS"] = typical_traffic(airport, b, concurrent)
            slot_rows.append(r)
        slot_p = modeling.predict(bundle, slot_rows)

        def slot_label(b: str) -> str:
            return "Before 06" if b.startswith("0001") else f"{b[:2]}:00"

        colors = [modeling.flight_category(v, BASE)["color"] for v in slot_p]
        fig = go.Figure(
            go.Bar(
                x=[slot_label(b) for b in blocks],
                y=slot_p,
                marker_color=colors,
                marker_line_color=ui.INK,
                marker_line_width=[3 if b == block else 0 for b in blocks],
                hovertext=[f"{b}<br>{v * 100:.0f}% chance of a late departure" for b, v in zip(blocks, slot_p)],
                hoverinfo="text",
            )
        )
        fig.add_hline(y=BASE, line_dash="dot", line_color=ui.INK)
        fig.update_yaxes(tickformat=".0%")
        fig.update_xaxes(tickangle=-45)
        fig.update_layout(title="Same flight, every slot of the day (outlined bar is your slot)")
        chart(ui.style_fig(fig, height=300))

        best = int(np.argmin(slot_p))
        if blocks[best] != block and slot_p[best] < p - 0.005:
            st.markdown(
                f"The calmest slot on this plan is **{blocks[best]}** at **{slot_p[best]:.0%}**, "
                f"against {p:.0%} for {block}. Other slots assume typical traffic for that time."
            )
        else:
            st.markdown("Your slot is already among the calmest on this plan.")

        # What is pushing the odds.
        contrib = modeling.contributions(bundle, row)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:6]
        fig = go.Figure(
            go.Bar(
                y=[config.FEATURE_LABELS.get(f, f) for f in top.index],
                x=top.values,
                orientation="h",
                marker_color=["#D1352B" if v > 0 else "#2E9B4E" for v in top.values],
                text=[f"x{math.exp(v):.2f}" for v in top.values],
                textposition="outside",
                hoverinfo="skip",
                cliponaxis=False,
            )
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(showticklabels=False, zeroline=True, zerolinecolor=ui.INK)
        fig.update_layout(title="What is moving the odds (red raises them, green lowers them)")
        chart(ui.style_fig(fig, height=300))

        with st.expander("How to read the colours"):
            st.write(
                "Delay risk borrows the four flight categories pilots use for weather. "
                "The cut-offs are multiples of the average delay rate in the data."
            )
            ui.legend(BASE)

        st.caption(
            "A statistical estimate from historical departures, not a status report. "
            "Check the airline's app for the real thing."
        )


# --------------------------------------------------------------------------
# 2. Airport radar
# --------------------------------------------------------------------------
with tabs[1]:
    ap = pd.DataFrame.from_dict(lk["airports"], orient="index").rename_axis("airport").reset_index()
    ap = ap.dropna(subset=["latitude", "longitude", "delay_rate"])

    st.subheader("Where departures run late")
    if ap.empty:
        st.info("No airport coordinates found in the data.")
    else:
        top_cut = max(int(ap["flights"].quantile(0.75)), 1)
        floor = st.slider(
            "Hide airports with fewer departures than this in the data",
            0, top_cut, 0, step=max(top_cut // 50, 1),
        )
        shown = ap[ap["flights"] >= floor]

        fig = go.Figure(
            go.Scattergeo(
                lon=shown["longitude"],
                lat=shown["latitude"],
                text=shown["airport"],
                customdata=np.stack([shown["delay_rate"] * 100, shown["flights"]], axis=-1),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]:.1f}% late<br>%{customdata[1]:,.0f} departures<extra></extra>",
                marker=dict(
                    size=shown["flights"],
                    sizemode="area",
                    sizeref=2.0 * float(shown["flights"].max()) / (38.0**2),
                    sizemin=7,
                    color=shown["delay_rate"],
                    colorscale=[[0, "#2E9B4E"], [0.5, "#E7C84A"], [1, "#D1352B"]],
                    cmin=float(shown["delay_rate"].min()),
                    cmax=float(shown["delay_rate"].max()),
                    colorbar=dict(title="Late", tickformat=".0%", len=0.7),
                    line=dict(width=1, color=ui.INK),
                    opacity=0.9,
                ),
            )
        )
        fig.update_geos(
            scope="usa",
            projection_type="albers usa",
            showland=True, landcolor="#F7F9FA",
            showlakes=True, lakecolor="#CFE0EE",
            showsubunits=True, subunitcolor="#B9C6D2",
            showcountries=False, bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(title="Bubble size is traffic, colour is late-departure rate")
        chart(ui.style_fig(fig, height=520))

        board = pd.DataFrame(
            {"Airport": shown["airport"], "Late rate": shown["delay_rate"] * 100, "Departures": shown["flights"]}
        )
        cfg = {
            "Late rate": st.column_config.ProgressColumn(
                "Late rate", format="%.1f%%", min_value=0.0, max_value=float(max(board["Late rate"].max(), 1) * 1.1)
            ),
            "Departures": st.column_config.NumberColumn("Departures", format="%d"),
        }
        b1, b2 = st.columns(2, gap="large")
        with b1:
            st.subheader("Watch list")
            table(board.sort_values("Late rate", ascending=False).head(10), column_config=cfg)
        with b2:
            st.subheader("On the money")
            table(board.sort_values("Late rate").head(10), column_config=cfg)


# --------------------------------------------------------------------------
# 3. Airlines and fleet
# --------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Airlines and aircraft")
    cr_df = pd.DataFrame.from_dict(lk["carriers"], orient="index").rename_axis("carrier").reset_index()
    cr_df = cr_df.sort_values("delay_rate")
    records = [{"label": r.carrier, "rate": r.delay_rate, "n": r.flights} for r in cr_df.itertuples()]
    chart(
        ui.rate_bars(
            records, BASE, "Late-departure rate by airline (dotted line is the average)",
            horizontal=True, height=max(320, 28 * len(records) + 90),
        )
    )
    st.markdown(
        '<div class="note">Airlines fly different airports, hours and seasons, so this is not a clean '
        "scorecard. The briefing tab holds those factors fixed.</div>",
        unsafe_allow_html=True,
    )

    f1, f2 = st.columns(2, gap="large")
    with f1:
        chart(ui.rate_bars(eda["plane_age"], BASE, "By aircraft age (years)"))
    with f2:
        chart(ui.rate_bars(eda["seats"], BASE, "By seats on the aircraft"))


# --------------------------------------------------------------------------
# 4. Sky and clock
# --------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Time of day and day of week")
    heat = eda["heat"]
    z = np.array(heat["z"], dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=heat["blocks"],
            y=[config.DAYS[d - 1] for d in heat["days"]],
            colorscale=[[0, "#2E9B4E"], [0.5, "#E7C84A"], [1, "#D1352B"]],
            zmin=float(np.nanmin(z)), zmax=float(np.nanmax(z)),
            colorbar=dict(tickformat=".0%", title="Late"),
            hovertemplate="%{y}, %{x}<br>%{z:.1%} late<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-60)
    chart(ui.style_fig(fig, height=380))

    t1, t2 = st.columns(2, gap="large")
    with t1:
        chart(ui.rate_bars(eda["month"], BASE, "By month", label_fn=lambda m: config.MONTHS[int(m) - 1][:3]))
    with t2:
        chart(ui.rate_bars(eda["day_of_week"], BASE, "By day", label_fn=lambda d: config.DAYS[int(d) - 1][:3]))

    st.subheader("Weather")
    element = st.radio("Weather element", list(eda["weather"]), horizontal=True, label_visibility="collapsed")
    chart(ui.rate_bars(eda["weather"][element], BASE, f"Late-departure rate by {element.lower()}"))
    st.markdown(
        '<div class="note">Weather is one daily reading per airport, so it cannot see a thunderstorm '
        "that only hit at 4 pm. Rain and snow are in inches.</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Ramp congestion and knock-on delays")
    k1, k2 = st.columns(2, gap="large")
    with k1:
        chart(ui.rate_bars(eda["concurrent"], BASE, "By flights leaving in the same slot"))
    with k2:
        chart(
            ui.rate_bars(
                eda["segment"], BASE, "By the aircraft's leg of the day",
                label_fn=lambda s: f"Leg {int(s)}" if int(s) < 8 else "Leg 8+",
            )
        )


# --------------------------------------------------------------------------
# 5. Model bay
# --------------------------------------------------------------------------
with tabs[4]:
    st.subheader("How good is the forecast?")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", help="0.5 is a coin flip, 1.0 is perfect ranking.")
    m2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}", help=f"A no-skill model scores about {metrics['base_rate_test']:.2f} here.")
    m3.metric(
        "Top-decile lift", f"{metrics['top_decile_lift']:.1f}x",
        help="Late-departure rate among the 10% of flights the model ranks riskiest, divided by the average.",
    )
    m4.metric("Recall at the alert line", f"{metrics['recall']:.0%}", help=f"Precision there is {metrics['precision']:.0%}.")

    st.markdown(
        f"Delays are the minority outcome (about {metrics['base_rate_test']:.0%} of test departures), so raw accuracy "
        "would flatter a model that never predicts a delay. These measures judge how well flights are ranked by risk "
        f"instead. The alert line is a {metrics['threshold']:.0%} probability, picked to balance precision and recall."
    )

    g1, g2 = st.columns(2, gap="large")
    with g1:
        roc = metrics["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#8A98AD"), name="Coin flip"))
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=ui.INK, width=3), name="Model"))
        fig.update_xaxes(title="False alarm rate")
        fig.update_yaxes(title="Delays caught")
        fig.update_layout(title="ROC curve")
        chart(ui.style_fig(fig, height=340, legend=True))
    with g2:
        cal = metrics["calibration"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#8A98AD"), name="Perfect"))
        fig.add_trace(
            go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
                       line=dict(color=config.AMBER, width=3), marker=dict(size=8), name="Model")
        )
        fig.update_xaxes(title="Forecast chance of delay", tickformat=".0%")
        fig.update_yaxes(title="Share actually late", tickformat=".0%")
        fig.update_layout(title="Calibration: do 30% forecasts come true 30% of the time?")
        chart(ui.style_fig(fig, height=340, legend=True))

    imp = list(metrics["importance"].items())[:12]
    fig = go.Figure(
        go.Bar(
            y=[config.FEATURE_LABELS.get(k, k) for k, _ in imp],
            x=[v for _, v in imp],
            orientation="h",
            marker_color=ui.INK,
            hovertemplate="%{y}: %{x:.1%} of total gain<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickformat=".0%")
    fig.update_layout(title="What the model leans on most (share of total gain)")
    chart(ui.style_fig(fig, height=400))

    cm = metrics["confusion"]
    st.markdown("**Test flights at the alert line**")
    table(
        pd.DataFrame(
            {
                "": ["Actually on time", "Actually late"],
                "Forecast on time": [f"{cm['tn']:,}", f"{cm['fn']:,}"],
                "Forecast late": [f"{cm['fp']:,}", f"{cm['tp']:,}"],
            }
        )
    )
    st.caption(
        f"Trained {metrics['trained_at'][:10]} on {metrics['n_train']:,} departures "
        f"({metrics['sample_frac']:.0%} of the file), evaluated on {metrics['n_test']:,} held-out departures."
    )


# --------------------------------------------------------------------------
# 6. Field notes
# --------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Field notes")
    for name in ["important_topics.md", "data_dictionary.md"]:
        path = config.DOCS_DIR / name
        if path.exists():
            st.markdown(path.read_text(encoding="utf-8"))
    st.subheader("Source documentation")
    for name in ["raw_data_documentation.txt", "train_sets_documentation.txt"]:
        path = config.DOCS_DIR / name
        if path.exists():
            with st.expander(name):
                st.code(path.read_text(encoding="utf-8"), language=None)
