import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.pricing import bs_price
from core.greeks import all_greeks
from core.backtesting import simulate_gbm
from core.market_data import POPULAR_TICKERS, get_live_params

st.set_page_config(page_title="Risk Dashboard", layout="wide")
st.title("Real-Time Risk Dashboard")

# ── Live Data Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Live Market Data")
_rd_ticker = st.sidebar.selectbox("Ticker", POPULAR_TICKERS, index=0, key="rd_ticker")

_prev_ticker = st.session_state.get("rd_loaded_ticker")
_need_load = st.sidebar.button("Load Live Data", key="rd_load") or (
    _prev_ticker is not None and _prev_ticker != _rd_ticker
)

if _need_load:
    with st.sidebar:
        with st.spinner("Fetching…"):
            lp = get_live_params(_rd_ticker)
            st.session_state["rd_params"] = lp
            st.session_state["rd_loaded_ticker"] = _rd_ticker
            _s = lp["spot"]
            _r = lp["r"]
            _sig = lp["sigma"]
            st.session_state["book"] = [
                {"label": "ATM Call 1M", "S": _s, "K": float(round(_s)), "T": 1/12, "r": _r, "sigma": _sig, "option_type": "call", "quantity": 50},
                {"label": "OTM Put 1M", "S": _s, "K": float(round(_s * 0.95)), "T": 1/12, "r": _r, "sigma": round(_sig * 1.1, 4), "option_type": "put", "quantity": -30},
                {"label": "ITM Call 3M", "S": _s, "K": float(round(_s * 0.95)), "T": 0.25, "r": _r, "sigma": round(_sig * 0.9, 4), "option_type": "call", "quantity": 20},
                {"label": "ATM Put 3M", "S": _s, "K": float(round(_s)), "T": 0.25, "r": _r, "sigma": _sig, "option_type": "put", "quantity": -25},
                {"label": "OTM Call 6M", "S": _s, "K": float(round(_s * 1.1)), "T": 0.5, "r": _r, "sigma": round(_sig * 0.95, 4), "option_type": "call", "quantity": 15},
                {"label": "Deep OTM Put 6M", "S": _s, "K": float(round(_s * 0.85)), "T": 0.5, "r": _r, "sigma": round(_sig * 1.4, 4), "option_type": "put", "quantity": 40},
            ]
            st.rerun()

_rp = st.session_state.get("rd_params", {})
if _rp:
    st.sidebar.caption(f"**{_rp.get('name', _rd_ticker)}** — ${_rp.get('spot', 0):.2f}")
st.sidebar.divider()

SAMPLE_BOOK = [
    {"label": "ATM Call 1M", "S": 100, "K": 100, "T": 1/12, "r": 0.05, "sigma": 0.20, "option_type": "call", "quantity": 50},
    {"label": "OTM Put 1M", "S": 100, "K": 95, "T": 1/12, "r": 0.05, "sigma": 0.22, "option_type": "put", "quantity": -30},
    {"label": "ITM Call 3M", "S": 100, "K": 95, "T": 0.25, "r": 0.05, "sigma": 0.18, "option_type": "call", "quantity": 20},
    {"label": "ATM Put 3M", "S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.20, "option_type": "put", "quantity": -25},
    {"label": "OTM Call 6M", "S": 100, "K": 110, "T": 0.5, "r": 0.05, "sigma": 0.19, "option_type": "call", "quantity": 15},
    {"label": "Deep OTM Put 6M", "S": 100, "K": 85, "T": 0.5, "r": 0.05, "sigma": 0.28, "option_type": "put", "quantity": 40},
]

if "book" not in st.session_state:
    st.session_state["book"] = SAMPLE_BOOK.copy()

book = st.session_state["book"]

if st.sidebar.button("Reset to Sample Book"):
    st.session_state["book"] = SAMPLE_BOOK.copy()
    st.session_state.pop("rd_params", None)
    st.rerun()

var_confidence = st.sidebar.slider("VaR Confidence", 0.90, 0.99, 0.95, 0.01)
var_horizon = st.sidebar.selectbox("VaR Horizon (days)", [1, 5, 10], index=0)
n_simulations = st.sidebar.selectbox("Monte Carlo Paths", [1000, 5000, 10000], index=1)

# ---------- Compute portfolio-level risk ----------
total_value = 0
total_greeks = {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0, "Vanna": 0, "Volga": 0}
position_details = []

for p in book:
    price = bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
    g = all_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
    pos_val = price * p["quantity"]
    total_value += pos_val
    for k in total_greeks:
        total_greeks[k] += g[k] * p["quantity"]
    position_details.append({
        "Label": p["label"],
        "Type": p["option_type"],
        "K": p["K"],
        "T (y)": f"{p['T']:.3f}",
        "σ": f"{p['sigma']:.2%}",
        "Qty": p["quantity"],
        "Price": f"${price:.2f}",
        "Value": f"${pos_val:.2f}",
        "Delta": f"{g['Delta'] * p['quantity']:.2f}",
        "Gamma": f"{g['Gamma'] * p['quantity']:.4f}",
        "Vega": f"{g['Vega'] * p['quantity']:.2f}",
    })

# ---------- Header Metrics ----------
st.subheader("Portfolio Overview")
cols = st.columns(7)
cols[0].metric("Total Value", f"${total_value:.2f}")
cols[1].metric("Net Delta", f"{total_greeks['Delta']:.2f}")
cols[2].metric("Net Gamma", f"{total_greeks['Gamma']:.4f}")
cols[3].metric("Net Vega", f"{total_greeks['Vega']:.2f}")
cols[4].metric("Net Theta", f"{total_greeks['Theta']:.2f}")
cols[5].metric("Net Vanna", f"{total_greeks['Vanna']:.4f}")
cols[6].metric("Net Volga", f"{total_greeks['Volga']:.4f}")

st.divider()

tab_book, tab_var, tab_exposure, tab_alerts = st.tabs([
    "Position Book", "VaR Analysis", "Greeks Exposure", "Risk Alerts"
])

with tab_book:
    st.subheader("Current Positions")
    st.dataframe(pd.DataFrame(position_details), width="stretch")

    fig_pie = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]],
                            subplot_titles=["Value Contribution", "Delta Contribution"])
    labels = [p["label"] for p in book]
    values_abs = [abs(bs_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"]) * p["quantity"]) for p in book]
    deltas_abs = [abs(all_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])["Delta"] * p["quantity"]) for p in book]
    fig_pie.add_trace(go.Pie(labels=labels, values=values_abs, hole=0.4), row=1, col=1)
    fig_pie.add_trace(go.Pie(labels=labels, values=deltas_abs, hole=0.4), row=1, col=2)
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, width="stretch")

with tab_var:
    st.subheader(f"Value at Risk ({var_confidence:.0%}, {var_horizon}d)")

    np.random.seed(0)
    S0 = book[0]["S"]
    avg_sigma = np.mean([p["sigma"] for p in book])
    dt = var_horizon / 252

    sim_returns = np.random.normal(0, avg_sigma * np.sqrt(dt), n_simulations)
    sim_spots = S0 * np.exp(sim_returns)

    pnl_dist = []
    for s in sim_spots:
        scenario_val = 0
        for p in book:
            new_T = max(p["T"] - dt, 0)
            scenario_val += p["quantity"] * bs_price(s, p["K"], new_T, p["r"], p["sigma"], p["option_type"])
        pnl_dist.append(scenario_val - total_value)

    pnl_dist = np.array(pnl_dist)
    var_value = np.percentile(pnl_dist, (1 - var_confidence) * 100)
    cvar_value = pnl_dist[pnl_dist <= var_value].mean() if np.any(pnl_dist <= var_value) else var_value

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"VaR ({var_confidence:.0%})", f"${var_value:.2f}")
    c2.metric(f"CVaR ({var_confidence:.0%})", f"${cvar_value:.2f}")
    c3.metric("Mean P&L", f"${pnl_dist.mean():.2f}")
    c4.metric("P&L Std", f"${pnl_dist.std():.2f}")

    fig_var = go.Figure()
    fig_var.add_trace(go.Histogram(
        x=pnl_dist, nbinsx=80, name="P&L Distribution",
        marker_color="#2962FF", opacity=0.7,
    ))
    fig_var.add_vline(x=var_value, line_color="red", line_width=2,
                      annotation_text=f"VaR = ${var_value:.2f}")
    fig_var.add_vline(x=cvar_value, line_color="darkred", line_width=2, line_dash="dash",
                      annotation_text=f"CVaR = ${cvar_value:.2f}")
    fig_var.add_vline(x=0, line_color="gray", line_dash="dot")
    fig_var.update_layout(
        title=f"P&L Distribution ({n_simulations:,} simulations, {var_horizon}d horizon)",
        xaxis_title="P&L", yaxis_title="Frequency", height=500,
    )
    st.plotly_chart(fig_var, width="stretch")

    st.subheader("P&L Percentiles")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pctl_vals = [np.percentile(pnl_dist, p) for p in percentiles]
    st.dataframe(
        pd.DataFrame({"Percentile": [f"{p}%" for p in percentiles], "P&L": [f"${v:.2f}" for v in pctl_vals]}).T,
        width="stretch",
    )

with tab_exposure:
    st.subheader("Greeks Exposure by Expiry")
    expiry_groups = {}
    for p in book:
        bucket = f"{p['T']*365:.0f}d"
        if bucket not in expiry_groups:
            expiry_groups[bucket] = {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0}
        g = all_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
        for k in expiry_groups[bucket]:
            expiry_groups[bucket][k] += g[k] * p["quantity"]

    buckets = list(expiry_groups.keys())
    fig_exp = make_subplots(rows=2, cols=2, subplot_titles=["Delta", "Gamma", "Vega", "Theta"])
    colors = ["#2962FF", "#00C853", "#FF6D00", "#D50000"]
    for idx, greek in enumerate(["Delta", "Gamma", "Vega", "Theta"]):
        r_pos, c_pos = divmod(idx, 2)
        vals = [expiry_groups[b][greek] for b in buckets]
        fig_exp.add_trace(go.Bar(
            x=buckets, y=vals, name=greek,
            marker_color=[colors[idx] if v >= 0 else "#FF8A80" for v in vals],
        ), row=r_pos + 1, col=c_pos + 1)
    fig_exp.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_exp, width="stretch")

    st.subheader("Greeks Exposure by Strike")
    strike_groups = {}
    for p in book:
        k_str = f"K={p['K']}"
        if k_str not in strike_groups:
            strike_groups[k_str] = {"Delta": 0, "Gamma": 0, "Vega": 0}
        g = all_greeks(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
        for gk in strike_groups[k_str]:
            strike_groups[k_str][gk] += g[gk] * p["quantity"]

    strikes_labels = list(strike_groups.keys())
    fig_stk = go.Figure()
    for greek, color in [("Delta", "#2962FF"), ("Gamma", "#00C853"), ("Vega", "#FF6D00")]:
        fig_stk.add_trace(go.Bar(
            x=strikes_labels,
            y=[strike_groups[s][greek] for s in strikes_labels],
            name=greek, marker_color=color,
        ))
    fig_stk.update_layout(barmode="group", height=400, yaxis_title="Exposure")
    st.plotly_chart(fig_stk, width="stretch")

with tab_alerts:
    st.subheader("Risk Alert Configuration")
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        max_delta = st.number_input("Max |Net Delta|", value=50.0, step=5.0, key="alert_delta")
        max_gamma = st.number_input("Max |Net Gamma|", value=5.0, step=0.5, key="alert_gamma")
        max_vega = st.number_input("Max |Net Vega|", value=10.0, step=1.0, key="alert_vega")
    with col_a2:
        max_var = st.number_input("Max VaR Loss", value=500.0, step=50.0, key="alert_var")
        max_theta = st.number_input("Max |Daily Theta|", value=20.0, step=1.0, key="alert_theta")

    alerts = []
    if abs(total_greeks["Delta"]) > max_delta:
        alerts.append(("Delta Breach", f"|Net Delta| = {abs(total_greeks['Delta']):.2f} > {max_delta}", "error"))
    if abs(total_greeks["Gamma"]) > max_gamma:
        alerts.append(("Gamma Breach", f"|Net Gamma| = {abs(total_greeks['Gamma']):.4f} > {max_gamma}", "error"))
    if abs(total_greeks["Vega"]) > max_vega:
        alerts.append(("Vega Breach", f"|Net Vega| = {abs(total_greeks['Vega']):.2f} > {max_vega}", "error"))
    if abs(total_greeks["Theta"]) > max_theta:
        alerts.append(("Theta Breach", f"|Net Theta| = {abs(total_greeks['Theta']):.2f} > {max_theta}", "warning"))
    if abs(var_value) > max_var:
        alerts.append(("VaR Breach", f"|VaR| = ${abs(var_value):.2f} > ${max_var}", "error"))

    if not alerts:
        st.success("All risk metrics within limits.")
    else:
        for title, msg, level in alerts:
            if level == "error":
                st.error(f"**{title}**: {msg}")
            else:
                st.warning(f"**{title}**: {msg}")

    st.subheader("Risk Summary Table")
    risk_df = pd.DataFrame([
        {"Metric": "Net Delta", "Value": f"{total_greeks['Delta']:.2f}", "Limit": f"{max_delta:.0f}",
         "Status": "BREACH" if abs(total_greeks['Delta']) > max_delta else "OK"},
        {"Metric": "Net Gamma", "Value": f"{total_greeks['Gamma']:.4f}", "Limit": f"{max_gamma:.1f}",
         "Status": "BREACH" if abs(total_greeks['Gamma']) > max_gamma else "OK"},
        {"Metric": "Net Vega", "Value": f"{total_greeks['Vega']:.2f}", "Limit": f"{max_vega:.1f}",
         "Status": "BREACH" if abs(total_greeks['Vega']) > max_vega else "OK"},
        {"Metric": "Net Theta", "Value": f"{total_greeks['Theta']:.2f}", "Limit": f"{max_theta:.1f}",
         "Status": "BREACH" if abs(total_greeks['Theta']) > max_theta else "OK"},
        {"Metric": f"VaR ({var_confidence:.0%})", "Value": f"${var_value:.2f}", "Limit": f"${max_var:.0f}",
         "Status": "BREACH" if abs(var_value) > max_var else "OK"},
    ])
    st.dataframe(risk_df, width="stretch")
