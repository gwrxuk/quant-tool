import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.backtesting import (
    backtest_delta_hedge, backtest_delta_hedge_heston, backtest_strategy,
    backtest_strategy_from_data, simulate_gbm, STRATEGY_TEMPLATES,
)
from core.pricing import bs_price
from core.market_data import POPULAR_TICKERS, get_live_params

st.set_page_config(page_title="Strategy Backtester", layout="wide")
st.title("Options Strategy Backtester")

# ── Live Data Sidebar ────────────────────────────────────────────────────
st.sidebar.header("Live Market Data")
_bt_ticker = st.sidebar.selectbox("Ticker", POPULAR_TICKERS, index=0, key="bt_ticker")
_bt_prev = st.session_state.get("bt_loaded_ticker")
if st.sidebar.button("Load Live Data", key="bt_load") or (
    _bt_prev is not None and _bt_prev != _bt_ticker
):
    with st.sidebar:
        with st.spinner("Fetching…"):
            _bt_lp = get_live_params(_bt_ticker)
            st.session_state["bt_params"] = _bt_lp
            st.session_state["bt_loaded_ticker"] = _bt_ticker
            for _k in ("dh_S", "dh_K", "dh_r", "dh_sigma",
                        "strat_S", "strat_r", "strat_sigma",
                        "hist_r", "hist_sigma"):
                st.session_state.pop(_k, None)
_bp = st.session_state.get("bt_params", {})
_def_S = _bp.get("spot", 100.0)
_def_sigma = _bp.get("sigma", 0.20)
_def_r = _bp.get("r", 0.05)
if _bp:
    st.sidebar.caption(f"**{_bp.get('name', _bt_ticker)}** — ${_def_S:.2f}")
st.sidebar.divider()

tab_dh, tab_strat, tab_hist = st.tabs(["Delta Hedging", "Multi-Leg Strategies", "Historical Data"])

# ---------- Delta Hedging ----------
with tab_dh:
    col_d1, col_d2 = st.columns([1, 3])
    with col_d1:
        st.subheader("Parameters")
        dh_S = st.number_input("Spot", value=_def_S, step=1.0, key="dh_S")
        dh_K = st.number_input("Strike", value=float(round(_def_S)), step=1.0, key="dh_K")
        dh_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="dh_T")
        dh_r = st.number_input("Rate", value=_def_r, step=0.005, format="%.4f", key="dh_r")
        dh_sigma = st.number_input("Volatility", value=_def_sigma, step=0.01, format="%.4f", key="dh_sigma")
        dh_type = st.selectbox("Option Type", ["call", "put"], key="dh_type")
        dh_freq = st.selectbox("Hedge Frequency (days)", [1, 2, 5, 10, 20], key="dh_freq")
        dh_seed = st.number_input("Seed", value=42, step=1, key="dh_seed")

        st.subheader("Simulation Model")
        dh_model = st.selectbox("Model", ["GBM", "Heston"], key="dh_model")
        if dh_model == "Heston":
            dh_v0 = st.number_input("Initial Var (v₀)", value=0.04, step=0.005, format="%.4f", key="dh_v0")
            dh_kappa = st.number_input("Mean Rev (κ)", value=2.0, step=0.1, key="dh_kappa")
            dh_theta_v = st.number_input("Long Var (θ)", value=0.04, step=0.005, format="%.4f", key="dh_theta_v")
            dh_sigma_v = st.number_input("Vol of Vol (σᵥ)", value=0.3, step=0.05, key="dh_sigma_v")
            dh_rho_h = st.number_input("Correlation (ρ)", value=-0.7, step=0.05, min_value=-0.99, max_value=0.99, key="dh_rho_h")

    with col_d2:
        if st.button("Run Delta Hedge Backtest", key="dh_run"):
            n_steps = int(dh_T * 252)
            if dh_model == "Heston":
                df = backtest_delta_hedge_heston(
                    dh_S, dh_K, dh_T, dh_r, dh_v0, dh_kappa, dh_theta_v,
                    dh_sigma_v, dh_rho_h, dh_type, n_steps, dh_freq,
                    hedge_vol=dh_sigma, seed=int(dh_seed),
                )
            else:
                df = backtest_delta_hedge(dh_S, dh_K, dh_T, dh_r, dh_sigma,
                                          dh_type, n_steps, dh_freq, int(dh_seed))

            st.subheader("Results")
            c1, c2, c3 = st.columns(3)
            c1.metric("Final Cumulative P&L", f"${df['cumulative_pnl'].iloc[-1]:.4f}")
            c2.metric("Max Drawdown", f"${df['cumulative_pnl'].min():.4f}")
            c3.metric("P&L Std (daily)", f"${df['daily_pnl'].std():.4f}")

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=["Spot Price", "Cumulative P&L", "Delta Position"],
                                vertical_spacing=0.06)
            fig.add_trace(go.Scatter(x=df["day"], y=df["spot"], name="Spot",
                                     line=dict(color="#2962FF")), row=1, col=1)
            fig.add_hline(y=dh_K, line_dash="dash", line_color="gray", row=1, col=1)
            fig.add_trace(go.Scatter(x=df["day"], y=df["cumulative_pnl"], name="Cum P&L",
                                     fill="tozeroy", line=dict(color="#00C853")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["day"], y=df["delta"], name="Delta",
                                     line=dict(color="#FF6D00")), row=3, col=1)

            if dh_model == "Heston" and "realized_vol" in df.columns:
                fig.add_trace(go.Scatter(x=df["day"], y=df["realized_vol"], name="Realized Vol",
                                         line=dict(color="#AA00FF", dash="dot")), row=1, col=1)

            fig.update_layout(height=800, showlegend=True)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Daily P&L Components")
            fig2 = go.Figure()
            for col_name, color in [("hedge_pnl", "#2962FF"), ("option_pnl", "#D50000"), ("interest", "#00C853")]:
                fig2.add_trace(go.Bar(x=df["day"], y=df[col_name], name=col_name.replace("_", " ").title(),
                                      marker_color=color, opacity=0.7))
            fig2.update_layout(barmode="relative", height=400, xaxis_title="Day", yaxis_title="P&L")
            st.plotly_chart(fig2, width="stretch")

# ---------- Multi-Leg Strategies ----------
with tab_strat:
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        st.subheader("Strategy Selection")
        strat_S = st.number_input("Spot", value=_def_S, step=1.0, key="strat_S")
        strat_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="strat_T")
        strat_r = st.number_input("Rate", value=_def_r, step=0.005, format="%.4f", key="strat_r")
        strat_sigma = st.number_input("Volatility", value=_def_sigma, step=0.01, format="%.4f", key="strat_sigma")
        strat_name = st.selectbox("Strategy", list(STRATEGY_TEMPLATES.keys()), key="strat_name")
        strat_seed = st.number_input("Seed", value=42, step=1, key="strat_seed")

        if strat_name in ["Long Strangle", "Bull Call Spread", "Bear Put Spread", "Long Butterfly"]:
            strat_width = st.number_input("Width", value=5.0, step=1.0, key="strat_w")
        elif strat_name == "Iron Condor":
            strat_inner = st.number_input("Inner Width", value=3.0, step=1.0, key="strat_inner")
            strat_outer = st.number_input("Outer Width", value=7.0, step=1.0, key="strat_outer")
        elif strat_name == "Calendar Spread":
            strat_T_near = st.number_input("Near Expiry (years)", value=1/12, step=0.01, min_value=0.01,
                                           format="%.4f", key="strat_T_near")
            strat_T_far = st.number_input("Far Expiry (years)", value=0.25, step=0.01, min_value=0.02,
                                          format="%.4f", key="strat_T_far")

        st.subheader("Simulation Model")
        strat_model = st.selectbox("Model", ["GBM", "Heston"], key="strat_model")
        strat_heston_params = None
        if strat_model == "Heston":
            st_v0 = st.number_input("v₀", value=0.04, step=0.005, format="%.4f", key="st_v0")
            st_kappa = st.number_input("κ", value=2.0, step=0.1, key="st_kappa")
            st_theta_v = st.number_input("θ", value=0.04, step=0.005, format="%.4f", key="st_theta_v")
            st_sigma_v = st.number_input("σᵥ", value=0.3, step=0.05, key="st_sigma_v")
            st_rho_h = st.number_input("ρ", value=-0.7, step=0.05, min_value=-0.99, max_value=0.99, key="st_rho_h")
            strat_heston_params = {"v0": st_v0, "kappa": st_kappa, "theta": st_theta_v,
                                   "sigma_v": st_sigma_v, "rho": st_rho_h}

    with col_s2:
        if st.button("Run Strategy Backtest", key="strat_run"):
            if strat_name in ["Long Strangle", "Bull Call Spread", "Bear Put Spread", "Long Butterfly"]:
                positions = STRATEGY_TEMPLATES[strat_name](strat_S, strat_width)
            elif strat_name == "Iron Condor":
                positions = STRATEGY_TEMPLATES[strat_name](strat_S, strat_inner, strat_outer)
            elif strat_name == "Calendar Spread":
                positions = STRATEGY_TEMPLATES[strat_name](strat_S, T_near=strat_T_near, T_far=strat_T_far)
            else:
                positions = STRATEGY_TEMPLATES[strat_name](strat_S)

            n_steps = int(strat_T * 252)
            df, cost, final_pnl = backtest_strategy(
                strat_S, positions, strat_T, strat_r, strat_sigma, n_steps, int(strat_seed),
                model=strat_model.lower(), heston_params=strat_heston_params,
            )

            st.subheader("Strategy Legs")
            legs_data = []
            for p in positions:
                leg_T = p.get("T", strat_T)
                leg_sigma = p.get("sigma", strat_sigma)
                legs_data.append({
                    "K": p["K"], "Type": p["option_type"], "Qty": p["quantity"],
                    "T": f"{leg_T:.4f}", "σ": f"{leg_sigma:.2%}",
                    "Price": f"${bs_price(strat_S, p['K'], leg_T, strat_r, leg_sigma, p['option_type']):.4f}",
                })
            st.dataframe(pd.DataFrame(legs_data), width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Initial Cost", f"${cost:.4f}")
            c2.metric("Terminal Payoff P&L", f"${final_pnl:.4f}")
            c3.metric("Max Unrealized P&L", f"${df['pnl'].max():.4f}")

            fig = make_subplots(rows=2, cols=2, shared_xaxes=True,
                                subplot_titles=["Spot", "Portfolio P&L", "Delta", "Gamma & Vega"],
                                vertical_spacing=0.1, horizontal_spacing=0.08)
            fig.add_trace(go.Scatter(x=df["day"], y=df["spot"], name="Spot",
                                     line=dict(color="#2962FF")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df["day"], y=df["pnl"], name="P&L",
                                     fill="tozeroy", line=dict(color="#00C853")), row=1, col=2)
            fig.add_trace(go.Scatter(x=df["day"], y=df["delta"], name="Delta",
                                     line=dict(color="#FF6D00")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df["day"], y=df["gamma"], name="Gamma",
                                     line=dict(color="#AA00FF")), row=2, col=2)
            fig.add_trace(go.Scatter(x=df["day"], y=df["vega"], name="Vega",
                                     line=dict(color="#00BFA5")), row=2, col=2)
            fig.update_layout(height=700)
            st.plotly_chart(fig, width="stretch")

            st.subheader("Payoff at Expiry")
            T_max = max(p.get("T", strat_T) for p in positions)
            payoff_spots = np.linspace(strat_S * 0.7, strat_S * 1.3, 200)
            payoffs = []
            for s in payoff_spots:
                pf = 0
                for p in positions:
                    leg_T = p.get("T", strat_T)
                    if leg_T <= T_max + 1e-10:
                        if p["option_type"] == "call":
                            pf += p["quantity"] * max(s - p["K"], 0)
                        else:
                            pf += p["quantity"] * max(p["K"] - s, 0)
                    else:
                        leg_sigma = p.get("sigma", strat_sigma)
                        t_remain = leg_T - T_max
                        pf += p["quantity"] * bs_price(s, p["K"], t_remain, strat_r, leg_sigma, p["option_type"])
                payoffs.append(pf - cost)

            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=payoff_spots, y=payoffs, name="Net P&L",
                                            line=dict(color="#2962FF", width=2)))
            fig_payoff.add_hline(y=0, line_color="gray", line_dash="dash")
            fig_payoff.add_vline(x=strat_S, line_color="gray", line_dash="dot")
            fig_payoff.update_layout(xaxis_title="Spot at Expiry", yaxis_title="P&L", height=400)
            st.plotly_chart(fig_payoff, width="stretch")

# ---------- Historical Data ----------
with tab_hist:
    st.subheader("Backtest on Historical Price Data")
    st.markdown("""
    Upload a CSV file with historical spot prices. The file should have a column
    named `close` or `price` (case-insensitive), or the first numeric column will be used.
    """)

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="hist_csv")

    col_h1, col_h2 = st.columns([1, 3])
    with col_h1:
        hist_r = st.number_input("Rate", value=_def_r, step=0.005, format="%.4f", key="hist_r")
        hist_sigma = st.number_input("Volatility", value=_def_sigma, step=0.01, format="%.4f", key="hist_sigma")
        hist_strat = st.selectbox("Strategy", list(STRATEGY_TEMPLATES.keys()), key="hist_strat")

    with col_h2:
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
            st.write(f"Loaded {len(raw)} rows. Columns: {list(raw.columns)}")

            price_col = None
            for c in raw.columns:
                if c.lower() in ("close", "price", "adj close", "adj_close"):
                    price_col = c
                    break
            if price_col is None:
                for c in raw.columns:
                    if pd.api.types.is_numeric_dtype(raw[c]):
                        price_col = c
                        break

            if price_col is None:
                st.error("No numeric price column found.")
            else:
                spot_data = raw[price_col].dropna().values
                S0_hist = spot_data[0]
                st.line_chart(pd.DataFrame({"Spot": spot_data}))

                if st.button("Run Historical Backtest", key="hist_run"):
                    if hist_strat in ["Long Strangle", "Bull Call Spread", "Bear Put Spread", "Long Butterfly"]:
                        positions = STRATEGY_TEMPLATES[hist_strat](S0_hist, 5)
                    elif hist_strat == "Iron Condor":
                        positions = STRATEGY_TEMPLATES[hist_strat](S0_hist, 3, 7)
                    elif hist_strat == "Calendar Spread":
                        positions = STRATEGY_TEMPLATES[hist_strat](S0_hist)
                    else:
                        positions = STRATEGY_TEMPLATES[hist_strat](S0_hist)

                    df, cost, final_pnl = backtest_strategy_from_data(
                        spot_data, positions, r=hist_r, sigma=hist_sigma,
                    )

                    if not df.empty:
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Initial Cost", f"${cost:.4f}")
                        c2.metric("Terminal P&L", f"${final_pnl:.4f}")
                        c3.metric("Max P&L", f"${df['pnl'].max():.4f}")

                        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                            subplot_titles=["Spot", "Strategy P&L"],
                                            vertical_spacing=0.08)
                        fig.add_trace(go.Scatter(x=df["day"], y=df["spot"], name="Spot",
                                                 line=dict(color="#2962FF")), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df["day"], y=df["pnl"], name="P&L",
                                                 fill="tozeroy", line=dict(color="#00C853")), row=2, col=1)
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, width="stretch")
