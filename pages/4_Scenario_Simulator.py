import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.scenarios import (
    spot_vol_scenario_grid, time_decay_scenario,
    spot_ladder, stress_test_table, portfolio_greeks, portfolio_value,
)

st.set_page_config(page_title="Scenario Simulator", layout="wide")
st.title("Scenario Simulator & Stress Testing")


def build_portfolio_ui():
    st.subheader("Portfolio Construction")
    if "positions" not in st.session_state:
        st.session_state["positions"] = []

    col_a, col_b = st.columns(2)
    with col_a:
        pos_S = st.number_input("Spot", value=100.0, step=1.0, key="sc_S")
        pos_K = st.number_input("Strike", value=100.0, step=1.0, key="sc_K")
        pos_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="sc_T")
    with col_b:
        pos_r = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="sc_r")
        pos_sigma = st.number_input("Volatility", value=0.20, step=0.01, format="%.4f", key="sc_sigma")
        pos_type = st.selectbox("Type", ["call", "put"], key="sc_type")
        pos_qty = st.number_input("Quantity", value=1, step=1, key="sc_qty")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Add Position"):
            st.session_state["positions"].append({
                "S": pos_S, "K": pos_K, "T": pos_T, "r": pos_r,
                "sigma": pos_sigma, "option_type": pos_type, "quantity": pos_qty,
            })
    with c2:
        if st.button("Clear All"):
            st.session_state["positions"] = []

    if not st.session_state["positions"]:
        st.info("Add positions to the portfolio, or use the quick-add below.")
        if st.button("Load Sample Portfolio"):
            st.session_state["positions"] = [
                {"S": 100, "K": 100, "T": 0.25, "r": 0.05, "sigma": 0.20, "option_type": "call", "quantity": 10},
                {"S": 100, "K": 95, "T": 0.25, "r": 0.05, "sigma": 0.22, "option_type": "put", "quantity": -5},
                {"S": 100, "K": 110, "T": 0.5, "r": 0.05, "sigma": 0.18, "option_type": "call", "quantity": -8},
                {"S": 100, "K": 90, "T": 0.5, "r": 0.05, "sigma": 0.25, "option_type": "put", "quantity": 3},
            ]
            st.rerun()
        return []

    st.dataframe(pd.DataFrame(st.session_state["positions"]), use_container_width=True)
    return st.session_state["positions"]


positions = build_portfolio_ui()

if positions:
    st.divider()

    tab_heat, tab_ladder, tab_decay, tab_stress = st.tabs([
        "P&L Heatmap", "Spot Ladder", "Time Decay", "Stress Test"
    ])

    with tab_heat:
        st.subheader("Spot × Vol P&L Heatmap")
        col_h1, col_h2 = st.columns([1, 3])
        with col_h1:
            spot_pct = st.slider("Spot Range (%)", 5, 50, 20, key="hm_sr")
            vol_pct = st.slider("Vol Range (pp)", 1, 30, 10, key="hm_vr")
            n_grid = st.slider("Grid Size", 11, 41, 21, step=2, key="hm_n")
        with col_h2:
            sb, vb, pnl = spot_vol_scenario_grid(
                positions,
                spot_range=(-spot_pct / 100, spot_pct / 100),
                vol_range=(-vol_pct / 100, vol_pct / 100),
                n_spot=n_grid, n_vol=n_grid,
            )
            fig = go.Figure(data=go.Heatmap(
                x=np.round(sb, 1), y=np.round(vb, 1), z=pnl,
                colorscale="RdYlGn", zmid=0,
                colorbar=dict(title="P&L"),
                text=np.round(pnl, 2), texttemplate="%{text}",
            ))
            fig.update_layout(
                title="Portfolio P&L: Spot Bump (%) vs Vol Bump (pp)",
                xaxis_title="Spot Change (%)", yaxis_title="Vol Change (pp)",
                height=600,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_ladder:
        st.subheader("Spot Ladder")
        lad_range = st.slider("Range (%)", 5, 50, 20, key="lad_r")
        df_ladder = spot_ladder(positions, pct_range=lad_range / 100, n_points=51)

        fig_lad = make_subplots(rows=2, cols=2,
                                subplot_titles=["P&L vs Spot", "Delta vs Spot", "Gamma vs Spot", "Vega vs Spot"],
                                vertical_spacing=0.12, horizontal_spacing=0.08)
        fig_lad.add_trace(go.Scatter(x=df_ladder["spot"], y=df_ladder["pnl"], name="P&L",
                                     fill="tozeroy", line=dict(color="#2962FF")), row=1, col=1)
        fig_lad.add_trace(go.Scatter(x=df_ladder["spot"], y=df_ladder["Delta"], name="Delta",
                                     line=dict(color="#FF6D00")), row=1, col=2)
        fig_lad.add_trace(go.Scatter(x=df_ladder["spot"], y=df_ladder["Gamma"], name="Gamma",
                                     line=dict(color="#AA00FF")), row=2, col=1)
        fig_lad.add_trace(go.Scatter(x=df_ladder["spot"], y=df_ladder["Vega"], name="Vega",
                                     line=dict(color="#00BFA5")), row=2, col=2)
        fig_lad.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig_lad, use_container_width=True)

    with tab_decay:
        st.subheader("Theta Decay Analysis")
        decay_days = st.slider("Days Forward", 5, 90, 30, key="decay_d")
        df_decay = time_decay_scenario(positions, days_forward=decay_days)

        fig_dec = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=["Portfolio Value Over Time", "Greeks Over Time"],
                                vertical_spacing=0.1)
        fig_dec.add_trace(go.Scatter(x=df_decay["day"], y=df_decay["value"], name="Value",
                                     fill="tozeroy", line=dict(color="#2962FF")), row=1, col=1)
        fig_dec.add_trace(go.Scatter(x=df_decay["day"], y=df_decay["Delta"], name="Delta",
                                     line=dict(color="#FF6D00")), row=2, col=1)
        fig_dec.add_trace(go.Scatter(x=df_decay["day"], y=df_decay["Gamma"], name="Gamma",
                                     line=dict(color="#AA00FF")), row=2, col=1)
        fig_dec.update_layout(height=600, xaxis2_title="Days")
        st.plotly_chart(fig_dec, use_container_width=True)

    with tab_stress:
        st.subheader("Stress Test Results")
        df_stress = stress_test_table(positions)
        st.dataframe(
            df_stress.style.format({
                "Portfolio Value": "${:.2f}",
                "P&L": "${:.2f}",
                "P&L %": "{:.1f}%",
                "Delta": "{:.4f}",
                "Gamma": "{:.6f}",
                "Vega": "{:.4f}",
            }).map(
                lambda v: "background-color: #c8e6c9" if isinstance(v, (int, float)) and v > 0
                else ("background-color: #ffcdd2" if isinstance(v, (int, float)) and v < 0 else ""),
                subset=["P&L"],
            ),
            use_container_width=True,
        )

        base_greeks = portfolio_greeks(positions)
        base_val = portfolio_value(positions)
        st.subheader("Current Portfolio Summary")
        cols = st.columns(5)
        cols[0].metric("Portfolio Value", f"${base_val:.2f}")
        cols[1].metric("Delta", f"{base_greeks['Delta']:.4f}")
        cols[2].metric("Gamma", f"{base_greeks['Gamma']:.6f}")
        cols[3].metric("Vega", f"{base_greeks['Vega']:.4f}")
        cols[4].metric("Theta", f"{base_greeks['Theta']:.4f}")
