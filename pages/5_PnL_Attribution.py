import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.pnl import (daily_pnl_attribution, generate_sample_pnl_data,
                       pnl_summary, portfolio_pnl_attribution)

st.set_page_config(page_title="P&L Attribution", layout="wide")
st.title("P&L Attribution & Greeks Decomposition")

tab_single, tab_portfolio = st.tabs(["Single Position", "Portfolio Attribution"])

# ---------- Single Position ----------
with tab_single:
    col1, col2 = st.columns([1, 3])

    with col1:
        st.subheader("Position Parameters")
        pnl_S = st.number_input("Initial Spot", value=100.0, step=1.0, key="pnl_S")
        pnl_K = st.number_input("Strike", value=100.0, step=1.0, key="pnl_K")
        pnl_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="pnl_T")
        pnl_r = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="pnl_r")
        pnl_sigma = st.number_input("Initial Vol", value=0.20, step=0.01, format="%.4f", key="pnl_sigma")
        pnl_type = st.selectbox("Type", ["call", "put"], key="pnl_type")
        pnl_qty = st.number_input("Quantity", value=10, step=1, key="pnl_qty")
        pnl_days = st.slider("Simulation Days", 10, 120, 60, key="pnl_days")
        pnl_seed = st.number_input("Seed", value=42, step=1, key="pnl_seed")

        st.subheader("Simulation Settings")
        spot_vol = st.slider("Daily Spot Vol", 0.005, 0.05, 0.015, 0.001, key="pnl_sv")
        iv_drift = st.slider("IV Drift Vol", 0.0005, 0.01, 0.002, 0.0005, key="pnl_ivd")

    with col2:
        if st.button("Run P&L Attribution", key="pnl_run") or True:
            S_series, sigma_series = generate_sample_pnl_data(
                S0=pnl_S, K=pnl_K, T=pnl_T, r=pnl_r, sigma=pnl_sigma,
                drift_vol=iv_drift, spot_vol=spot_vol, n_days=pnl_days, seed=int(pnl_seed),
            )

            df = daily_pnl_attribution(
                S_series, sigma_series, pnl_K, pnl_T, pnl_r, pnl_type, pnl_qty,
            )

            if df.empty:
                st.warning("No data generated.")
            else:
                st.subheader("Cumulative P&L Decomposition")
                fig_cum = go.Figure()
                components = [
                    ("cumulative_delta", "Delta P&L", "rgba(41,98,255,0.5)"),
                    ("cumulative_gamma", "Gamma P&L", "rgba(0,200,83,0.5)"),
                    ("cumulative_vega", "Vega P&L", "rgba(255,109,0,0.5)"),
                    ("cumulative_theta", "Theta P&L", "rgba(213,0,0,0.5)"),
                    ("cumulative_vanna", "Vanna P&L", "rgba(170,0,255,0.5)"),
                    ("cumulative_volga", "Volga P&L", "rgba(0,191,165,0.5)"),
                    ("cumulative_unexplained", "Unexplained", "rgba(153,153,153,0.5)"),
                ]
                for col_name, label, color in components:
                    if col_name in df.columns:
                        fig_cum.add_trace(go.Scatter(
                            x=df["day"], y=df[col_name], name=label,
                            stackgroup="one", line=dict(width=0.5),
                            fillcolor=color,
                        ))
                fig_cum.add_trace(go.Scatter(
                    x=df["day"], y=df["cumulative_actual"], name="Actual P&L",
                    line=dict(color="black", width=2.5),
                ))
                fig_cum.update_layout(height=500, yaxis_title="Cumulative P&L", xaxis_title="Day")
                st.plotly_chart(fig_cum, use_container_width=True)

                st.subheader("Daily P&L Breakdown")
                fig_daily = go.Figure()
                daily_components = [
                    ("delta_pnl", "Delta", "#2962FF"),
                    ("gamma_pnl", "Gamma", "#00C853"),
                    ("vega_pnl", "Vega", "#FF6D00"),
                    ("theta_pnl", "Theta", "#D50000"),
                    ("vanna_pnl", "Vanna", "#AA00FF"),
                    ("volga_pnl", "Volga", "#00BFA5"),
                    ("unexplained_pnl", "Unexplained", "#999999"),
                ]
                for col_name, label, color in daily_components:
                    if col_name in df.columns:
                        fig_daily.add_trace(go.Bar(
                            x=df["day"], y=df[col_name], name=label,
                            marker_color=color, opacity=0.8,
                        ))
                fig_daily.add_trace(go.Scatter(
                    x=df["day"], y=df["actual_pnl"], name="Actual",
                    line=dict(color="black", width=2), mode="lines",
                ))
                fig_daily.update_layout(barmode="relative", height=400,
                                        xaxis_title="Day", yaxis_title="Daily P&L")
                st.plotly_chart(fig_daily, use_container_width=True)

                st.subheader("Spot & Implied Vol Paths")
                fig_paths = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                          subplot_titles=["Spot Price", "Implied Volatility"],
                                          vertical_spacing=0.08)
                fig_paths.add_trace(go.Scatter(
                    x=df["day"], y=df["spot"], name="Spot",
                    line=dict(color="#2962FF"),
                ), row=1, col=1)
                fig_paths.add_hline(y=pnl_K, line_dash="dash", line_color="gray", row=1, col=1)
                fig_paths.add_trace(go.Scatter(
                    x=df["day"], y=df["implied_vol"] * 100, name="IV (%)",
                    line=dict(color="#FF6D00"),
                ), row=2, col=1)
                fig_paths.update_layout(height=450)
                st.plotly_chart(fig_paths, use_container_width=True)

                st.subheader("Greeks Over Time")
                fig_greeks = make_subplots(rows=2, cols=2,
                                           subplot_titles=["Delta", "Gamma", "Vega", "Theta"],
                                           vertical_spacing=0.12, horizontal_spacing=0.08)
                fig_greeks.add_trace(go.Scatter(x=df["day"], y=df["delta"], line=dict(color="#2962FF")), row=1, col=1)
                fig_greeks.add_trace(go.Scatter(x=df["day"], y=df["gamma"], line=dict(color="#00C853")), row=1, col=2)
                fig_greeks.add_trace(go.Scatter(x=df["day"], y=df["vega"], line=dict(color="#FF6D00")), row=2, col=1)
                fig_greeks.add_trace(go.Scatter(x=df["day"], y=df["theta"], line=dict(color="#D50000")), row=2, col=2)
                fig_greeks.update_layout(height=500, showlegend=False)
                st.plotly_chart(fig_greeks, use_container_width=True)

                st.subheader("P&L Summary Statistics")
                summary = pnl_summary(df)
                st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)

# ---------- Portfolio Attribution ----------
with tab_portfolio:
    st.subheader("Portfolio P&L Attribution")
    st.markdown("Attribute P&L across a multi-position portfolio sharing the same underlying.")

    if "port_positions" not in st.session_state:
        st.session_state["port_positions"] = []

    col_pa, col_pb = st.columns(2)
    with col_pa:
        pp_K = st.number_input("Strike", value=100.0, step=1.0, key="pp_K")
        pp_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="pp_T")
        pp_sigma = st.number_input("Implied Vol", value=0.20, step=0.01, format="%.4f", key="pp_sigma")
    with col_pb:
        pp_type = st.selectbox("Type", ["call", "put"], key="pp_type")
        pp_qty = st.number_input("Quantity", value=10, step=1, key="pp_qty")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Add Position", key="pp_add"):
            st.session_state["port_positions"].append({
                "K": pp_K, "T": pp_T, "sigma": pp_sigma,
                "option_type": pp_type, "quantity": pp_qty,
            })
    with c2:
        if st.button("Clear All", key="pp_clear"):
            st.session_state["port_positions"] = []
    with c3:
        if st.button("Load Sample Portfolio", key="pp_sample"):
            st.session_state["port_positions"] = [
                {"K": 100, "T": 0.25, "sigma": 0.20, "option_type": "call", "quantity": 10},
                {"K": 95, "T": 0.25, "sigma": 0.22, "option_type": "put", "quantity": -5},
                {"K": 110, "T": 0.5, "sigma": 0.18, "option_type": "call", "quantity": -8},
                {"K": 90, "T": 0.5, "sigma": 0.25, "option_type": "put", "quantity": 3},
            ]
            st.rerun()

    positions = st.session_state["port_positions"]
    if positions:
        st.dataframe(pd.DataFrame(positions), use_container_width=True)

        col_ps1, col_ps2 = st.columns([1, 3])
        with col_ps1:
            pp_S0 = st.number_input("Spot", value=100.0, step=1.0, key="pp_S0")
            pp_days = st.slider("Days", 10, 120, 60, key="pp_days")
            pp_r = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="pp_r")
            pp_spot_vol = st.slider("Spot Vol", 0.005, 0.05, 0.015, 0.001, key="pp_sv")
            pp_iv_drift = st.slider("IV Drift", 0.0005, 0.01, 0.002, 0.0005, key="pp_ivd")
            pp_seed = st.number_input("Seed", value=42, step=1, key="pp_seed")

        with col_ps2:
            if st.button("Run Portfolio Attribution", key="pp_run"):
                agg, per_pos = portfolio_pnl_attribution(
                    positions, S0=pp_S0, n_days=pp_days,
                    spot_vol=pp_spot_vol, iv_drift=pp_iv_drift, r=pp_r, seed=int(pp_seed),
                )

                if agg.empty:
                    st.warning("No data.")
                else:
                    st.subheader("Aggregate Cumulative P&L")
                    fig_agg = go.Figure()
                    components = [
                        ("cumulative_delta", "Delta", "rgba(41,98,255,0.5)"),
                        ("cumulative_gamma", "Gamma", "rgba(0,200,83,0.5)"),
                        ("cumulative_vega", "Vega", "rgba(255,109,0,0.5)"),
                        ("cumulative_theta", "Theta", "rgba(213,0,0,0.5)"),
                        ("cumulative_vanna", "Vanna", "rgba(170,0,255,0.5)"),
                        ("cumulative_volga", "Volga", "rgba(0,191,165,0.5)"),
                        ("cumulative_unexplained", "Unexplained", "rgba(153,153,153,0.5)"),
                    ]
                    for col_name, label, color in components:
                        if col_name in agg.columns:
                            fig_agg.add_trace(go.Scatter(
                                x=agg["day"], y=agg[col_name], name=label,
                                stackgroup="one", line=dict(width=0.5),
                                fillcolor=color,
                            ))
                    fig_agg.add_trace(go.Scatter(
                        x=agg["day"], y=agg["cumulative_actual"], name="Actual",
                        line=dict(color="black", width=2.5),
                    ))
                    fig_agg.update_layout(height=500, yaxis_title="Cumulative P&L", xaxis_title="Day")
                    st.plotly_chart(fig_agg, use_container_width=True)

                    st.subheader("Per-Position Breakdown")
                    for i, pdf in enumerate(per_pos):
                        if pdf.empty:
                            continue
                        label = pdf["label"].iloc[0] if "label" in pdf.columns else f"Position {i+1}"
                        with st.expander(f"{label} — Total P&L: ${pdf['actual_pnl'].sum():.2f}"):
                            fig_pos = go.Figure()
                            for col_name, clabel, color in [
                                ("cumulative_delta", "Delta", "#2962FF"),
                                ("cumulative_gamma", "Gamma", "#00C853"),
                                ("cumulative_vega", "Vega", "#FF6D00"),
                                ("cumulative_theta", "Theta", "#D50000"),
                            ]:
                                if col_name in pdf.columns:
                                    fig_pos.add_trace(go.Scatter(
                                        x=pdf["day"], y=pdf[col_name], name=clabel,
                                        stackgroup="one",
                                    ))
                            if "cumulative_actual" in pdf.columns:
                                fig_pos.add_trace(go.Scatter(
                                    x=pdf["day"], y=pdf["cumulative_actual"], name="Actual",
                                    line=dict(color="black", width=2),
                                ))
                            fig_pos.update_layout(height=350)
                            st.plotly_chart(fig_pos, use_container_width=True)

                    st.subheader("Portfolio P&L Summary")
                    summary = pnl_summary(agg)
                    if not isinstance(summary, dict) or isinstance(summary, pd.DataFrame):
                        st.dataframe(summary.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("Add positions above or load the sample portfolio.")
