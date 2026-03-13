import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.pricing import bs_price, heston_price, implied_vol
from core.greeks import all_greeks

st.set_page_config(page_title="Options Pricer", layout="wide")
st.title("Options Pricer & Greeks Calculator")

tab_bs, tab_heston, tab_profiles = st.tabs(["Black-Scholes", "Heston Model", "Greeks Profiles"])

# ---------- Black-Scholes Tab ----------
with tab_bs:
    col_input, col_result = st.columns([1, 2])
    with col_input:
        st.subheader("Parameters")
        S = st.number_input("Spot Price (S)", value=100.0, step=1.0, key="bs_S")
        K = st.number_input("Strike (K)", value=100.0, step=1.0, key="bs_K")
        T = st.number_input("Time to Expiry (years)", value=0.25, step=0.01, min_value=0.001, key="bs_T")
        r = st.number_input("Risk-Free Rate", value=0.05, step=0.005, format="%.4f", key="bs_r")
        sigma = st.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.4f", key="bs_sigma")
        option_type = st.selectbox("Option Type", ["call", "put"], key="bs_type")

    with col_result:
        price = bs_price(S, K, T, r, sigma, option_type)
        greeks = all_greeks(S, K, T, r, sigma, option_type)

        st.subheader("Price")
        st.metric(f"{option_type.title()} Price", f"${price:.4f}")

        st.subheader("Greeks")
        g_cols = st.columns(3)
        greek_items = list(greeks.items())
        for i, (name, val) in enumerate(greek_items):
            with g_cols[i % 3]:
                st.metric(name, f"{val:.6f}")

        st.subheader("Implied Volatility Calculator")
        market_price = st.number_input("Market Price", value=round(price, 2), step=0.01, key="iv_price")
        iv = implied_vol(market_price, S, K, T, r, option_type)
        if np.isnan(iv):
            st.warning("Could not compute implied vol for given price.")
        else:
            st.metric("Implied Volatility", f"{iv:.4%}")

# ---------- Heston Tab ----------
with tab_heston:
    st.subheader("Heston Stochastic Volatility Model")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        h_S = st.number_input("Spot (S)", value=100.0, step=1.0, key="h_S")
        h_K = st.number_input("Strike (K)", value=100.0, step=1.0, key="h_K")
        h_T = st.number_input("Expiry (years)", value=0.5, step=0.05, min_value=0.01, key="h_T")
        h_r = st.number_input("Rate (r)", value=0.05, step=0.005, format="%.4f", key="h_r")
        h_type = st.selectbox("Type", ["call", "put"], key="h_type")
    with col_h2:
        h_v0 = st.number_input("Initial Variance (v₀)", value=0.04, step=0.005, format="%.4f", key="h_v0")
        h_kappa = st.number_input("Mean Reversion (κ)", value=2.0, step=0.1, key="h_kappa")
        h_theta = st.number_input("Long-Run Variance (θ)", value=0.04, step=0.005, format="%.4f", key="h_theta")
        h_sigma = st.number_input("Vol of Vol (σᵥ)", value=0.3, step=0.05, key="h_sigma_v")
        h_rho = st.number_input("Correlation (ρ)", value=-0.7, step=0.05, min_value=-0.99, max_value=0.99, key="h_rho")

    if st.button("Price with Heston", key="heston_btn"):
        with st.spinner("Computing Heston price via characteristic function integration..."):
            h_price = heston_price(h_S, h_K, h_T, h_r, h_v0, h_kappa, h_theta, h_sigma, h_rho, h_type)
            bs_ref = bs_price(h_S, h_K, h_T, h_r, np.sqrt(h_v0), h_type)
            h_iv = implied_vol(h_price, h_S, h_K, h_T, h_r, h_type)

        c1, c2, c3 = st.columns(3)
        c1.metric("Heston Price", f"${h_price:.4f}")
        c2.metric("BS Price (σ=√v₀)", f"${bs_ref:.4f}")
        c3.metric("Heston Implied Vol", f"{h_iv:.4%}" if not np.isnan(h_iv) else "N/A")

        strikes_h = np.linspace(h_S * 0.7, h_S * 1.3, 25)
        with st.spinner("Computing implied vol smile..."):
            heston_prices = [heston_price(h_S, k, h_T, h_r, h_v0, h_kappa, h_theta, h_sigma, h_rho, h_type) for k in strikes_h]
            heston_ivs = [implied_vol(p, h_S, k, h_T, h_r, h_type) for p, k in zip(heston_prices, strikes_h)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes_h, y=heston_ivs, mode="lines+markers", name="Heston IV Smile"))
        fig.add_hline(y=np.sqrt(h_v0), line_dash="dash", annotation_text="√v₀")
        fig.update_layout(title="Heston Implied Volatility Smile", xaxis_title="Strike", yaxis_title="Implied Vol", height=450)
        st.plotly_chart(fig, use_container_width=True)

# ---------- Greeks Profiles ----------
with tab_profiles:
    st.subheader("Greeks as a Function of Spot Price")
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        p_K = st.number_input("Strike", value=100.0, step=1.0, key="p_K")
        p_T = st.number_input("Expiry (y)", value=0.25, step=0.05, min_value=0.01, key="p_T")
        p_r = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="p_r")
        p_sigma = st.number_input("Vol", value=0.20, step=0.01, format="%.4f", key="p_sigma")
        p_type = st.selectbox("Type", ["call", "put"], key="p_type")
        p_range = st.slider("Spot Range (% around strike)", 10, 50, 30, key="p_range")

    with col_p2:
        spots = np.linspace(p_K * (1 - p_range / 100), p_K * (1 + p_range / 100), 200)
        greeks_data = {name: [] for name in ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Volga"]}
        prices = []
        for s in spots:
            g = all_greeks(s, p_K, p_T, p_r, p_sigma, p_type)
            for name in greeks_data:
                greeks_data[name].append(g[name])
            prices.append(bs_price(s, p_K, p_T, p_r, p_sigma, p_type))

        fig = make_subplots(rows=3, cols=2, subplot_titles=list(greeks_data.keys()),
                            vertical_spacing=0.08, horizontal_spacing=0.08)
        positions = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
        colors = ["#2962FF", "#00C853", "#FF6D00", "#D50000", "#AA00FF", "#00BFA5"]
        for idx, (name, vals) in enumerate(greeks_data.items()):
            r_pos, c_pos = positions[idx]
            fig.add_trace(go.Scatter(x=spots, y=vals, name=name, line=dict(color=colors[idx], width=2)),
                          row=r_pos, col=c_pos)
            fig.add_vline(x=p_K, line_dash="dot", line_color="gray", row=r_pos, col=c_pos)
        fig.update_layout(height=900, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Option Price Profile")
        fig_price = go.Figure()
        fig_price.add_trace(go.Scatter(x=spots, y=prices, name="Option Price", line=dict(color="#2962FF", width=2)))
        payoff = [max(s - p_K, 0) if p_type == "call" else max(p_K - s, 0) for s in spots]
        fig_price.add_trace(go.Scatter(x=spots, y=payoff, name="Intrinsic Value", line=dict(color="#999", dash="dash")))
        fig_price.add_vline(x=p_K, line_dash="dot", line_color="gray")
        fig_price.update_layout(xaxis_title="Spot", yaxis_title="Value", height=400)
        st.plotly_chart(fig_price, use_container_width=True)

        st.divider()
        st.subheader("Greeks vs Time to Expiry")
        expiries_range = np.linspace(0.01, 2.0, 200)
        greeks_vs_T = {name: [] for name in ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Volga"]}
        for t_val in expiries_range:
            g = all_greeks(p_K, p_K, t_val, p_r, p_sigma, p_type)
            for name in greeks_vs_T:
                greeks_vs_T[name].append(g[name])

        fig_t = make_subplots(rows=3, cols=2, subplot_titles=list(greeks_vs_T.keys()),
                              vertical_spacing=0.08, horizontal_spacing=0.08)
        for idx, (name, vals) in enumerate(greeks_vs_T.items()):
            r_pos, c_pos = positions[idx]
            fig_t.add_trace(go.Scatter(x=expiries_range, y=vals, name=name,
                                       line=dict(color=colors[idx], width=2)),
                            row=r_pos, col=c_pos)
        fig_t.update_layout(height=900, showlegend=False,
                            xaxis_title="T (years)", xaxis2_title="T (years)",
                            xaxis3_title="T (years)", xaxis4_title="T (years)",
                            xaxis5_title="T (years)", xaxis6_title="T (years)")
        st.plotly_chart(fig_t, use_container_width=True)

        st.divider()
        st.subheader("Greeks vs Volatility")
        vols_range = np.linspace(0.05, 1.0, 200)
        greeks_vs_vol = {name: [] for name in ["Delta", "Gamma", "Vega", "Theta", "Vanna", "Volga"]}
        for v_val in vols_range:
            g = all_greeks(p_K, p_K, p_T, p_r, v_val, p_type)
            for name in greeks_vs_vol:
                greeks_vs_vol[name].append(g[name])

        fig_v = make_subplots(rows=3, cols=2, subplot_titles=list(greeks_vs_vol.keys()),
                              vertical_spacing=0.08, horizontal_spacing=0.08)
        for idx, (name, vals) in enumerate(greeks_vs_vol.items()):
            r_pos, c_pos = positions[idx]
            fig_v.add_trace(go.Scatter(x=vols_range * 100, y=vals, name=name,
                                       line=dict(color=colors[idx], width=2)),
                            row=r_pos, col=c_pos)
        fig_v.update_layout(height=900, showlegend=False,
                            xaxis_title="Vol (%)", xaxis2_title="Vol (%)",
                            xaxis3_title="Vol (%)", xaxis4_title="Vol (%)",
                            xaxis5_title="Vol (%)", xaxis6_title="Vol (%)")
        st.plotly_chart(fig_v, use_container_width=True)
