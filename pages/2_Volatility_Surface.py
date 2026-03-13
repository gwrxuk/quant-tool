import streamlit as st
import numpy as np
import plotly.graph_objects as go
from core.volatility import (
    generate_synthetic_surface, build_vol_surface,
    sabr_calibrate, sabr_implied_vol, dupire_local_vol,
)
from core.pricing import bs_price, local_vol_mc_price

st.set_page_config(page_title="Volatility Surface", layout="wide")
st.title("Volatility Surface & SABR Calibration")

tab_surface, tab_sabr, tab_smile, tab_local = st.tabs(["3D Surface", "SABR Calibration", "Vol Smile Analysis", "Local Volatility"])

# ---------- 3D Surface ----------
with tab_surface:
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        st.subheader("Surface Parameters")
        s_spot = st.number_input("Spot", value=100.0, step=1.0, key="surf_S")
        s_base_vol = st.slider("Base Vol", 0.05, 0.60, 0.20, 0.01, key="surf_bv")
        s_seed = st.number_input("Random Seed", value=42, step=1, key="surf_seed")
        if st.button("Generate Surface", key="gen_surf"):
            st.session_state["surf_generated"] = True

    with col_s2:
        if st.session_state.get("surf_generated", False) or True:
            np.random.seed(int(s_seed))
            strikes, expiries, vol_matrix = generate_synthetic_surface(
                S=s_spot, base_vol=s_base_vol, n_strikes=20, n_expiries=8
            )

            fig_3d = go.Figure(data=[go.Surface(
                x=strikes, y=expiries * 365, z=vol_matrix * 100,
                colorscale="Viridis", colorbar=dict(title="Vol (%)"),
            )])
            fig_3d.update_layout(
                title="Implied Volatility Surface",
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Days to Expiry",
                    zaxis_title="Implied Vol (%)",
                    camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
                ),
                height=650,
            )
            st.plotly_chart(fig_3d, use_container_width=True)

            st.subheader("Volatility Matrix (Heatmap)")
            fig_heat = go.Figure(data=go.Heatmap(
                x=np.round(strikes, 1),
                y=[f"{d:.0f}d" for d in expiries * 365],
                z=vol_matrix * 100,
                colorscale="RdYlBu_r",
                text=np.round(vol_matrix * 100, 2),
                texttemplate="%{text}%",
                colorbar=dict(title="Vol (%)"),
            ))
            fig_heat.update_layout(
                title="Vol Surface Heatmap",
                xaxis_title="Strike", yaxis_title="Expiry",
                height=400,
            )
            st.plotly_chart(fig_heat, use_container_width=True)

# ---------- SABR Calibration ----------
with tab_sabr:
    st.subheader("SABR Model Calibration")
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        sabr_F = st.number_input("Forward (F)", value=100.0, step=1.0, key="sabr_F")
        sabr_T = st.number_input("Expiry (years)", value=0.25, step=0.05, min_value=0.01, key="sabr_T")
        sabr_beta = st.selectbox("Beta (β)", [0.0, 0.25, 0.5, 0.75, 1.0], index=2, key="sabr_beta")
        sabr_n_strikes = st.slider("Number of Strikes", 5, 25, 11, key="sabr_ns")

        st.markdown("**Synthetic Market Data Parameters**")
        sabr_true_alpha = st.number_input("True α", value=0.3, step=0.05, format="%.3f", key="sabr_ta")
        sabr_true_rho = st.number_input("True ρ", value=-0.3, step=0.05, min_value=-0.99, max_value=0.99, format="%.3f", key="sabr_tr")
        sabr_true_nu = st.number_input("True ν", value=0.4, step=0.05, format="%.3f", key="sabr_tn")

    with col_c2:
        strikes_sabr = np.linspace(sabr_F * 0.75, sabr_F * 1.25, sabr_n_strikes)
        market_vols = np.array([
            sabr_implied_vol(sabr_F, K, sabr_T, sabr_true_alpha, sabr_beta,
                             sabr_true_rho, sabr_true_nu)
            for K in strikes_sabr
        ])
        noise = np.random.normal(0, 0.002, len(strikes_sabr))
        market_vols_noisy = market_vols + noise

        result = sabr_calibrate(sabr_F, strikes_sabr, market_vols_noisy, sabr_T, beta=sabr_beta)

        fitted_vols = np.array([
            sabr_implied_vol(sabr_F, K, sabr_T, result["alpha"], result["beta"],
                             result["rho"], result["nu"])
            for K in strikes_sabr
        ])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("α (calibrated)", f"{result['alpha']:.4f}", f"true: {sabr_true_alpha:.3f}")
        c2.metric("ρ (calibrated)", f"{result['rho']:.4f}", f"true: {sabr_true_rho:.3f}")
        c3.metric("ν (calibrated)", f"{result['nu']:.4f}", f"true: {sabr_true_nu:.3f}")
        c4.metric("RMSE", f"{result['rmse']:.6f}")

        fig_sabr = go.Figure()
        fig_sabr.add_trace(go.Scatter(
            x=strikes_sabr, y=market_vols_noisy * 100,
            mode="markers", name="Market (noisy)", marker=dict(size=8, color="#FF6D00"),
        ))
        fig_sabr.add_trace(go.Scatter(
            x=strikes_sabr, y=market_vols * 100,
            mode="lines", name="True SABR", line=dict(dash="dash", color="#999"),
        ))
        fig_sabr.add_trace(go.Scatter(
            x=strikes_sabr, y=fitted_vols * 100,
            mode="lines+markers", name="Calibrated SABR", line=dict(color="#2962FF", width=2),
        ))
        fig_sabr.update_layout(
            title="SABR Calibration Fit",
            xaxis_title="Strike", yaxis_title="Implied Vol (%)",
            height=500,
        )
        st.plotly_chart(fig_sabr, use_container_width=True)

# ---------- Vol Smile Analysis ----------
with tab_smile:
    st.subheader("Volatility Smile by Expiry")
    col_sm1, col_sm2 = st.columns([1, 3])
    with col_sm1:
        sm_S = st.number_input("Spot", value=100.0, step=1.0, key="sm_S")
        sm_base = st.slider("ATM Vol", 0.05, 0.60, 0.20, 0.01, key="sm_base")
        sm_skew = st.slider("Skew Intensity", -0.5, 0.0, -0.15, 0.01, key="sm_skew")
        sm_convex = st.slider("Smile Convexity", 0.0, 1.0, 0.4, 0.05, key="sm_conv")

    with col_sm2:
        strikes_sm = np.linspace(sm_S * 0.7, sm_S * 1.3, 50)
        expiries_sm = [0.083, 0.25, 0.5, 1.0]
        colors_sm = ["#2962FF", "#00C853", "#FF6D00", "#D50000"]

        fig_smile = go.Figure()
        for T_sm, color in zip(expiries_sm, colors_sm):
            vols_sm = []
            for K in strikes_sm:
                m = np.log(K / sm_S)
                v = sm_base + sm_skew * m + sm_convex * m**2 - 0.02 * np.sqrt(T_sm)
                vols_sm.append(max(v, 0.03))
            fig_smile.add_trace(go.Scatter(
                x=strikes_sm, y=np.array(vols_sm) * 100,
                name=f"T = {T_sm:.3f}y ({T_sm*365:.0f}d)",
                line=dict(color=color, width=2),
            ))
        fig_smile.add_vline(x=sm_S, line_dash="dot", line_color="gray", annotation_text="ATM")
        fig_smile.update_layout(
            title="Implied Volatility Smile Across Expiries",
            xaxis_title="Strike", yaxis_title="Implied Vol (%)",
            height=500,
        )
        st.plotly_chart(fig_smile, use_container_width=True)

        st.subheader("Skew and Term Structure")
        moneyness = np.log(strikes_sm / sm_S)
        fig_skew = go.Figure()
        for T_sm, color in zip(expiries_sm, colors_sm):
            vols_sm = [max(sm_base + sm_skew * m + sm_convex * m**2 - 0.02 * np.sqrt(T_sm), 0.03)
                       for m in moneyness]
            fig_skew.add_trace(go.Scatter(
                x=moneyness, y=np.array(vols_sm) * 100,
                name=f"T = {T_sm*365:.0f}d",
                line=dict(color=color, width=2),
            ))
        fig_skew.add_vline(x=0, line_dash="dot", line_color="gray", annotation_text="ATM")
        fig_skew.update_layout(
            title="Vol vs Log-Moneyness",
            xaxis_title="Log(K/S)", yaxis_title="Implied Vol (%)",
            height=450,
        )
        st.plotly_chart(fig_skew, use_container_width=True)

# ---------- Local Volatility (Dupire) ----------
with tab_local:
    st.subheader("Dupire Local Volatility Surface")
    st.markdown("""
    The Dupire (1994) local volatility model extracts a state-dependent volatility function
    σ(S, t) from the implied volatility surface, ensuring exact calibration to all European
    option prices. The local vol is derived via:

    σ²_local(K,T) = (∂C/∂T + rK·∂C/∂K) / (½K²·∂²C/∂K²)
    """)

    col_lv1, col_lv2 = st.columns([1, 3])
    with col_lv1:
        lv_spot = st.number_input("Spot", value=100.0, step=1.0, key="lv_S")
        lv_base_vol = st.slider("Base IV", 0.05, 0.60, 0.20, 0.01, key="lv_bv")
        lv_r = st.number_input("Rate", value=0.05, step=0.005, format="%.4f", key="lv_r")
        lv_seed = st.number_input("Seed", value=42, step=1, key="lv_seed")

    with col_lv2:
        np.random.seed(int(lv_seed))
        lv_strikes, lv_expiries, lv_vol_matrix = generate_synthetic_surface(
            S=lv_spot, base_vol=lv_base_vol, n_strikes=20, n_expiries=8
        )

        lv_matrix = dupire_local_vol(lv_strikes, lv_expiries, lv_vol_matrix, lv_spot, lv_r)

        fig_lv3d = go.Figure(data=[go.Surface(
            x=lv_strikes, y=lv_expiries * 365, z=lv_matrix * 100,
            colorscale="Inferno", colorbar=dict(title="Local Vol (%)"),
        )])
        fig_lv3d.update_layout(
            title="Dupire Local Volatility Surface",
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Days to Expiry",
                zaxis_title="Local Vol (%)",
                camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
            ),
            height=650,
        )
        st.plotly_chart(fig_lv3d, use_container_width=True)

        st.subheader("Implied Vol vs Local Vol Comparison")
        mid_exp_idx = len(lv_expiries) // 2
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=lv_strikes, y=lv_vol_matrix[mid_exp_idx] * 100,
            mode="lines+markers", name="Implied Vol",
            line=dict(color="#2962FF", width=2),
        ))
        fig_cmp.add_trace(go.Scatter(
            x=lv_strikes, y=lv_matrix[mid_exp_idx] * 100,
            mode="lines+markers", name="Local Vol (Dupire)",
            line=dict(color="#FF6D00", width=2, dash="dash"),
        ))
        fig_cmp.update_layout(
            title=f"Slice at T = {lv_expiries[mid_exp_idx]*365:.0f} days",
            xaxis_title="Strike", yaxis_title="Volatility (%)",
            height=450,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        st.subheader("Local Vol Heatmap")
        fig_lv_heat = go.Figure(data=go.Heatmap(
            x=np.round(lv_strikes, 1),
            y=[f"{d:.0f}d" for d in lv_expiries * 365],
            z=lv_matrix * 100,
            colorscale="Inferno",
            text=np.round(lv_matrix * 100, 2),
            texttemplate="%{text}%",
            colorbar=dict(title="Local Vol (%)"),
        ))
        fig_lv_heat.update_layout(
            title="Local Vol Surface Heatmap",
            xaxis_title="Strike", yaxis_title="Expiry",
            height=400,
        )
        st.plotly_chart(fig_lv_heat, use_container_width=True)
