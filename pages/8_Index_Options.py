import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.index_options import (
    INDEX_SPECS, get_index_spec, list_index_symbols,
    european_price, european_greeks, put_call_parity_check,
    implied_dividend_yield, cash_settlement_value, breakeven_price,
    compute_term_structure, term_structure_with_forwards,
    compute_skew_metrics, vix_option_price, vix_futures_price_approximation,
    vix_term_structure, section_1256_tax, next_monthly_expiry,
    quarterly_expiries, weekly_expiries,
)
from core.pricing import implied_vol
from core.market_data import get_quote, get_expiry_dates, get_options_chain, get_risk_free_rate

st.set_page_config(page_title="Index Options", layout="wide")
st.title("Index Options Analytics")

# ── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.header("Index Selection")
idx_symbols = list_index_symbols()
selected_idx = st.sidebar.selectbox("Index", idx_symbols, index=0)
spec = get_index_spec(selected_idx)

if spec:
    st.sidebar.markdown(f"**{spec['name']}**")
    st.sidebar.caption(spec["description"])
    st.sidebar.markdown(f"""
    - **Underlying**: `{spec['underlying']}`
    - **Multiplier**: {spec['multiplier']}x
    - **Style**: {spec['style'].title()}
    - **Settlement**: {spec['settlement_type']}
    - **Exchange**: {spec['exchange']}
    - **Sec. 1256**: {'Yes' if spec.get('section_1256') else 'No'}
    """)

st.sidebar.divider()

# Fetch live data
_idx_spot = None
if st.sidebar.button("Load Live Data", key="idx_load"):
    with st.sidebar:
        with st.spinner("Fetching index data…"):
            try:
                q = get_quote(spec["underlying"])
                _idx_spot = q.get("price", None)
                st.session_state["idx_spot"] = _idx_spot
                st.session_state["idx_name"] = q.get("name", selected_idx)
            except Exception as e:
                st.error(f"Could not fetch data: {e}")

_idx_spot = st.session_state.get("idx_spot")
_idx_name = st.session_state.get("idx_name", selected_idx)
if _idx_spot:
    st.sidebar.metric(f"{_idx_name}", f"{_idx_spot:,.2f}")

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_pricer, tab_parity, tab_term, tab_skew, tab_vix, tab_calendar, tab_tax = st.tabs([
    "European Pricer", "Put-Call Parity", "Term Structure",
    "Skew Analysis", "VIX Options", "Expiry Calendar", "Tax (§1256)",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – European Index Options Pricer
# ═══════════════════════════════════════════════════════════════════════════
with tab_pricer:
    st.subheader(f"{spec.get('name', selected_idx)} — European Pricer")

    col_in, col_out = st.columns([1, 2])

    with col_in:
        default_spot = _idx_spot or 5000.0 if selected_idx in ("SPX", "SPXW") else _idx_spot or 1000.0
        ep_S = st.number_input("Index Level", value=float(default_spot), step=10.0, key="ep_S")
        ep_K = st.number_input("Strike", value=float(round(default_spot)), step=10.0, key="ep_K")
        ep_T = st.number_input("Time to Expiry (years)", value=0.25, step=0.01, min_value=0.001, key="ep_T")
        ep_r = st.number_input("Risk-Free Rate", value=0.045, step=0.005, format="%.4f", key="ep_r")
        ep_q = st.number_input("Dividend Yield", value=0.013, step=0.001, format="%.4f", key="ep_q")
        ep_sigma = st.number_input("Volatility (σ)", value=0.18, step=0.01, format="%.4f", key="ep_sigma")
        ep_type = st.selectbox("Option Type", ["call", "put"], key="ep_type")
        mult = spec.get("multiplier", 100)

    with col_out:
        price = european_price(ep_S, ep_K, ep_T, ep_r, ep_sigma, ep_type, ep_q)
        greeks = european_greeks(ep_S, ep_K, ep_T, ep_r, ep_sigma, ep_type, ep_q)
        contract_value = price * mult

        st.subheader("Price")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"{ep_type.title()} Price (per point)", f"${price:.4f}")
        c2.metric("Contract Value", f"${contract_value:,.2f}")
        be = breakeven_price(price, ep_K, ep_type, 1)
        c3.metric("Breakeven", f"{be:,.2f}")

        st.subheader("Greeks")
        g_cols = st.columns(5)
        for i, (name, val) in enumerate(greeks.items()):
            with g_cols[i]:
                fmt = f"{val:.6f}" if abs(val) < 1 else f"{val:.4f}"
                st.metric(name, fmt)

        st.divider()
        st.subheader("Cash Settlement Scenarios")
        settle_levels = np.linspace(ep_K * 0.9, ep_K * 1.1, 21)
        settle_pnl = []
        for lvl in settle_levels:
            settle_val = cash_settlement_value(lvl, ep_K, ep_type, mult)
            net = settle_val - contract_value
            settle_pnl.append({"Settlement Level": lvl, "Settlement Value": settle_val, "Net P&L": net})

        settle_df = pd.DataFrame(settle_pnl)

        fig_settle = go.Figure()
        fig_settle.add_trace(go.Scatter(
            x=settle_df["Settlement Level"], y=settle_df["Net P&L"],
            mode="lines+markers", name="Net P&L",
            line=dict(color="#2962FF", width=2),
        ))
        fig_settle.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_settle.add_vline(x=ep_K, line_dash="dot", line_color="gray", annotation_text=f"K={ep_K:,.0f}")
        fig_settle.add_vline(x=be, line_dash="dash", line_color="green", annotation_text=f"BE={be:,.0f}")
        fig_settle.update_layout(
            title="Cash Settlement P&L at Expiration",
            xaxis_title="Settlement Level", yaxis_title="Net P&L ($)",
            height=420,
        )
        st.plotly_chart(fig_settle, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Put-Call Parity
# ═══════════════════════════════════════════════════════════════════════════
with tab_parity:
    st.subheader("European Put-Call Parity Checker")
    st.markdown("""
    For European index options: **C − P = S·e^(−qT) − K·e^(−rT)**
    """)

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        pc_S = st.number_input("Index Level", value=float(_idx_spot or 5000.0), step=10.0, key="pc_S")
        pc_K = st.number_input("Strike", value=float(round(_idx_spot or 5000.0)), step=10.0, key="pc_K")
        pc_T = st.number_input("Time to Expiry (y)", value=0.25, step=0.01, min_value=0.001, key="pc_T")
    with col_p2:
        pc_r = st.number_input("Risk-Free Rate", value=0.045, step=0.005, format="%.4f", key="pc_r")
        pc_q = st.number_input("Dividend Yield", value=0.013, step=0.001, format="%.4f", key="pc_q")
        pc_call = st.number_input("Call Market Price", value=150.0, step=1.0, key="pc_call")
        pc_put = st.number_input("Put Market Price", value=120.0, step=1.0, key="pc_put")

    if st.button("Check Parity", key="pc_check"):
        parity = put_call_parity_check(pc_call, pc_put, pc_S, pc_K, pc_T, pc_r, pc_q)
        div_yield = implied_dividend_yield(pc_call, pc_put, pc_S, pc_K, pc_T, pc_r)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Theoretical C−P", f"${parity['theoretical_spread']:,.2f}")
        c2.metric("Actual C−P", f"${parity['actual_spread']:,.2f}")
        c3.metric("Violation", f"${parity['violation']:,.2f}")
        c4.metric("Implied Div Yield", f"{div_yield:.4%}" if not np.isnan(div_yield) else "N/A")

        if parity["parity_holds"]:
            st.success(f"Put-call parity holds within $0.50 (violation: ${abs(parity['violation']):.2f})")
        else:
            st.warning(f"Put-call parity violation detected: ${parity['violation']:.2f} ({parity['violation_pct']:.3f}% of spot)")
            if parity["violation"] > 0:
                st.info("Arbitrage: Sell call + buy put + buy forward (synthetic long is cheap)")
            else:
                st.info("Arbitrage: Buy call + sell put + sell forward (synthetic long is rich)")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Term Structure
# ═══════════════════════════════════════════════════════════════════════════
with tab_term:
    st.subheader("Implied Volatility Term Structure")

    ts_ticker = spec.get("underlying", "^GSPC") if spec else "^GSPC"
    ts_use_etf = st.checkbox("Use ETF proxy for options data", value=True)
    if ts_use_etf:
        etf_map = {"^GSPC": "SPY", "^NDX": "QQQ", "^RUT": "IWM", "^DJI": "DIA", "^VIX": "VIX"}
        ts_ticker = etf_map.get(ts_ticker, ts_ticker)
        st.caption(f"Using **{ts_ticker}** as ETF proxy for options chain data")

    if st.button("Build Term Structure", key="ts_build"):
        with st.spinner("Fetching options chains across expiries…"):
            try:
                expiries = get_expiry_dates(ts_ticker)
                if not expiries:
                    st.warning("No expiry dates available.")
                else:
                    spot_q = get_quote(ts_ticker)
                    spot = spot_q.get("price", np.nan) if spot_q else np.nan

                    strikes_by_exp, iv_by_exp, T_list = [], [], []
                    for exp in expiries[:12]:
                        try:
                            chain = get_options_chain(ts_ticker, exp)
                            if chain and chain["T"] > 1 / 365:
                                calls = chain["calls"]
                                iv_col = "iv" if "iv" in calls.columns else "impliedVolatility"
                                valid = calls[iv_col].notna() & (calls[iv_col] > 0.001)
                                strikes_by_exp.append(calls.loc[valid, "strike"].values)
                                iv_by_exp.append(calls.loc[valid, iv_col].values)
                                T_list.append(chain["T"])
                        except Exception:
                            continue

                    if len(T_list) < 2:
                        st.warning("Not enough expiry data to build term structure.")
                    else:
                        ts_df = compute_term_structure(spot, strikes_by_exp, iv_by_exp, T_list)
                        ts_df = term_structure_with_forwards(ts_df)

                        fig_ts = go.Figure()
                        fig_ts.add_trace(go.Scatter(
                            x=ts_df["days"], y=ts_df["atm_iv"] * 100,
                            mode="lines+markers", name="ATM IV",
                            line=dict(color="#2962FF", width=2),
                        ))
                        if "forward_vol" in ts_df.columns:
                            fwd_valid = ts_df["forward_vol"].notna()
                            fig_ts.add_trace(go.Scatter(
                                x=ts_df.loc[fwd_valid, "days"],
                                y=ts_df.loc[fwd_valid, "forward_vol"] * 100,
                                mode="lines+markers", name="Forward Vol",
                                line=dict(color="#FF6D00", width=2, dash="dash"),
                            ))
                        fig_ts.update_layout(
                            title=f"{selected_idx} ATM IV Term Structure",
                            xaxis_title="Days to Expiry",
                            yaxis_title="Implied Volatility (%)",
                            height=500,
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)

                        st.subheader("Term Structure Data")
                        display_ts = ts_df.copy()
                        for col in ["atm_iv", "forward_vol"]:
                            if col in display_ts.columns:
                                display_ts[col] = display_ts[col].apply(
                                    lambda x: f"{x:.2%}" if not np.isnan(x) else "—"
                                )
                        st.dataframe(display_ts, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Skew Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab_skew:
    st.subheader("Volatility Skew Analysis")

    sk_ticker = spec.get("underlying", "^GSPC") if spec else "^GSPC"
    sk_use_etf = st.checkbox("Use ETF proxy", value=True, key="sk_etf")
    if sk_use_etf:
        etf_map = {"^GSPC": "SPY", "^NDX": "QQQ", "^RUT": "IWM", "^DJI": "DIA", "^VIX": "VIX"}
        sk_ticker = etf_map.get(sk_ticker, sk_ticker)

    if st.button("Analyze Skew", key="sk_build"):
        with st.spinner("Fetching options data…"):
            try:
                expiries = get_expiry_dates(sk_ticker)
                if not expiries:
                    st.warning("No options data available.")
                else:
                    spot_q = get_quote(sk_ticker)
                    spot = spot_q.get("price", np.nan)

                    sel_expiries = expiries[:6]
                    fig_skew = go.Figure()
                    skew_rows = []
                    colors = ["#2962FF", "#00C853", "#FF6D00", "#D50000", "#AA00FF", "#00BFA5"]

                    for i, exp in enumerate(sel_expiries):
                        try:
                            chain = get_options_chain(sk_ticker, exp)
                            if not chain or chain["T"] < 1 / 365:
                                continue
                            calls = chain["calls"]
                            iv_col = "iv" if "iv" in calls.columns else "impliedVolatility"
                            valid = calls[iv_col].notna() & (calls[iv_col] > 0.001)
                            strikes = calls.loc[valid, "strike"].values
                            ivs = calls.loc[valid, iv_col].values

                            lo, hi = spot * 0.85, spot * 1.15
                            mask = (strikes >= lo) & (strikes <= hi)
                            if mask.sum() < 5:
                                continue

                            fig_skew.add_trace(go.Scatter(
                                x=strikes[mask], y=ivs[mask] * 100,
                                mode="lines+markers", name=f"{exp} ({chain['T']*365:.0f}d)",
                                line=dict(color=colors[i % len(colors)], width=2),
                            ))

                            metrics = compute_skew_metrics(strikes, ivs, spot, chain["T"])
                            if metrics:
                                metrics["expiry"] = exp
                                metrics["days"] = int(chain["T"] * 365)
                                skew_rows.append(metrics)
                        except Exception:
                            continue

                    fig_skew.add_vline(x=spot, line_dash="dot", line_color="gray",
                                       annotation_text=f"Spot {spot:,.2f}")
                    fig_skew.update_layout(
                        title=f"{selected_idx} Volatility Skew by Expiry",
                        xaxis_title="Strike", yaxis_title="Implied Vol (%)",
                        height=550,
                    )
                    st.plotly_chart(fig_skew, use_container_width=True)

                    if skew_rows:
                        st.subheader("Skew Metrics")
                        skew_df = pd.DataFrame(skew_rows)
                        fmt_cols = ["atm_iv", "put_25d_iv", "call_25d_iv", "skew_25d",
                                    "risk_reversal_25d", "butterfly_25d"]
                        for c in fmt_cols:
                            if c in skew_df.columns:
                                skew_df[c] = skew_df[c].apply(
                                    lambda x: f"{x:.2%}" if not np.isnan(x) else "—"
                                )
                        display_cols = ["expiry", "days"] + [c for c in fmt_cols if c in skew_df.columns]
                        st.dataframe(skew_df[display_cols], use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – VIX Options
# ═══════════════════════════════════════════════════════════════════════════
with tab_vix:
    st.subheader("VIX Options Pricing (Black '76)")
    st.markdown("""
    VIX options settle against VIX futures (SOQ), not spot VIX.
    Pricing uses the **Black '76 model** with VIX futures as the forward.
    """)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        vix_spot = st.number_input("VIX Spot Level", value=18.0, step=0.5, key="vix_spot")
        vix_K = st.number_input("Strike", value=20.0, step=0.5, key="vix_K")
        vix_T = st.number_input("Time to Expiry (y)", value=0.083, step=0.01, min_value=0.001, key="vix_T")
    with col_v2:
        vix_r = st.number_input("Risk-Free Rate", value=0.045, step=0.005, format="%.4f", key="vix_r")
        vix_sigma = st.number_input("VIX Vol of Vol", value=0.80, step=0.05, key="vix_sigma")
        vix_kappa = st.number_input("Mean Reversion Speed (κ)", value=5.0, step=0.5, key="vix_kappa")
        vix_theta = st.number_input("Long-Run VIX (θ)", value=20.0, step=1.0, key="vix_theta")
        vix_type = st.selectbox("Type", ["call", "put"], key="vix_type")

    vix_F = vix_futures_price_approximation(vix_spot, vix_T, vix_kappa, vix_theta)
    st.info(f"Estimated VIX Futures Price: **{vix_F:.2f}** (mean-reversion model)")

    vix_price = vix_option_price(vix_F, vix_K, vix_T, vix_r, vix_sigma, vix_type)
    vix_contract_val = vix_price * 100

    c1, c2, c3 = st.columns(3)
    c1.metric(f"VIX {vix_type.title()} Price", f"${vix_price:.4f}")
    c2.metric("Contract Value", f"${vix_contract_val:,.2f}")
    c3.metric("VIX Futures Estimate", f"{vix_F:.2f}")

    st.divider()
    st.subheader("VIX Futures Term Structure")

    vix_ts = vix_term_structure(vix_spot, n_months=8, kappa=vix_kappa, theta=vix_theta)
    fig_vts = go.Figure()
    fig_vts.add_trace(go.Scatter(
        x=vix_ts["month"], y=vix_ts["futures_price"],
        mode="lines+markers", name="VIX Futures",
        line=dict(color="#D50000", width=2),
    ))
    fig_vts.add_hline(y=vix_spot, line_dash="dot", line_color="gray",
                       annotation_text=f"Spot VIX = {vix_spot:.1f}")
    fig_vts.add_hline(y=vix_theta, line_dash="dash", line_color="blue",
                       annotation_text=f"θ = {vix_theta:.1f}")
    fig_vts.update_layout(
        title="VIX Futures Term Structure (Mean-Reversion Model)",
        xaxis_title="Months to Expiry", yaxis_title="VIX Level",
        height=420,
    )
    st.plotly_chart(fig_vts, use_container_width=True)

    st.subheader("VIX Option Price Surface")
    vix_strikes = np.linspace(max(vix_spot * 0.5, 10), vix_spot * 2.0, 20)
    vix_expiries_m = np.arange(1, 7)
    price_matrix = np.zeros((len(vix_expiries_m), len(vix_strikes)))

    for i, m in enumerate(vix_expiries_m):
        T_m = m / 12
        F_m = vix_futures_price_approximation(vix_spot, T_m, vix_kappa, vix_theta)
        for j, K_v in enumerate(vix_strikes):
            price_matrix[i, j] = vix_option_price(F_m, K_v, T_m, vix_r, vix_sigma, vix_type)

    fig_vsurf = go.Figure(data=[go.Surface(
        x=np.round(vix_strikes, 1), y=vix_expiries_m,
        z=price_matrix, colorscale="Inferno",
        colorbar=dict(title="Price ($)"),
    )])
    fig_vsurf.update_layout(
        title=f"VIX {vix_type.title()} Price Surface",
        scene=dict(xaxis_title="Strike", yaxis_title="Months", zaxis_title="Price ($)"),
        height=550,
    )
    st.plotly_chart(fig_vsurf, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 – Expiry Calendar
# ═══════════════════════════════════════════════════════════════════════════
with tab_calendar:
    st.subheader("Index Options Expiry Calendar")

    col_cal1, col_cal2, col_cal3 = st.columns(3)

    with col_cal1:
        st.markdown("**Next Monthly Expiry**")
        next_exp = next_monthly_expiry()
        days_to_exp = (next_exp - _dt.date.today()).days
        st.metric("Date", next_exp.strftime("%Y-%m-%d"), delta=f"{days_to_exp} days")

    with col_cal2:
        st.markdown("**Quarterly Expiries**")
        q_exps = quarterly_expiries(n=4)
        for qe in q_exps:
            days_q = (qe - _dt.date.today()).days
            st.text(f"{qe.strftime('%Y-%m-%d')}  ({days_q}d)")

    with col_cal3:
        st.markdown("**Weekly Expiries**")
        w_exps = weekly_expiries(n_weeks=8)
        for we in w_exps:
            days_w = (we - _dt.date.today()).days
            st.text(f"{we.strftime('%Y-%m-%d')}  ({days_w}d)")

    import datetime as _dt_mod

    st.divider()
    st.subheader("Contract Specifications Reference")
    spec_rows = []
    for sym, s in INDEX_SPECS.items():
        spec_rows.append({
            "Symbol": sym,
            "Name": s["name"],
            "Underlying": s["underlying"],
            "Multiplier": s["multiplier"],
            "Style": s["style"].title(),
            "Settlement": s["settlement_type"],
            "Exchange": s["exchange"],
            "§1256": "Yes" if s.get("section_1256") else "No",
        })
    st.dataframe(pd.DataFrame(spec_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 7 – Section 1256 Tax Treatment
# ═══════════════════════════════════════════════════════════════════════════
with tab_tax:
    st.subheader("Section 1256 Contract Tax Calculator")
    st.markdown("""
    Index options qualify as **Section 1256 contracts**: gains/losses are taxed
    **60% long-term / 40% short-term** regardless of holding period.
    This is a significant advantage over equity options (taxed as 100% short-term if held < 1 year).
    """)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        tax_gain = st.number_input("Net Gain/Loss ($)", value=10000.0, step=1000.0, key="tax_gain")
        tax_st_rate = st.number_input("Short-Term Rate", value=0.37, step=0.01, format="%.2f", key="tax_st")
        tax_lt_rate = st.number_input("Long-Term Rate", value=0.20, step=0.01, format="%.2f", key="tax_lt")

    with col_t2:
        if tax_gain > 0:
            result_1256 = section_1256_tax(tax_gain, tax_st_rate, tax_lt_rate)
            all_st = tax_gain * tax_st_rate
            all_lt = tax_gain * tax_lt_rate
            savings = all_st - result_1256["tax"]

            st.metric("§1256 Blended Tax", f"${result_1256['tax']:,.2f}")
            st.metric("Effective Rate", f"{result_1256['effective_rate']:.2%}")
            st.metric("vs 100% Short-Term", f"${all_st:,.2f}")
            st.metric("Tax Savings", f"${savings:,.2f}", delta=f"{savings/tax_gain*100:.1f}% saved")
        else:
            st.info("Enter a positive gain to calculate tax treatment.")

    st.divider()
    st.subheader("Tax Comparison: §1256 vs Equity Options")

    gain_range = np.linspace(1000, 100000, 50)
    tax_1256 = [section_1256_tax(g, tax_st_rate, tax_lt_rate)["tax"] for g in gain_range]
    tax_equity_st = gain_range * tax_st_rate
    tax_equity_lt = gain_range * tax_lt_rate

    fig_tax = go.Figure()
    fig_tax.add_trace(go.Scatter(
        x=gain_range, y=tax_1256,
        name="§1256 (60/40)", line=dict(color="#2962FF", width=2),
    ))
    fig_tax.add_trace(go.Scatter(
        x=gain_range, y=tax_equity_st,
        name=f"100% Short-Term ({tax_st_rate:.0%})", line=dict(color="#D50000", width=2, dash="dash"),
    ))
    fig_tax.add_trace(go.Scatter(
        x=gain_range, y=tax_equity_lt,
        name=f"100% Long-Term ({tax_lt_rate:.0%})", line=dict(color="#00C853", width=2, dash="dot"),
    ))
    fig_tax.update_layout(
        title="Tax Liability by Treatment",
        xaxis_title="Net Gain ($)", yaxis_title="Tax ($)",
        height=450,
    )
    st.plotly_chart(fig_tax, use_container_width=True)
