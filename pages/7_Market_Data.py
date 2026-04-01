import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from core.market_data import (
    get_quote, get_historical, get_expiry_dates, get_options_chain,
    get_iv_surface, get_risk_free_rate, get_market_overview,
    enrich_chain_with_greeks, PERIOD_MAP, POPULAR_TICKERS,
)

st.set_page_config(page_title="Market Data", layout="wide")
st.title("Real-Time Market Data")

# ── Sidebar ──────────────────────────────────────────────────────────────
ticker = st.sidebar.selectbox("Ticker", POPULAR_TICKERS, index=0)
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.sidebar.caption("Page will refresh every 30 seconds.")

if auto_refresh:
    import time as _time
    _time.sleep(30)
    st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────────
tab_quote, tab_chart, tab_chain, tab_surface, tab_overview = st.tabs([
    "Live Quote", "Price Chart", "Options Chain", "IV Surface", "Market Overview",
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Live Quote
# ═══════════════════════════════════════════════════════════════════════════
with tab_quote:
    try:
        q = get_quote(ticker)
    except Exception as exc:
        st.error(f"Failed to fetch quote for **{ticker}**: {exc}")
        q = None

    if q and not np.isnan(q.get("price", np.nan)):
        st.subheader(f"{q['name']}  ({q['symbol']})")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price", f"${q['price']:.2f}",
                  delta=f"{q['change']:+.2f} ({q['change_pct']:+.2f}%)")
        c2.metric("Day Range",
                  f"${q['day_low']:.2f} – ${q['day_high']:.2f}"
                  if not (np.isnan(q['day_low']) or np.isnan(q['day_high'])) else "N/A")
        c3.metric("Volume", f"{q['volume']:,.0f}" if q['volume'] else "N/A")
        c4.metric("Market Cap",
                  f"${q['market_cap']/1e9:.1f}B" if q.get("market_cap") else "N/A")

        st.divider()

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Bid", f"${q['bid']:.2f}" if not np.isnan(q['bid']) else "N/A")
        c6.metric("Ask", f"${q['ask']:.2f}" if not np.isnan(q['ask']) else "N/A")
        c7.metric("52-Wk High",
                  f"${q['fifty_two_wk_high']:.2f}" if not np.isnan(q['fifty_two_wk_high']) else "N/A")
        c8.metric("52-Wk Low",
                  f"${q['fifty_two_wk_low']:.2f}" if not np.isnan(q['fifty_two_wk_low']) else "N/A")

        st.caption(f"Exchange: {q['exchange']}  ·  Currency: {q['currency']}  ·  Updated: {q['timestamp']}")
    elif q:
        st.warning(f"Quote data for **{ticker}** is incomplete. Try refreshing.")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Price Chart
# ═══════════════════════════════════════════════════════════════════════════
with tab_chart:
    period = st.radio("Period", list(PERIOD_MAP.keys()), horizontal=True, index=3)

    try:
        hist = get_historical(ticker, period)
    except Exception as exc:
        st.error(f"Could not load historical data: {exc}")
        hist = pd.DataFrame()

    if hist.empty:
        st.warning("No historical data available.")
    else:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.03,
        )

        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist["Open"], high=hist["High"],
            low=hist["Low"], close=hist["Close"],
            increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
            name="OHLC",
        ), row=1, col=1)

        colors = np.where(hist["Close"] >= hist["Open"], "#26A69A", "#EF5350")
        fig.add_trace(go.Bar(
            x=hist.index, y=hist["Volume"], marker_color=colors,
            name="Volume", showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            title=f"{ticker} — {period}",
            xaxis_rangeslider_visible=False,
            height=620,
            margin=dict(t=40, b=20),
            legend=dict(orientation="h", y=1.02, x=0),
        )
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        st.plotly_chart(fig, width="stretch")

        with st.expander("Raw data"):
            st.dataframe(hist.tail(50), width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Options Chain
# ═══════════════════════════════════════════════════════════════════════════
with tab_chain:
    try:
        expiries = get_expiry_dates(ticker)
    except Exception as exc:
        st.error(f"Options data unavailable: {exc}")
        expiries = ()

    if not expiries:
        st.info("No options data for this ticker.")
    else:
        sel_expiry = st.selectbox("Expiry Date", expiries, index=0)
        try:
            chain = get_options_chain(ticker, sel_expiry)
        except Exception as exc:
            st.error(f"Failed to load chain: {exc}")
            chain = None

        if chain:
            spot = chain["spot"]
            T = chain["T"]
            st.caption(f"Spot: **${spot:.2f}**  ·  T: **{T:.4f}y** ({T*365:.0f} days)")

            rfr = get_risk_free_rate()

            col_calls, col_puts = st.columns(2)

            with col_calls:
                st.subheader("Calls")
                calls = enrich_chain_with_greeks(chain["calls"], spot, T, rfr, "call")
                display_cols = ["strike", "lastPrice", "bid", "ask", "volume",
                                "openInterest", "iv", "delta", "gamma", "vega", "theta"]
                available = [c for c in display_cols if c in calls.columns]
                st.dataframe(
                    calls[available].style.format({
                        "lastPrice": "${:.2f}", "bid": "${:.2f}", "ask": "${:.2f}",
                        "iv": "{:.2%}", "delta": "{:.4f}", "gamma": "{:.4f}",
                        "vega": "{:.4f}", "theta": "{:.4f}",
                    }, na_rep="—"),
                    width="stretch", height=420,
                )

            with col_puts:
                st.subheader("Puts")
                puts = enrich_chain_with_greeks(chain["puts"], spot, T, rfr, "put")
                available_p = [c for c in display_cols if c in puts.columns]
                st.dataframe(
                    puts[available_p].style.format({
                        "lastPrice": "${:.2f}", "bid": "${:.2f}", "ask": "${:.2f}",
                        "iv": "{:.2%}", "delta": "{:.4f}", "gamma": "{:.4f}",
                        "vega": "{:.4f}", "theta": "{:.4f}",
                    }, na_rep="—"),
                    width="stretch", height=420,
                )

            st.divider()
            st.subheader("IV Smile for Selected Expiry")

            fig_smile = go.Figure()
            if "iv" in calls.columns:
                mask_c = calls["iv"].notna() & (calls["iv"] > 0)
                fig_smile.add_trace(go.Scatter(
                    x=calls.loc[mask_c, "strike"],
                    y=calls.loc[mask_c, "iv"] * 100,
                    mode="lines+markers", name="Calls",
                    line=dict(color="#2962FF", width=2),
                ))
            if "iv" in puts.columns:
                mask_p = puts["iv"].notna() & (puts["iv"] > 0)
                fig_smile.add_trace(go.Scatter(
                    x=puts.loc[mask_p, "strike"],
                    y=puts.loc[mask_p, "iv"] * 100,
                    mode="lines+markers", name="Puts",
                    line=dict(color="#FF6D00", width=2),
                ))
            fig_smile.add_vline(x=spot, line_dash="dot", line_color="gray",
                                annotation_text=f"Spot ${spot:.2f}")
            fig_smile.update_layout(
                title=f"IV Smile — {ticker} {sel_expiry}",
                xaxis_title="Strike", yaxis_title="Implied Vol (%)",
                height=450,
            )
            st.plotly_chart(fig_smile, width="stretch")

            st.subheader("Open Interest Distribution")
            fig_oi = go.Figure()
            if "openInterest" in calls.columns:
                fig_oi.add_trace(go.Bar(
                    x=calls["strike"], y=calls["openInterest"],
                    name="Calls OI", marker_color="#26A69A", opacity=0.7,
                ))
            if "openInterest" in puts.columns:
                fig_oi.add_trace(go.Bar(
                    x=puts["strike"], y=puts["openInterest"],
                    name="Puts OI", marker_color="#EF5350", opacity=0.7,
                ))
            fig_oi.add_vline(x=spot, line_dash="dot", line_color="gray")
            fig_oi.update_layout(
                barmode="group", height=400,
                xaxis_title="Strike", yaxis_title="Open Interest",
            )
            st.plotly_chart(fig_oi, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Implied Volatility Surface
# ═══════════════════════════════════════════════════════════════════════════
with tab_surface:
    st.subheader(f"Implied Volatility Surface — {ticker}")
    max_exp = st.slider("Max expiries to include", 2, 12, 8, key="iv_max_exp")
    m_lo = st.slider("Moneyness lower bound", 0.50, 0.95, 0.80, 0.05, key="iv_m_lo")
    m_hi = st.slider("Moneyness upper bound", 1.05, 1.50, 1.20, 0.05, key="iv_m_hi")

    if st.button("Build IV Surface", key="build_iv"):
        with st.spinner("Fetching options chains across expiries…"):
            surface = get_iv_surface(ticker, max_expiries=max_exp,
                                     moneyness_range=(m_lo, m_hi))

        if not surface:
            st.warning("Not enough options data to build a surface.")
        else:
            strikes = surface["strikes"]
            T_years = surface["expiries_years"]
            iv_mat = surface["iv_matrix"] * 100
            spot = surface["spot"]

            fig_3d = go.Figure(data=[go.Surface(
                x=strikes, y=T_years * 365, z=iv_mat,
                colorscale="Viridis",
                colorbar=dict(title="IV (%)"),
            )])
            fig_3d.update_layout(
                title=f"{ticker} Implied Volatility Surface (Market Data)",
                scene=dict(
                    xaxis_title="Strike",
                    yaxis_title="Days to Expiry",
                    zaxis_title="IV (%)",
                    camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
                ),
                height=650,
            )
            st.plotly_chart(fig_3d, width="stretch")

            st.subheader("IV Heatmap")
            labels = surface["expiry_labels"]
            fig_heat = go.Figure(data=go.Heatmap(
                x=np.round(strikes, 1),
                y=labels,
                z=iv_mat,
                colorscale="RdYlBu_r",
                text=np.round(iv_mat, 1),
                texttemplate="%{text}%",
                colorbar=dict(title="IV (%)"),
            ))
            fig_heat.update_layout(
                title=f"{ticker} IV Heatmap",
                xaxis_title="Strike", yaxis_title="Expiry",
                height=420,
            )
            st.plotly_chart(fig_heat, width="stretch")

            st.subheader("Term Structure (ATM)")
            atm_idx = np.argmin(np.abs(strikes - spot))
            atm_ivs = iv_mat[:, atm_idx]
            fig_ts = go.Figure(go.Scatter(
                x=T_years * 365, y=atm_ivs,
                mode="lines+markers",
                line=dict(color="#2962FF", width=2),
            ))
            fig_ts.update_layout(
                title="ATM IV Term Structure",
                xaxis_title="Days to Expiry", yaxis_title="IV (%)",
                height=380,
            )
            st.plotly_chart(fig_ts, width="stretch")

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – Market Overview
# ═══════════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Major Indices")
    try:
        overview = get_market_overview()
    except Exception as exc:
        st.error(f"Could not load market overview: {exc}")
        overview = pd.DataFrame()

    if overview.empty:
        st.info("Market data unavailable.")
    else:
        cols = st.columns(len(overview))
        for i, (_, row) in enumerate(overview.iterrows()):
            with cols[i]:
                delta_str = f"{row['Change']:+.2f} ({row['Change %']:+.2f}%)"
                st.metric(
                    row["Index"],
                    f"{row['Price']:,.2f}",
                    delta=delta_str,
                )

        st.divider()
        st.dataframe(overview.set_index("Index"), width="stretch")

    st.subheader("Risk-Free Rate Estimate")
    try:
        rfr_3m = get_risk_free_rate("3m")
        rfr_10y = get_risk_free_rate("10y")
    except Exception:
        rfr_3m, rfr_10y = 0.045, 0.045
    c1, c2 = st.columns(2)
    c1.metric("3-Month T-Bill", f"{rfr_3m:.2%}")
    c2.metric("10-Year Treasury", f"{rfr_10y:.2%}")
