import streamlit as st

st.set_page_config(
    page_title="Quant Options Toolkit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Quantitative Options Analysis Toolkit")

st.markdown("""
**A comprehensive front-office quant toolkit for options market making and proprietary trading.**

Use the sidebar to navigate between modules:
""")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Pricing & Analytics
    - **Options Pricer** — Black-Scholes, Heston stochastic volatility, and Local
      Volatility (Dupire) pricing with 9 analytical Greeks and multi-dimensional
      sensitivity profiles (vs Spot, Time, and Volatility)
    - **Volatility Surface** — Interactive 3D implied and local vol surface
      construction, SABR calibration, Dupire local vol extraction, and smile analysis
    - **Index Options** — SPX, NDX, RUT, VIX cash-settled European pricing,
      put-call parity, term structure with forward vols, skew analysis,
      VIX futures pricing (Black '76), expiry calendar, and §1256 tax calculator

    ### Strategy & Backtesting
    - **Strategy Backtester** — 8 strategy templates (straddles, strangles,
      butterflies, iron condors, calendar spreads) under GBM or Heston dynamics,
      delta-hedging with vol mismatch, and historical data CSV upload
    """)

with col2:
    st.markdown("""
    ### Risk Management
    - **Scenario Simulator** — Stress-test portfolios across spot, vol, and time
      dimensions with interactive P&L heatmaps and spot ladders
    - **P&L Attribution** — Full second-order Taylor decomposition (Delta, Gamma,
      Vega, Theta, Vanna, Volga) for single positions and multi-position portfolios

    ### Dashboard
    - **Risk Dashboard** — Portfolio risk monitoring with Monte Carlo VaR/CVaR,
      Greeks exposure by expiry and strike, and configurable alert thresholds

    ### Live Data
    - **Market Data** — Real-time quotes, interactive candlestick charts,
      live options chains with Greeks, implied volatility surfaces from
      market data, and major index overview

    ### Broker Integration
    - **Broker** — Unified trading interface for Interactive Brokers, Alpaca,
      and Schwab with built-in paper trading, order management, position
      tracking, and trade analytics
    """)

st.divider()

st.markdown("""
### Technical Stack
- **Pricing Models**: Black-Scholes, Heston (characteristic function integration),
  Dupire Local Volatility (Monte Carlo), SABR, Black '76 (VIX futures)
- **Greeks**: Analytical (BS) and numerical (finite difference) — Delta, Gamma, Vega,
  Theta, Rho, Vanna, Volga, Charm, Speed — with continuous dividend yield support
- **Index Options**: European pricing with dividend yield, put-call parity verification,
  term structure & forward vol extraction, skew metrics (25Δ risk reversal, butterfly)
- **Volatility**: Cubic spline surface interpolation, SABR calibration (Hagan et al. 2002),
  Dupire local vol extraction
- **Simulation**: GBM and Heston Euler-discretized paths, historical data backtesting
- **Risk**: Full second-order Taylor P&L attribution (incl. Vanna & Volga),
  Monte Carlo VaR, scenario stress testing, portfolio-level attribution
- **Market Data**: Real-time quotes via yfinance, live options chains,
  implied volatility surfaces from market data, Treasury yield risk-free rate
- **Broker Integration**: IBKR (ib_async), Alpaca (alpaca-py), Schwab (schwab-py),
  and built-in paper trading with order management and position tracking
""")
