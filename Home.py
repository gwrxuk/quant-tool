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
    """)

st.divider()

st.markdown("""
### Technical Stack
- **Pricing Models**: Black-Scholes, Heston (characteristic function integration),
  Dupire Local Volatility (Monte Carlo), SABR
- **Greeks**: Analytical (BS) and numerical (finite difference) — Delta, Gamma, Vega,
  Theta, Rho, Vanna, Volga, Charm, Speed
- **Volatility**: Cubic spline surface interpolation, SABR calibration (Hagan et al. 2002),
  Dupire local vol extraction
- **Simulation**: GBM and Heston Euler-discretized paths, historical data backtesting
- **Risk**: Full second-order Taylor P&L attribution (incl. Vanna & Volga),
  Monte Carlo VaR, scenario stress testing, portfolio-level attribution
""")
