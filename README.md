# Quantitative Options Analysis Toolkit (QOAT)

A containerized, interactive platform for front-office derivatives analytics — covering the full workflow from options pricing and volatility calibration through strategy backtesting, P&L attribution, and real-time risk monitoring.

![Home](screenshots/01_home.png)

## Features

### Options Pricer & Greeks Calculator

Three pricing models with nine analytical Greeks and multi-dimensional sensitivity profiles.

- **Black-Scholes** closed-form pricing with implied volatility solver (Brent's method)
- **Heston stochastic volatility** via characteristic function integration
- **Dupire local volatility** extraction with Monte Carlo pricing
- Analytical Greeks: Delta, Gamma, Vega, Theta, Rho, Vanna, Volga, Charm, Speed
- Sensitivity profiles across spot, time-to-expiry, and volatility dimensions

![Options Pricer](screenshots/02_options_pricer.png)

### Volatility Surface & SABR Calibration

3D implied and local volatility surface construction with parametric calibration.

- Cubic spline interpolation across strikes and expiries
- **SABR model** calibration (Hagan et al., 2002) via Nelder-Mead optimization
- **Dupire local vol** surface extraction from implied vol, with comparison views and heatmaps
- Vol smile analysis by expiry slice

![Volatility Surface](screenshots/03_vol_surface.png)
![Local Volatility](screenshots/10_local_vol_surface.png)

### Strategy Backtester

Delta hedging and multi-leg strategy evaluation under simulated or historical market data.

- Delta hedging under **GBM** or **Heston** dynamics (with vol mismatch analysis)
- 8 pre-built strategy templates: Long/Short Straddle, Strangle, Bull Call Spread, Bear Put Spread, Butterfly, Iron Condor, Calendar Spread
- Per-leg expiry and volatility parameters for multi-expiry structures
- **Historical data backtesting** via CSV upload with automatic price column detection

![Strategy Backtester](screenshots/04_strategy_backtester.png)
![Heston Delta Hedge](screenshots/12_heston_delta_hedge.png)

### Scenario Simulator & Stress Testing

Portfolio-level what-if analysis across multiple risk dimensions.

- Spot x Vol P&L heatmaps
- 10 pre-configured stress scenarios (crash, rally, tail risk, time decay)
- Spot ladder and theta decay projections

### P&L Attribution

Full second-order Taylor decomposition for single positions and multi-position portfolios.

- Six Greek components: Delta, Gamma, Vega, Theta, **Vanna**, **Volga**
- Cumulative stacked area charts, daily breakdowns, Greeks evolution
- **Portfolio-level attribution** across multiple positions sharing the same underlying
- Summary statistics with annualized Sharpe ratio

![P&L Attribution](screenshots/15_pnl_vanna_volga.png)
![Portfolio P&L](screenshots/16_portfolio_pnl.png)

### Risk Dashboard

Real-time portfolio monitoring with VaR, Greeks exposure, and configurable alerts.

- Monte Carlo **VaR / CVaR** at configurable confidence levels (90%–99%)
- Greeks exposure breakdown by expiry bucket and strike bucket
- Configurable risk limits with breach notifications

![Risk Dashboard](screenshots/07_risk_dashboard.png)

## Quick Start

### With Docker (recommended)

```bash
cd quant-tool/
docker compose up --build -d
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### Without Docker

```bash
cd quant-tool/
pip install -r requirements.txt
streamlit run Home.py
```

### Stopping

```bash
docker compose down
```

## Project Structure

```
quant-tool/
├── Home.py                      # Entry point and navigation
├── pages/
│   ├── 1_Options_Pricer.py      # BS, Heston, Local Vol pricing + Greeks
│   ├── 2_Volatility_Surface.py  # 3D surface, SABR, Dupire local vol
│   ├── 3_Strategy_Backtester.py # Delta hedge, strategies, historical data
│   ├── 4_Scenario_Simulator.py  # Stress testing and P&L heatmaps
│   ├── 5_PnL_Attribution.py     # Single-position and portfolio P&L
│   └── 6_Risk_Dashboard.py      # VaR, Greeks exposure, risk alerts
├── core/
│   ├── pricing.py               # BS, Heston, Local Vol pricing engines
│   ├── greeks.py                # Analytical and numerical Greeks
│   ├── volatility.py            # SABR, vol surface, Dupire local vol
│   ├── backtesting.py           # GBM/Heston simulation, strategy engines
│   ├── scenarios.py             # Scenario and stress testing
│   └── pnl.py                   # P&L attribution engine
├── screenshots/                 # UI screenshots for documentation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

The `core/` modules are independent of Streamlit and can be imported into Jupyter notebooks or used as a standalone library.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| Web Framework | Streamlit |
| Numerical | NumPy, SciPy |
| Data | Pandas |
| Visualization | Plotly |
| Containerization | Docker |


