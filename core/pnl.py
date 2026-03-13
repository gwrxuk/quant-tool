"""
P&L attribution via Taylor expansion of option price (Greeks-based decomposition).
"""
import numpy as np
import pandas as pd
from core.pricing import bs_price
from core.greeks import (delta as bs_delta, gamma as bs_gamma, vega as bs_vega,
                         theta as bs_theta, vanna as bs_vanna, volga as bs_volga)


def daily_pnl_attribution(S_series, sigma_series, K, T_start, r, option_type="call", quantity=1):
    """
    Decompose daily P&L into Delta, Gamma, Vega, Theta, Vanna, Volga,
    and Unexplained components using a second-order Taylor expansion.

    S_series: array of daily spot prices (length N+1 for N days).
    sigma_series: array of daily implied vols (length N+1).
    """
    n_days = len(S_series) - 1
    dt = T_start / max(n_days, 1)
    records = []

    for i in range(n_days):
        S_t = S_series[i]
        S_next = S_series[i + 1]
        sigma_t = sigma_series[i]
        sigma_next = sigma_series[i + 1]
        T_t = T_start - i * dt

        if T_t <= 0:
            break

        dS = S_next - S_t
        dsigma = sigma_next - sigma_t

        d = bs_delta(S_t, K, T_t, r, sigma_t, option_type)
        g = bs_gamma(S_t, K, T_t, r, sigma_t)
        v = bs_vega(S_t, K, T_t, r, sigma_t) * 100
        th = bs_theta(S_t, K, T_t, r, sigma_t, option_type) * 365
        va = bs_vanna(S_t, K, T_t, r, sigma_t)
        vo = bs_volga(S_t, K, T_t, r, sigma_t)

        delta_pnl = d * dS * quantity
        gamma_pnl = 0.5 * g * dS**2 * quantity
        vega_pnl = v * dsigma * quantity
        theta_pnl = th * (-dt) * quantity
        vanna_pnl = va * dS * dsigma * quantity
        volga_pnl = 0.5 * vo * dsigma**2 * quantity

        actual_prev = bs_price(S_t, K, T_t, r, sigma_t, option_type) * quantity
        T_next = max(T_t - dt, 0)
        actual_next = bs_price(S_next, K, T_next, r, sigma_next, option_type) * quantity
        actual_pnl = actual_next - actual_prev

        explained = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + vanna_pnl + volga_pnl
        unexplained = actual_pnl - explained

        records.append({
            "day": i + 1,
            "spot": S_next,
            "implied_vol": sigma_next,
            "actual_pnl": actual_pnl,
            "delta_pnl": delta_pnl,
            "gamma_pnl": gamma_pnl,
            "vega_pnl": vega_pnl,
            "theta_pnl": theta_pnl,
            "vanna_pnl": vanna_pnl,
            "volga_pnl": volga_pnl,
            "explained_pnl": explained,
            "unexplained_pnl": unexplained,
            "delta": d * quantity,
            "gamma": g * quantity,
            "vega": v / 100 * quantity,
            "theta": th / 365 * quantity,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df["cumulative_actual"] = df["actual_pnl"].cumsum()
        df["cumulative_delta"] = df["delta_pnl"].cumsum()
        df["cumulative_gamma"] = df["gamma_pnl"].cumsum()
        df["cumulative_vega"] = df["vega_pnl"].cumsum()
        df["cumulative_theta"] = df["theta_pnl"].cumsum()
        df["cumulative_vanna"] = df["vanna_pnl"].cumsum()
        df["cumulative_volga"] = df["volga_pnl"].cumsum()
        df["cumulative_unexplained"] = df["unexplained_pnl"].cumsum()
    return df


def generate_sample_pnl_data(S0=100, K=100, T=0.25, r=0.05, sigma=0.20,
                              drift_vol=0.002, spot_vol=0.015, n_days=60, seed=42):
    """Generate synthetic spot and vol paths for P&L attribution demo."""
    np.random.seed(seed)
    dt = 1 / 252
    S_series = [S0]
    sigma_series = [sigma]
    for _ in range(n_days):
        dS = S_series[-1] * (r * dt + spot_vol * np.random.randn())
        S_series.append(S_series[-1] + dS)
        dsigma = drift_vol * np.random.randn()
        sigma_series.append(max(sigma_series[-1] + dsigma, 0.05))
    return np.array(S_series), np.array(sigma_series)


def pnl_summary(df):
    """Create summary statistics from P&L attribution DataFrame."""
    if df.empty:
        return {}
    components = ["actual_pnl", "delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl",
                  "vanna_pnl", "volga_pnl", "unexplained_pnl"]
    summary = {}
    for col in components:
        if col not in df.columns:
            continue
        label = col.replace("_pnl", "").replace("_", " ").title()
        summary[label] = {
            "Total": df[col].sum(),
            "Mean": df[col].mean(),
            "Std": df[col].std(),
            "Min": df[col].min(),
            "Max": df[col].max(),
            "Sharpe": df[col].mean() / df[col].std() * np.sqrt(252) if df[col].std() > 0 else 0,
        }
    return pd.DataFrame(summary).T


def portfolio_pnl_attribution(positions, S0=100, n_days=60, spot_vol=0.015,
                               iv_drift=0.002, r=0.05, seed=42):
    """
    Portfolio-level P&L attribution across multiple positions on the same underlying.
    positions: list of dicts with K, T, sigma, option_type, quantity.
    Returns (aggregate_df, list_of_per_position_dfs).
    """
    np.random.seed(seed)

    S_series = [S0]
    dt = 1 / 252
    for _ in range(n_days):
        dS = S_series[-1] * (r * dt + spot_vol * np.random.randn())
        S_series.append(S_series[-1] + dS)
    S_series = np.array(S_series)

    per_position_dfs = []
    for idx, pos in enumerate(positions):
        np.random.seed(seed + idx * 137)
        sigma0 = pos["sigma"]
        sigma_series = [sigma0]
        for _ in range(n_days):
            dsigma = iv_drift * np.random.randn()
            sigma_series.append(max(sigma_series[-1] + dsigma, 0.05))
        sigma_series = np.array(sigma_series)

        df = daily_pnl_attribution(
            S_series, sigma_series, pos["K"], pos["T"], pos.get("r", r),
            pos["option_type"], pos["quantity"],
        )
        df["label"] = f"{pos['option_type'].upper()} K={pos['K']} Q={pos['quantity']}"
        per_position_dfs.append(df)

    if not per_position_dfs or all(d.empty for d in per_position_dfs):
        return pd.DataFrame(), []

    agg = per_position_dfs[0][["day", "spot"]].copy()
    pnl_cols = ["actual_pnl", "delta_pnl", "gamma_pnl", "vega_pnl", "theta_pnl",
                "vanna_pnl", "volga_pnl", "unexplained_pnl"]
    for col in pnl_cols:
        agg[col] = sum(
            d[col].values for d in per_position_dfs
            if col in d.columns and not d.empty
        )

    cum_map = {"actual_pnl": "cumulative_actual", "delta_pnl": "cumulative_delta",
               "gamma_pnl": "cumulative_gamma", "vega_pnl": "cumulative_vega",
               "theta_pnl": "cumulative_theta", "vanna_pnl": "cumulative_vanna",
               "volga_pnl": "cumulative_volga", "unexplained_pnl": "cumulative_unexplained"}
    for src, dst in cum_map.items():
        if src in agg.columns:
            agg[dst] = agg[src].cumsum()

    return agg, per_position_dfs
