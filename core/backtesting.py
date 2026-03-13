"""
Strategy backtesting engine for options trading strategies.
Supports delta hedging, straddle, strangle, butterfly, and custom strategies.
"""
import numpy as np
import pandas as pd
from core.pricing import bs_price
from core.greeks import delta as bs_delta, gamma as bs_gamma, vega as bs_vega, theta as bs_theta


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=1, seed=None):
    """Geometric Brownian Motion simulation."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.standard_normal((n_steps, n_paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_returns, axis=0)])
    return S0 * np.exp(log_prices)


def simulate_heston_paths(S0, v0, mu, kappa, theta, sigma_v, rho, T, n_steps, n_paths=1, seed=None):
    """Heston stochastic volatility path simulation via Euler discretization."""
    if seed is not None:
        np.random.seed(seed)
    dt = T / n_steps
    S = np.zeros((n_steps + 1, n_paths))
    v = np.zeros((n_steps + 1, n_paths))
    S[0] = S0
    v[0] = v0
    for t in range(n_steps):
        Z1 = np.random.standard_normal(n_paths)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_paths)
        v_pos = np.maximum(v[t], 0)
        v[t + 1] = v[t] + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos * dt) * Z2
        v[t + 1] = np.maximum(v[t + 1], 0)
        S[t + 1] = S[t] * np.exp((mu - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z1)
    return S, v


def backtest_delta_hedge(S0, K, T, r, sigma, option_type="call", n_steps=252,
                          hedge_freq=1, seed=42):
    """
    Backtest a delta-hedging strategy for a short option position.
    Returns a DataFrame with daily P&L breakdown.
    """
    paths = simulate_gbm(S0, r, sigma, T, n_steps, n_paths=1, seed=seed)
    spot = paths[:, 0]
    dt = T / n_steps
    times = np.linspace(T, 0, n_steps + 1)

    records = []
    option_prem = bs_price(S0, K, T, r, sigma, option_type)
    cash = option_prem
    shares = 0.0

    for i in range(n_steps):
        S_t = spot[i]
        t_remain = times[i]

        if i % hedge_freq == 0:
            new_delta = bs_delta(S_t, K, t_remain, r, sigma, option_type)
            trade = new_delta - shares
            cash -= trade * S_t
            shares = new_delta

        g = bs_gamma(S_t, K, t_remain, r, sigma)
        v = bs_vega(S_t, K, t_remain, r, sigma)
        th = bs_theta(S_t, K, t_remain, r, sigma, option_type)

        dS = spot[i + 1] - spot[i]
        option_new = bs_price(spot[i + 1], K, times[i + 1], r, sigma, option_type)
        option_old = bs_price(S_t, K, t_remain, r, sigma, option_type)
        option_pnl = -(option_new - option_old)
        hedge_pnl = shares * dS
        interest = cash * r * dt
        cash += interest
        daily_pnl = hedge_pnl + option_pnl + interest

        records.append({
            "day": i + 1,
            "spot": spot[i + 1],
            "delta": shares,
            "gamma": g,
            "option_value": option_new,
            "hedge_pnl": hedge_pnl,
            "option_pnl": option_pnl,
            "interest": interest,
            "daily_pnl": daily_pnl,
        })

    df = pd.DataFrame(records)
    df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    return df


def backtest_strategy(S0, positions, T=0.25, r=0.05, sigma=0.20, n_steps=252, seed=42,
                      model="gbm", heston_params=None):
    """
    Backtest a multi-leg options strategy.
    positions: list of dicts with keys: K, option_type, quantity.
        Optional per-leg keys: T (own expiry), sigma (own vol).
    model: "gbm" or "heston".
    heston_params: dict with v0, kappa, theta, sigma_v, rho (for Heston dynamics).
    """
    resolved = []
    for p in positions:
        rp = dict(p)
        rp.setdefault("T", T)
        rp.setdefault("sigma", sigma)
        resolved.append(rp)

    T_max = max(rp["T"] for rp in resolved)

    if model == "heston" and heston_params is not None:
        hp = heston_params
        spot_all, _ = simulate_heston_paths(
            S0, hp["v0"], r, hp["kappa"], hp["theta"], hp["sigma_v"], hp["rho"],
            T_max, n_steps, n_paths=1, seed=seed,
        )
        spot = spot_all[:, 0]
    else:
        paths = simulate_gbm(S0, r, sigma, T_max, n_steps, n_paths=1, seed=seed)
        spot = paths[:, 0]

    dt = T_max / n_steps

    initial_cost = sum(
        rp["quantity"] * bs_price(S0, rp["K"], rp["T"], r, rp["sigma"], rp["option_type"])
        for rp in resolved
    )

    records = []
    for i in range(n_steps):
        t_elapsed = (i + 1) * dt
        S_t = spot[i + 1]
        pv, p_delta, p_gamma, p_theta, p_vega = 0.0, 0.0, 0.0, 0.0, 0.0

        for rp in resolved:
            t_remain = max(rp["T"] - t_elapsed, 0)
            p_sig = rp["sigma"]

            if t_remain <= 1e-10:
                intrinsic = (max(S_t - rp["K"], 0) if rp["option_type"] == "call"
                             else max(rp["K"] - S_t, 0))
                pv += rp["quantity"] * intrinsic
            else:
                pv += rp["quantity"] * bs_price(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_delta += rp["quantity"] * bs_delta(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_gamma += rp["quantity"] * bs_gamma(S_t, rp["K"], t_remain, r, p_sig)
                p_theta += rp["quantity"] * bs_theta(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_vega += rp["quantity"] * bs_vega(S_t, rp["K"], t_remain, r, p_sig)

        records.append({
            "day": i + 1, "spot": S_t, "portfolio_value": pv, "pnl": pv - initial_cost,
            "delta": p_delta, "gamma": p_gamma, "theta": p_theta, "vega": p_vega,
        })

    df = pd.DataFrame(records)

    terminal_val = 0.0
    for rp in resolved:
        t_remain = max(rp["T"] - T_max, 0)
        if t_remain <= 1e-10:
            intrinsic = (max(spot[-1] - rp["K"], 0) if rp["option_type"] == "call"
                         else max(rp["K"] - spot[-1], 0))
            terminal_val += rp["quantity"] * intrinsic
        else:
            terminal_val += rp["quantity"] * bs_price(
                spot[-1], rp["K"], t_remain, r, rp["sigma"], rp["option_type"]
            )
    final_pnl = terminal_val - initial_cost

    return df, initial_cost, final_pnl


STRATEGY_TEMPLATES = {
    "Long Straddle": lambda S, K_offset=0: [
        {"K": S + K_offset, "option_type": "call", "quantity": 1},
        {"K": S + K_offset, "option_type": "put", "quantity": 1},
    ],
    "Short Straddle": lambda S, K_offset=0: [
        {"K": S + K_offset, "option_type": "call", "quantity": -1},
        {"K": S + K_offset, "option_type": "put", "quantity": -1},
    ],
    "Long Strangle": lambda S, width=5: [
        {"K": S + width, "option_type": "call", "quantity": 1},
        {"K": S - width, "option_type": "put", "quantity": 1},
    ],
    "Bull Call Spread": lambda S, width=5: [
        {"K": S, "option_type": "call", "quantity": 1},
        {"K": S + width, "option_type": "call", "quantity": -1},
    ],
    "Bear Put Spread": lambda S, width=5: [
        {"K": S, "option_type": "put", "quantity": 1},
        {"K": S - width, "option_type": "put", "quantity": -1},
    ],
    "Long Butterfly": lambda S, width=5: [
        {"K": S - width, "option_type": "call", "quantity": 1},
        {"K": S, "option_type": "call", "quantity": -2},
        {"K": S + width, "option_type": "call", "quantity": 1},
    ],
    "Iron Condor": lambda S, inner=3, outer=7: [
        {"K": S - outer, "option_type": "put", "quantity": 1},
        {"K": S - inner, "option_type": "put", "quantity": -1},
        {"K": S + inner, "option_type": "call", "quantity": -1},
        {"K": S + outer, "option_type": "call", "quantity": 1},
    ],
    "Calendar Spread": lambda S, K_offset=0, T_near=None, T_far=None: [
        {"K": S + K_offset, "option_type": "call", "quantity": -1,
         "T": T_near if T_near is not None else 1 / 12},
        {"K": S + K_offset, "option_type": "call", "quantity": 1,
         "T": T_far if T_far is not None else 0.25},
    ],
}


def backtest_delta_hedge_heston(S0, K, T, r, v0, kappa, theta_v, sigma_v, rho_h,
                                option_type="call", n_steps=252, hedge_freq=1,
                                hedge_vol=None, seed=42):
    """
    Delta-hedge backtest under Heston stochastic volatility dynamics.
    The underlying follows Heston, but hedging uses BS delta with hedge_vol,
    creating a realistic volatility mismatch.
    """
    S_paths, v_paths = simulate_heston_paths(
        S0, v0, r, kappa, theta_v, sigma_v, rho_h, T, n_steps, n_paths=1, seed=seed
    )
    spot = S_paths[:, 0]
    vol_path = np.sqrt(np.maximum(v_paths[:, 0], 0))
    dt = T / n_steps
    times = np.linspace(T, 0, n_steps + 1)

    hedge_sigma = hedge_vol if hedge_vol is not None else np.sqrt(v0)
    option_prem = bs_price(S0, K, T, r, hedge_sigma, option_type)
    cash = option_prem
    shares = 0.0

    records = []
    for i in range(n_steps):
        S_t = spot[i]
        t_remain = times[i]

        if i % hedge_freq == 0 and t_remain > 0:
            new_delta = bs_delta(S_t, K, t_remain, r, hedge_sigma, option_type)
            trade = new_delta - shares
            cash -= trade * S_t
            shares = new_delta

        g = bs_gamma(S_t, K, t_remain, r, hedge_sigma)

        dS = spot[i + 1] - spot[i]
        option_new = bs_price(spot[i + 1], K, times[i + 1], r, hedge_sigma, option_type)
        option_old = bs_price(S_t, K, t_remain, r, hedge_sigma, option_type)
        option_pnl = -(option_new - option_old)
        hedge_pnl = shares * dS
        interest = cash * r * dt
        cash += interest
        daily_pnl = hedge_pnl + option_pnl + interest

        records.append({
            "day": i + 1, "spot": spot[i + 1], "realized_vol": vol_path[i],
            "delta": shares, "gamma": g, "option_value": option_new,
            "hedge_pnl": hedge_pnl, "option_pnl": option_pnl,
            "interest": interest, "daily_pnl": daily_pnl,
        })

    df = pd.DataFrame(records)
    df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    return df


def backtest_strategy_from_data(spot_data, positions, r=0.05, sigma=0.20):
    """
    Backtest a multi-leg strategy using historical spot price data.
    spot_data: array-like of daily spot prices.
    positions: list of dicts with K, option_type, quantity, and optionally T, sigma.
    """
    spot = np.asarray(spot_data, dtype=float)
    n_steps = len(spot) - 1
    if n_steps <= 0:
        return pd.DataFrame(), 0, 0

    resolved = []
    for p in positions:
        rp = dict(p)
        rp.setdefault("T", n_steps / 252)
        rp.setdefault("sigma", sigma)
        resolved.append(rp)

    T_max = max(rp["T"] for rp in resolved)
    dt = T_max / n_steps

    initial_cost = sum(
        rp["quantity"] * bs_price(spot[0], rp["K"], rp["T"], r, rp["sigma"], rp["option_type"])
        for rp in resolved
    )

    records = []
    for i in range(n_steps):
        t_elapsed = (i + 1) * dt
        S_t = spot[i + 1]
        pv, p_delta, p_gamma, p_theta, p_vega = 0.0, 0.0, 0.0, 0.0, 0.0

        for rp in resolved:
            t_remain = max(rp["T"] - t_elapsed, 0)
            p_sig = rp["sigma"]

            if t_remain <= 1e-10:
                intrinsic = (max(S_t - rp["K"], 0) if rp["option_type"] == "call"
                             else max(rp["K"] - S_t, 0))
                pv += rp["quantity"] * intrinsic
            else:
                pv += rp["quantity"] * bs_price(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_delta += rp["quantity"] * bs_delta(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_gamma += rp["quantity"] * bs_gamma(S_t, rp["K"], t_remain, r, p_sig)
                p_theta += rp["quantity"] * bs_theta(S_t, rp["K"], t_remain, r, p_sig, rp["option_type"])
                p_vega += rp["quantity"] * bs_vega(S_t, rp["K"], t_remain, r, p_sig)

        records.append({
            "day": i + 1, "spot": S_t, "portfolio_value": pv, "pnl": pv - initial_cost,
            "delta": p_delta, "gamma": p_gamma, "theta": p_theta, "vega": p_vega,
        })

    df = pd.DataFrame(records)

    terminal_val = 0.0
    for rp in resolved:
        t_remain = max(rp["T"] - T_max, 0)
        if t_remain <= 1e-10:
            intrinsic = (max(spot[-1] - rp["K"], 0) if rp["option_type"] == "call"
                         else max(rp["K"] - spot[-1], 0))
            terminal_val += rp["quantity"] * intrinsic
        else:
            terminal_val += rp["quantity"] * bs_price(
                spot[-1], rp["K"], t_remain, r, rp["sigma"], rp["option_type"]
            )
    final_pnl = terminal_val - initial_cost

    return df, initial_cost, final_pnl
