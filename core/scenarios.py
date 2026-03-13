"""
Scenario simulator for options portfolio stress testing.
Bump-and-reprice across spot, volatility, and time dimensions.
"""
import numpy as np
import pandas as pd
from core.pricing import bs_price
from core.greeks import all_greeks


def build_portfolio_from_positions(positions):
    """
    positions: list of dicts with keys:
        S, K, T, r, sigma, option_type, quantity, [label]
    """
    return positions


def portfolio_value(positions, S_override=None, sigma_override=None, T_offset=0):
    """Compute total portfolio value with optional overrides."""
    total = 0
    for p in positions:
        S = S_override if S_override is not None else p["S"]
        sigma = sigma_override if sigma_override is not None else p["sigma"]
        T = max(p["T"] + T_offset, 0)
        total += p["quantity"] * bs_price(S, p["K"], T, p["r"], sigma, p["option_type"])
    return total


def portfolio_greeks(positions, S_override=None):
    """Aggregate Greeks across all positions."""
    agg = {"Delta": 0, "Gamma": 0, "Vega": 0, "Theta": 0, "Rho": 0, "Vanna": 0, "Volga": 0}
    for p in positions:
        S = S_override if S_override is not None else p["S"]
        g = all_greeks(S, p["K"], p["T"], p["r"], p["sigma"], p["option_type"])
        for k in agg:
            agg[k] += p["quantity"] * g[k]
    return agg


def spot_vol_scenario_grid(positions, spot_range=(-0.20, 0.20), vol_range=(-0.10, 0.10),
                            n_spot=21, n_vol=21):
    """
    Generate a P&L heatmap across spot and volatility bumps.
    Returns spot_bumps, vol_bumps, pnl_matrix.
    """
    base_val = portfolio_value(positions)
    base_S = positions[0]["S"]
    base_sigma = positions[0]["sigma"]

    spot_bumps = np.linspace(spot_range[0], spot_range[1], n_spot)
    vol_bumps = np.linspace(vol_range[0], vol_range[1], n_vol)
    pnl_matrix = np.zeros((n_vol, n_spot))

    for i, dv in enumerate(vol_bumps):
        for j, ds in enumerate(spot_bumps):
            new_S = base_S * (1 + ds)
            new_positions = []
            for p in positions:
                new_p = dict(p)
                new_p["S"] = new_S
                new_p["sigma"] = p["sigma"] + dv
                new_positions.append(new_p)
            pnl_matrix[i, j] = portfolio_value(new_positions) - base_val

    return spot_bumps * 100, vol_bumps * 100, pnl_matrix


def time_decay_scenario(positions, days_forward=30):
    """Compute portfolio value over time (theta decay analysis)."""
    base_val = portfolio_value(positions)
    days = np.arange(0, days_forward + 1)
    values = []
    greeks_over_time = []
    for d in days:
        T_offset = -d / 365
        val = portfolio_value(positions, T_offset=T_offset)
        values.append(val)

        shifted_positions = []
        for p in positions:
            sp = dict(p)
            sp["T"] = max(p["T"] + T_offset, 0)
            shifted_positions.append(sp)
        g = portfolio_greeks(shifted_positions)
        g["day"] = d
        g["value"] = val
        g["pnl"] = val - base_val
        greeks_over_time.append(g)

    return pd.DataFrame(greeks_over_time)


def spot_ladder(positions, pct_range=0.20, n_points=41):
    """P&L and Greeks as a function of spot price."""
    base_S = positions[0]["S"]
    base_val = portfolio_value(positions)
    spots = np.linspace(base_S * (1 - pct_range), base_S * (1 + pct_range), n_points)
    records = []
    for S in spots:
        shifted = [{**p, "S": S} for p in positions]
        val = portfolio_value(shifted)
        g = portfolio_greeks(shifted)
        records.append({
            "spot": S,
            "value": val,
            "pnl": val - base_val,
            **g,
        })
    return pd.DataFrame(records)


def stress_test_table(positions, scenarios=None):
    """
    Run named stress scenarios.
    scenarios: list of dicts with name, spot_pct, vol_pct, time_days.
    """
    if scenarios is None:
        scenarios = [
            {"name": "Base", "spot_pct": 0, "vol_pct": 0, "time_days": 0},
            {"name": "Spot -10%", "spot_pct": -10, "vol_pct": 0, "time_days": 0},
            {"name": "Spot +10%", "spot_pct": 10, "vol_pct": 0, "time_days": 0},
            {"name": "Vol +5%", "spot_pct": 0, "vol_pct": 5, "time_days": 0},
            {"name": "Vol -5%", "spot_pct": 0, "vol_pct": -5, "time_days": 0},
            {"name": "Crash: Spot -20%, Vol +15%", "spot_pct": -20, "vol_pct": 15, "time_days": 0},
            {"name": "Rally: Spot +20%, Vol -10%", "spot_pct": 20, "vol_pct": -10, "time_days": 0},
            {"name": "1 Week Decay", "spot_pct": 0, "vol_pct": 0, "time_days": 7},
            {"name": "1 Month Decay", "spot_pct": 0, "vol_pct": 0, "time_days": 30},
            {"name": "Tail Risk: -30%, Vol +25%", "spot_pct": -30, "vol_pct": 25, "time_days": 0},
        ]

    base_val = portfolio_value(positions)
    base_S = positions[0]["S"]
    results = []
    for sc in scenarios:
        new_S = base_S * (1 + sc["spot_pct"] / 100)
        T_offset = -sc["time_days"] / 365
        shifted = []
        for p in positions:
            sp = dict(p)
            sp["S"] = new_S
            sp["sigma"] = p["sigma"] + sc["vol_pct"] / 100
            sp["T"] = max(p["T"] + T_offset, 0)
            shifted.append(sp)
        val = portfolio_value(shifted)
        g = portfolio_greeks(shifted)
        results.append({
            "Scenario": sc["name"],
            "Portfolio Value": val,
            "P&L": val - base_val,
            "P&L %": (val - base_val) / abs(base_val) * 100 if base_val != 0 else 0,
            "Delta": g["Delta"],
            "Gamma": g["Gamma"],
            "Vega": g["Vega"],
        })
    return pd.DataFrame(results)
