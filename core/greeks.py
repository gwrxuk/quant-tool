"""
Analytical and numerical Greeks for Black-Scholes and general pricing models.
"""
import numpy as np
from scipy.stats import norm
from core.pricing import bs_price


def _d1d2(S, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def delta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return 1.0 if S > K else 0.0
        return -1.0 if S < K else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100  # per 1% vol move


def theta(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    common = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
    if option_type == "call":
        return (common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return (common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def rho(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        return 0.0
    _, d2 = _d1d2(S, K, T, r, sigma)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


def vanna(S, K, T, r, sigma):
    """dDelta/dVol = dVega/dSpot."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return -norm.pdf(d1) * d2 / sigma


def volga(S, K, T, r, sigma):
    """d²Price/dVol² (Vomma). Sensitivity of vega to vol."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) * d1 * d2 / sigma


def charm(S, K, T, r, sigma, option_type="call"):
    """dDelta/dT (Delta decay)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    term = pdf_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T))
    if option_type == "call":
        return -term / 365
    return -term / 365


def speed(S, K, T, r, sigma):
    """dGamma/dSpot."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    g = gamma(S, K, T, r, sigma)
    return -g / S * (d1 / (sigma * np.sqrt(T)) + 1)


def all_greeks(S, K, T, r, sigma, option_type="call"):
    return {
        "Delta": delta(S, K, T, r, sigma, option_type),
        "Gamma": gamma(S, K, T, r, sigma),
        "Vega": vega(S, K, T, r, sigma),
        "Theta": theta(S, K, T, r, sigma, option_type),
        "Rho": rho(S, K, T, r, sigma, option_type),
        "Vanna": vanna(S, K, T, r, sigma),
        "Volga": volga(S, K, T, r, sigma),
        "Charm": charm(S, K, T, r, sigma, option_type),
        "Speed": speed(S, K, T, r, sigma),
    }


def numerical_greeks(pricer, params, option_type="call", h_S=0.01, h_sigma=0.001, h_T=1/365):
    """Compute Greeks via finite differences for any pricing model."""
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    base_params = {**params, "option_type": option_type}

    price_0 = pricer(**base_params)

    # Delta
    p_up = pricer(**{**base_params, "S": S * (1 + h_S)})
    p_dn = pricer(**{**base_params, "S": S * (1 - h_S)})
    dlt = (p_up - p_dn) / (2 * S * h_S)

    # Gamma
    gma = (p_up - 2 * price_0 + p_dn) / (S * h_S) ** 2

    # Vega (per 1%)
    p_vu = pricer(**{**base_params, "sigma": sigma + h_sigma})
    p_vd = pricer(**{**base_params, "sigma": max(sigma - h_sigma, 1e-6)})
    vga = (p_vu - p_vd) / (2 * h_sigma) / 100

    # Theta (per day)
    if T > h_T:
        p_t = pricer(**{**base_params, "T": T - h_T})
        tht = (p_t - price_0) / 365
    else:
        tht = 0.0

    return {
        "Delta": dlt,
        "Gamma": gma,
        "Vega": vga,
        "Theta": tht,
        "Price": price_0,
    }
