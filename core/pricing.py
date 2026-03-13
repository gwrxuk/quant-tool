"""
Options pricing models: Black-Scholes and Heston stochastic volatility.
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.integrate import quad


def bs_price(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K * np.exp(-r * max(T, 0)), 0.0)
        return max(K * np.exp(-r * max(T, 0)) - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_price_vec(S, K, T, r, sigma, option_type="call"):
    S, K, T, r, sigma = (np.asarray(x, dtype=float) for x in (S, K, T, r, sigma))
    mask = (T > 0) & (sigma > 0)
    prices = np.where(
        option_type == "call",
        np.maximum(S - K * np.exp(-r * np.maximum(T, 0)), 0.0),
        np.maximum(K * np.exp(-r * np.maximum(T, 0)) - S, 0.0),
    )
    if np.any(mask):
        sqrt_T = np.sqrt(np.where(mask, T, 1.0))
        sig_sqrt = sigma * sqrt_T
        d1 = np.where(mask, (np.log(S / K) + (r + 0.5 * sigma**2) * T) / sig_sqrt, 0)
        d2 = d1 - sig_sqrt
        if option_type == "call":
            prices = np.where(
                mask,
                S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
                prices,
            )
        else:
            prices = np.where(
                mask,
                K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1),
                prices,
            )
    return prices


def implied_vol(price, S, K, T, r, option_type="call"):
    if T <= 0:
        return np.nan
    intrinsic = max(S - K * np.exp(-r * T), 0) if option_type == "call" else max(K * np.exp(-r * T) - S, 0)
    if price <= intrinsic + 1e-10:
        return np.nan
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - price
    try:
        return brentq(objective, 1e-6, 10.0, xtol=1e-12)
    except (ValueError, RuntimeError):
        return np.nan


# --------------- Heston Model ---------------

def heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma_v, rho):
    """Heston characteristic function (Albrecher et al. formulation for stability)."""
    i = 1j
    xi = kappa - rho * sigma_v * i * u
    d = np.sqrt(xi**2 + sigma_v**2 * (i * u + u**2))
    g = (xi - d) / (xi + d)
    C = (r * i * u * T
         + kappa * theta / sigma_v**2
         * ((xi - d) * T - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))))
    D = (xi - d) / sigma_v**2 * (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    return np.exp(C + D * v0 + i * u * np.log(S))


def heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type="call"):
    """Heston model price via numerical integration of the characteristic function."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    def integrand_P(u, j):
        if j == 1:
            phi = heston_char_func(u - 1j, S, K, T, r, v0, kappa, theta, sigma_v, rho)
            phi /= heston_char_func(-1j, S, K, T, r, v0, kappa, theta, sigma_v, rho)
        else:
            phi = heston_char_func(u, S, K, T, r, v0, kappa, theta, sigma_v, rho)
        return np.real(np.exp(-1j * u * np.log(K)) * phi / (1j * u))

    P1 = 0.5 + 1.0 / np.pi * quad(lambda u: integrand_P(u, 1), 1e-8, 200, limit=500)[0]
    P2 = 0.5 + 1.0 / np.pi * quad(lambda u: integrand_P(u, 2), 1e-8, 200, limit=500)[0]

    call_price = S * P1 - K * np.exp(-r * T) * P2
    if option_type == "call":
        return max(call_price, 0.0)
    return max(call_price - S + K * np.exp(-r * T), 0.0)


def heston_price_grid(S, strikes, T, r, v0, kappa, theta, sigma_v, rho, option_type="call"):
    return np.array([
        heston_price(S, K, T, r, v0, kappa, theta, sigma_v, rho, option_type)
        for K in strikes
    ])


# --------------- Local Volatility Monte Carlo ---------------

def local_vol_mc_price(S, K, T, r, local_vol_func, option_type="call",
                       n_paths=10000, n_steps=100, seed=None):
    """
    Monte Carlo pricing under local volatility dynamics:
        dS = r*S*dt + sigma_local(S,t)*S*dW
    local_vol_func(S_val, t_val) -> local vol at given spot and time.
    Returns (price, std_error).
    """
    if seed is not None:
        np.random.seed(seed)
    if T <= 0:
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return intrinsic, 0.0

    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    S_paths = np.full(n_paths, float(S))

    for step in range(n_steps):
        t = step * dt
        Z = np.random.standard_normal(n_paths)
        sigma_local = np.array([local_vol_func(s, t) for s in S_paths])
        sigma_local = np.clip(sigma_local, 0.01, 5.0)
        S_paths *= np.exp((r - 0.5 * sigma_local**2) * dt + sigma_local * sqrt_dt * Z)

    if option_type == "call":
        payoffs = np.maximum(S_paths - K, 0)
    else:
        payoffs = np.maximum(K - S_paths, 0)

    price = np.exp(-r * T) * np.mean(payoffs)
    std_err = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_paths)
    return price, std_err
