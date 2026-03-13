"""
Volatility surface construction, SABR model calibration, implied vol utilities.
"""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.interpolate import RectBivariateSpline
from core.pricing import bs_price, implied_vol


# --------------- SABR Model ---------------

def sabr_implied_vol(F, K, T, alpha, beta, rho, nu):
    """
    Hagan et al. (2002) SABR implied volatility approximation.
    F: forward price, K: strike, T: time to expiry.
    """
    if F <= 0 or K <= 0 or T <= 0 or alpha <= 0:
        return np.nan

    if abs(F - K) < 1e-10:
        FK_mid = F
        A = alpha / FK_mid ** (1 - beta)
        B1 = ((1 - beta) ** 2 / 24) * alpha**2 / FK_mid ** (2 - 2 * beta)
        B2 = 0.25 * rho * beta * nu * alpha / FK_mid ** (1 - beta)
        B3 = (2 - 3 * rho**2) / 24 * nu**2
        return A * (1 + (B1 + B2 + B3) * T)

    FK = F * K
    log_FK = np.log(F / K)
    FK_beta = FK ** ((1 - beta) / 2)

    z = (nu / alpha) * FK_beta * log_FK
    x_z = np.log((np.sqrt(1 - 2 * rho * z + z**2) + z - rho) / (1 - rho))

    if abs(x_z) < 1e-12:
        zx_ratio = 1.0
    else:
        zx_ratio = z / x_z

    prefix = alpha / (FK_beta * (1 + (1 - beta)**2 / 24 * log_FK**2
                                  + (1 - beta)**4 / 1920 * log_FK**4))

    correction = 1 + (
        (1 - beta)**2 / 24 * alpha**2 / FK ** (1 - beta)
        + 0.25 * rho * beta * nu * alpha / FK_beta
        + (2 - 3 * rho**2) / 24 * nu**2
    ) * T

    return prefix * zx_ratio * correction


def sabr_calibrate(F, strikes, market_vols, T, beta=0.5):
    """
    Calibrate SABR parameters (alpha, rho, nu) given market implied vols.
    Beta is typically fixed (0, 0.5, or 1).
    """
    def objective(params):
        alpha, rho, nu = params
        if alpha <= 0 or nu <= 0 or abs(rho) >= 1:
            return 1e10
        model_vols = np.array([
            sabr_implied_vol(F, K, T, alpha, beta, rho, nu) for K in strikes
        ])
        if np.any(np.isnan(model_vols)):
            return 1e10
        return np.sum((model_vols - market_vols) ** 2)

    atm_vol = np.interp(F, strikes, market_vols)
    alpha0 = atm_vol * F ** (1 - beta)

    result = minimize(
        objective,
        x0=[alpha0, -0.2, 0.3],
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12},
    )
    alpha, rho, nu = result.x
    rho = np.clip(rho, -0.999, 0.999)
    return {"alpha": alpha, "beta": beta, "rho": rho, "nu": max(nu, 1e-6), "rmse": np.sqrt(result.fun / len(strikes))}


# --------------- Volatility Surface ---------------

def build_vol_surface(strikes, expiries, vol_matrix):
    """
    Construct a volatility surface via bivariate cubic spline interpolation.
    strikes: 1D array, expiries: 1D array (years), vol_matrix: 2D (len(expiries) x len(strikes)).
    Returns interpolator function(strike, expiry) -> vol.
    """
    spline = RectBivariateSpline(expiries, strikes, vol_matrix, kx=3, ky=3)

    def interpolator(K, T):
        return float(spline(T, K)[0, 0])

    return interpolator


def generate_synthetic_surface(S=100, r=0.05, base_vol=0.20, n_strikes=15, n_expiries=8):
    """
    Generate a realistic synthetic vol surface with skew and term structure.
    Returns strikes, expiries (years), vol_matrix.
    """
    strikes = np.linspace(0.7 * S, 1.3 * S, n_strikes)
    expiries = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0, 3.0])[:n_expiries]

    vol_matrix = np.zeros((len(expiries), len(strikes)))
    for i, T in enumerate(expiries):
        for j, K in enumerate(strikes):
            moneyness = np.log(K / S)
            skew = -0.15 * moneyness
            smile = 0.4 * moneyness**2
            term_adj = -0.02 * np.sqrt(T) + 0.01
            vol_matrix[i, j] = base_vol + skew + smile + term_adj + np.random.normal(0, 0.003)
            vol_matrix[i, j] = max(vol_matrix[i, j], 0.05)

    return strikes, expiries, vol_matrix


def compute_implied_vol_surface(S, strikes, expiries, prices_matrix, r, option_type="call"):
    """Convert a matrix of option prices to implied volatilities."""
    n_exp = len(expiries)
    n_str = len(strikes)
    iv_matrix = np.full((n_exp, n_str), np.nan)
    for i in range(n_exp):
        for j in range(n_str):
            iv_matrix[i, j] = implied_vol(prices_matrix[i, j], S, strikes[j], expiries[i], r, option_type)
    return iv_matrix


# --------------- Dupire Local Volatility ---------------

def dupire_local_vol(strikes, expiries, vol_matrix, S, r):
    """
    Extract Dupire local volatility surface from implied volatility surface.
    Applies Dupire's formula via finite differences on the BS call price surface.

    Returns local_vol_matrix of same shape as vol_matrix.
    """
    n_exp, n_str = vol_matrix.shape

    call_prices = np.zeros_like(vol_matrix)
    for i in range(n_exp):
        for j in range(n_str):
            call_prices[i, j] = bs_price(S, strikes[j], expiries[i], r,
                                         max(vol_matrix[i, j], 0.01), "call")

    local_vol = np.full_like(vol_matrix, np.nan)

    for i in range(1, n_exp - 1):
        dT = expiries[i + 1] - expiries[i - 1]
        for j in range(1, n_str - 1):
            dK = strikes[j + 1] - strikes[j - 1]
            dK_half = dK / 2.0

            dC_dT = (call_prices[i + 1, j] - call_prices[i - 1, j]) / dT
            dC_dK = (call_prices[i, j + 1] - call_prices[i, j - 1]) / dK
            d2C_dK2 = (call_prices[i, j + 1] - 2 * call_prices[i, j]
                       + call_prices[i, j - 1]) / (dK_half ** 2)

            numerator = dC_dT + r * strikes[j] * dC_dK
            denominator = 0.5 * strikes[j] ** 2 * d2C_dK2

            if denominator > 1e-10 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)

    if n_exp > 2:
        local_vol[0, :] = local_vol[1, :]
        local_vol[-1, :] = local_vol[-2, :]
    if n_str > 2:
        local_vol[:, 0] = local_vol[:, 1]
        local_vol[:, -1] = local_vol[:, -2]

    base_vol = np.nanmedian(vol_matrix)
    local_vol = np.where(np.isnan(local_vol), base_vol, local_vol)
    local_vol = np.clip(local_vol, 0.01, 5.0)

    return local_vol


def local_vol_surface_interpolator(strikes, expiries, local_vol_matrix):
    """
    Create a callable interpolator for the local vol surface.
    Returns a function f(S_val, t_val) -> local_vol.
    """
    spline = RectBivariateSpline(
        expiries, strikes, local_vol_matrix,
        kx=min(3, len(expiries) - 1), ky=min(3, len(strikes) - 1),
    )

    def interpolator(S_val, t_val):
        t_val = np.clip(t_val, expiries[0], expiries[-1])
        S_val = np.clip(S_val, strikes[0], strikes[-1])
        return float(np.clip(spline(t_val, S_val)[0, 0], 0.01, 5.0))

    return interpolator
