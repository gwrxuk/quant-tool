"""
Index options analytics: SPX, NDX, RUT, VIX — cash-settled, European-style
pricing with settlement conventions, term structure, and skew analysis.
"""
import datetime as _dt
import numpy as np
import pandas as pd
from scipy.stats import norm
from core.pricing import bs_price, implied_vol

# ---------------------------------------------------------------------------
# Index contract specifications
# ---------------------------------------------------------------------------

INDEX_SPECS = {
    "SPX": {
        "name": "S&P 500 Index Options",
        "underlying": "^GSPC",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "AM",
        "ticker_options": "SPX",
        "mini_ticker": "XSP",
        "mini_multiplier": 10,
        "exchange": "CBOE",
        "section_1256": True,
        "description": "Flagship S&P 500 index options, AM-settled, European exercise",
    },
    "SPXW": {
        "name": "S&P 500 Weekly Options (PM)",
        "underlying": "^GSPC",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "PM",
        "ticker_options": "SPXW",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "Weekly/daily SPX options with PM settlement (closing price)",
    },
    "NDX": {
        "name": "Nasdaq-100 Index Options",
        "underlying": "^NDX",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "AM",
        "ticker_options": "NDX",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "Nasdaq-100 index options, AM-settled, European exercise",
    },
    "RUT": {
        "name": "Russell 2000 Index Options",
        "underlying": "^RUT",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "AM",
        "ticker_options": "RUT",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "Russell 2000 index options, AM-settled, European exercise",
    },
    "VIX": {
        "name": "VIX Index Options",
        "underlying": "^VIX",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "AM",
        "ticker_options": "VIX",
        "settles_to": "VIX_futures",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "VIX options settle to VIX Special Opening Quotation (SOQ), not spot VIX",
    },
    "XSP": {
        "name": "Mini-SPX Index Options",
        "underlying": "^GSPC",
        "multiplier": 10,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "PM",
        "ticker_options": "XSP",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "1/10th size SPX options for smaller positions",
    },
    "DJX": {
        "name": "Dow Jones Index Options",
        "underlying": "^DJI",
        "multiplier": 100,
        "style": "european",
        "settlement": "cash",
        "settlement_type": "AM",
        "ticker_options": "DJX",
        "exchange": "CBOE",
        "section_1256": True,
        "description": "Dow Jones Industrial Average 1/100th scale index options",
    },
}


def get_index_spec(symbol: str) -> dict:
    """Return the contract specification for a given index option symbol."""
    return INDEX_SPECS.get(symbol.upper(), {})


def list_index_symbols() -> list[str]:
    return list(INDEX_SPECS.keys())


# ---------------------------------------------------------------------------
# European option pricing (no early exercise premium)
# ---------------------------------------------------------------------------

def european_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes price for European index option with continuous dividend yield q.
    Index options use the Merton model (cost-of-carry = r - q).
    """
    if T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S * np.exp(-q * max(T, 0)) - K * np.exp(-r * max(T, 0)), 0.0)
        return max(K * np.exp(-r * max(T, 0)) - S * np.exp(-q * max(T, 0)), 0.0)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def european_greeks(S, K, T, r, sigma, option_type="call", q=0.0):
    """Full Greek suite for European index options with dividend yield."""
    if T <= 0 or sigma <= 0:
        intrinsic_call = max(S - K, 0)
        return {
            "Delta": (1.0 if S > K else 0.0) if option_type == "call" else (-1.0 if S < K else 0.0),
            "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0,
        }

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)
    div_disc = np.exp(-q * T)

    gamma_val = div_disc * pdf_d1 / (S * sigma * sqrt_T)
    vega_val = S * div_disc * pdf_d1 * sqrt_T / 100

    if option_type == "call":
        delta_val = div_disc * norm.cdf(d1)
        theta_val = (
            -S * div_disc * pdf_d1 * sigma / (2 * sqrt_T)
            + q * S * div_disc * norm.cdf(d1)
            - r * K * disc * norm.cdf(d2)
        ) / 365
        rho_val = K * T * disc * norm.cdf(d2) / 100
    else:
        delta_val = -div_disc * norm.cdf(-d1)
        theta_val = (
            -S * div_disc * pdf_d1 * sigma / (2 * sqrt_T)
            - q * S * div_disc * norm.cdf(-d1)
            + r * K * disc * norm.cdf(-d2)
        ) / 365
        rho_val = -K * T * disc * norm.cdf(-d2) / 100

    return {
        "Delta": delta_val,
        "Gamma": gamma_val,
        "Vega": vega_val,
        "Theta": theta_val,
        "Rho": rho_val,
    }


# ---------------------------------------------------------------------------
# Put-call parity for European index options
# ---------------------------------------------------------------------------

def put_call_parity_check(call_price, put_price, S, K, T, r, q=0.0):
    """
    Check put-call parity: C - P = S*exp(-qT) - K*exp(-rT).
    Returns dict with theoretical spread, actual spread, and violation amount.
    """
    theoretical = S * np.exp(-q * T) - K * np.exp(-r * T)
    actual = call_price - put_price
    violation = actual - theoretical

    return {
        "call_price": call_price,
        "put_price": put_price,
        "theoretical_spread": theoretical,
        "actual_spread": actual,
        "violation": violation,
        "violation_pct": abs(violation) / S * 100 if S > 0 else np.nan,
        "parity_holds": abs(violation) < 0.50,
    }


def implied_dividend_yield(call_price, put_price, S, K, T, r):
    """
    Back out the implied continuous dividend yield from put-call parity.
    C - P = S*exp(-qT) - K*exp(-rT) => q = -ln((C - P + K*exp(-rT)) / S) / T
    """
    if T <= 0 or S <= 0:
        return np.nan
    forward_component = call_price - put_price + K * np.exp(-r * T)
    if forward_component <= 0:
        return np.nan
    return -np.log(forward_component / S) / T


# ---------------------------------------------------------------------------
# Settlement value computation
# ---------------------------------------------------------------------------

def cash_settlement_value(settlement_price, K, option_type, multiplier=100):
    """Compute cash settlement for an exercised index option."""
    if option_type == "call":
        intrinsic = max(settlement_price - K, 0)
    else:
        intrinsic = max(K - settlement_price, 0)
    return intrinsic * multiplier


def breakeven_price(premium, K, option_type, multiplier=100):
    """Breakeven index level at expiration given premium paid."""
    per_point = premium / multiplier
    if option_type == "call":
        return K + per_point
    return K - per_point


# ---------------------------------------------------------------------------
# Term structure analysis
# ---------------------------------------------------------------------------

def compute_term_structure(spot, strikes_by_expiry, iv_by_expiry, expiries_T):
    """
    Compute ATM implied volatility term structure from multiple expiry chains.

    strikes_by_expiry: list of strike arrays (one per expiry)
    iv_by_expiry: list of IV arrays (one per expiry)
    expiries_T: list of time-to-expiry in years

    Returns DataFrame with columns: T, days, atm_iv, atm_strike
    """
    rows = []
    for T, strikes, ivs in zip(expiries_T, strikes_by_expiry, iv_by_expiry):
        if len(strikes) == 0 or len(ivs) == 0:
            continue
        strikes = np.asarray(strikes, dtype=float)
        ivs = np.asarray(ivs, dtype=float)

        valid = ~np.isnan(ivs) & (ivs > 0.001)
        if valid.sum() < 1:
            continue

        atm_idx = np.argmin(np.abs(strikes[valid] - spot))
        rows.append({
            "T": T,
            "days": int(T * 365),
            "atm_iv": float(ivs[valid][atm_idx]),
            "atm_strike": float(strikes[valid][atm_idx]),
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["T", "days", "atm_iv", "atm_strike"])


def compute_forward_vol(iv1, T1, iv2, T2):
    """
    Compute forward implied volatility between two expiries.
    sigma_fwd = sqrt((sigma2^2 * T2 - sigma1^2 * T1) / (T2 - T1))
    """
    if T2 <= T1:
        return np.nan
    var_diff = iv2**2 * T2 - iv1**2 * T1
    if var_diff < 0:
        return np.nan
    return np.sqrt(var_diff / (T2 - T1))


def term_structure_with_forwards(term_df):
    """
    Augment a term structure DataFrame with forward volatilities.
    Input df must have columns: T, atm_iv.
    """
    if term_df.empty or len(term_df) < 2:
        return term_df

    df = term_df.sort_values("T").copy()
    fwd_vols = [np.nan]
    for i in range(1, len(df)):
        fwd = compute_forward_vol(
            df.iloc[i - 1]["atm_iv"], df.iloc[i - 1]["T"],
            df.iloc[i]["atm_iv"], df.iloc[i]["T"],
        )
        fwd_vols.append(fwd)
    df["forward_vol"] = fwd_vols
    return df


# ---------------------------------------------------------------------------
# Skew analysis
# ---------------------------------------------------------------------------

def compute_skew_metrics(strikes, ivs, spot, T):
    """
    Compute standard skew metrics for an option chain at a single expiry.

    Returns dict with: 25d_skew, risk_reversal_25d, butterfly_25d,
    skew_slope, smile_curvature, atm_iv.
    """
    strikes = np.asarray(strikes, dtype=float)
    ivs = np.asarray(ivs, dtype=float)
    valid = ~np.isnan(ivs) & (ivs > 0.001)
    if valid.sum() < 5:
        return {}

    strikes = strikes[valid]
    ivs = ivs[valid]
    moneyness = np.log(strikes / spot)

    atm_idx = np.argmin(np.abs(moneyness))
    atm_iv = ivs[atm_idx]

    otm_put_region = moneyness < -0.02
    otm_call_region = moneyness > 0.02

    put_25d_iv = np.nan
    call_25d_iv = np.nan

    if np.any(otm_put_region):
        target_m = -0.25 * atm_iv * np.sqrt(T) if T > 0 else -0.05
        put_25d_iv = float(np.interp(target_m, moneyness, ivs))

    if np.any(otm_call_region):
        target_m = 0.25 * atm_iv * np.sqrt(T) if T > 0 else 0.05
        call_25d_iv = float(np.interp(target_m, moneyness, ivs))

    skew_25d = put_25d_iv - call_25d_iv if not (np.isnan(put_25d_iv) or np.isnan(call_25d_iv)) else np.nan
    rr_25d = call_25d_iv - put_25d_iv if not np.isnan(skew_25d) else np.nan
    bf_25d = 0.5 * (put_25d_iv + call_25d_iv) - atm_iv if not (np.isnan(put_25d_iv) or np.isnan(call_25d_iv)) else np.nan

    if len(moneyness) >= 3:
        coeffs = np.polyfit(moneyness, ivs, 2)
        curvature = coeffs[0]
        slope = coeffs[1]
    else:
        slope, curvature = np.nan, np.nan

    return {
        "atm_iv": atm_iv,
        "put_25d_iv": put_25d_iv,
        "call_25d_iv": call_25d_iv,
        "skew_25d": skew_25d,
        "risk_reversal_25d": rr_25d,
        "butterfly_25d": bf_25d,
        "skew_slope": slope,
        "smile_curvature": curvature,
    }


# ---------------------------------------------------------------------------
# VIX-specific helpers
# ---------------------------------------------------------------------------

def vix_futures_price_approation(vix_spot, T, kappa=5.0, theta=20.0):
    """
    Approximate VIX futures price using mean-reversion model.
    F(T) = theta + (VIX_spot - theta) * exp(-kappa * T)
    """
    return theta + (vix_spot - theta) * np.exp(-kappa * T)


def vix_option_price(F, K, T, r, sigma, option_type="call"):
    """
    Price VIX options using Black '76 model on VIX futures.
    VIX options settle to VIX futures, not spot VIX.
    """
    if T <= 0 or sigma <= 0 or F <= 0:
        if option_type == "call":
            return max(F - K, 0.0)
        return max(K - F, 0.0)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    disc = np.exp(-r * T)

    if option_type == "call":
        return disc * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return disc * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def vix_term_structure(vix_spot, n_months=8, kappa=5.0, theta=20.0):
    """Generate approximate VIX futures term structure."""
    rows = []
    for m in range(1, n_months + 1):
        T = m / 12
        F = vix_futures_price_approximation(vix_spot, T, kappa, theta)
        rows.append({"month": m, "T": T, "futures_price": F})
    return pd.DataFrame(rows)


def vix_futures_price_approximation(vix_spot, T, kappa=5.0, theta=20.0):
    """Corrected name for VIX futures approximation."""
    return theta + (vix_spot - theta) * np.exp(-kappa * T)


# ---------------------------------------------------------------------------
# Expiry calendar helpers
# ---------------------------------------------------------------------------

def next_monthly_expiry(from_date=None):
    """Third Friday of the current or next month."""
    d = from_date or _dt.date.today()
    year, month = d.year, d.month
    first = _dt.date(year, month, 1)
    day_of_week = first.weekday()
    third_friday = first + _dt.timedelta(days=(4 - day_of_week) % 7 + 14)
    if third_friday <= d:
        month += 1
        if month > 12:
            month, year = 1, year + 1
        first = _dt.date(year, month, 1)
        day_of_week = first.weekday()
        third_friday = first + _dt.timedelta(days=(4 - day_of_week) % 7 + 14)
    return third_friday


def quarterly_expiries(from_date=None, n=4):
    """Generate next n quarterly expiry dates (March, June, Sept, Dec cycle)."""
    d = from_date or _dt.date.today()
    quarterly_months = [3, 6, 9, 12]
    results = []
    year = d.year
    for _ in range(n * 2):
        for m in quarterly_months:
            exp = _third_friday(year, m)
            if exp > d and len(results) < n:
                results.append(exp)
        year += 1
        if len(results) >= n:
            break
    return results[:n]


def _third_friday(year, month):
    first = _dt.date(year, month, 1)
    dow = first.weekday()
    return first + _dt.timedelta(days=(4 - dow) % 7 + 14)


def weekly_expiries(from_date=None, n_weeks=8):
    """Generate next n Friday expiry dates."""
    d = from_date or _dt.date.today()
    days_until_friday = (4 - d.weekday()) % 7
    if days_until_friday == 0:
        days_until_friday = 7
    next_fri = d + _dt.timedelta(days=days_until_friday)
    return [next_fri + _dt.timedelta(weeks=i) for i in range(n_weeks)]


# ---------------------------------------------------------------------------
# Tax treatment helper (Section 1256 contracts)
# ---------------------------------------------------------------------------

def section_1256_tax(net_gain, short_term_rate=0.37, long_term_rate=0.20):
    """
    Section 1256 contract tax treatment: 60% long-term / 40% short-term
    regardless of holding period. Returns blended tax and effective rate.
    """
    if net_gain <= 0:
        return {"tax": 0.0, "effective_rate": 0.0, "long_term_portion": 0.0, "short_term_portion": 0.0}
    lt = 0.6 * net_gain
    st = 0.4 * net_gain
    tax = lt * long_term_rate + st * short_term_rate
    return {
        "tax": tax,
        "effective_rate": tax / net_gain,
        "long_term_portion": lt,
        "short_term_portion": st,
    }
