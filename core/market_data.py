"""
Real-time market data: live quotes, options chains, implied volatility surfaces,
historical prices, and risk-free rate estimation via yfinance.
"""
import datetime as _dt
import functools
import time

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from core.pricing import bs_price, implied_vol


# ---------------------------------------------------------------------------
# Ticker universe
# ---------------------------------------------------------------------------

POPULAR_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM",
    "V", "UNH", "XOM", "JNJ", "WMT", "PG", "MA", "HD", "BAC", "COST",
    "NFLX", "DIS", "AMD", "INTC", "CRM", "PYPL", "UBER", "COIN",
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLF", "XLE", "XLK",
]

INDEX_TICKERS = {
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "RUT": "^RUT",
    "DJX": "^DJI",
    "VIX": "^VIX",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_yfinance():
    if yf is None:
        raise ImportError(
            "yfinance is required for live market data. "
            "Install it with: pip install yfinance"
        )


def _ttl_cache(seconds: int = 60):
    """Decorator: time-based cache invalidation (works with unhashable args)."""
    def decorator(fn):
        _cache: dict = {}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key = (args, tuple(sorted(kwargs.items())))
            now = time.monotonic()
            if key in _cache:
                ts, val = _cache[key]
                if now - ts < seconds:
                    return val
            val = fn(*args, **kwargs)
            _cache[key] = (now, val)
            return val

        wrapper.cache_clear = _cache.clear
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Live quotes
# ---------------------------------------------------------------------------

@_ttl_cache(seconds=15)
def get_quote(ticker: str) -> dict:
    """
    Fetch a live/delayed quote for *ticker*.

    Tries ``tk.info`` first; on transient failures (TLS, rate-limit) falls
    back to ``tk.history()`` so the page never hard-crashes.

    Returns dict with keys: symbol, name, price, previous_close, change,
    change_pct, day_high, day_low, volume, market_cap, bid, ask, fifty_two_wk_high,
    fifty_two_wk_low, currency, exchange, timestamp.
    """
    _require_yfinance()
    tk = yf.Ticker(ticker)

    info: dict = {}
    try:
        info = tk.info or {}
    except Exception:
        pass

    if not info or not info.get("currentPrice"):
        try:
            info = _quote_from_history(tk, ticker, info)
        except Exception:
            pass

    price = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or np.nan
    )
    prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose") or np.nan
    change = price - prev_close if not (np.isnan(price) or np.isnan(prev_close)) else np.nan
    change_pct = (change / prev_close * 100) if prev_close else np.nan

    return {
        "symbol": info.get("symbol", ticker.upper()),
        "name": info.get("shortName") or info.get("longName") or ticker.upper(),
        "price": price,
        "previous_close": prev_close,
        "change": change,
        "change_pct": change_pct,
        "day_high": info.get("dayHigh", np.nan),
        "day_low": info.get("dayLow", np.nan),
        "volume": info.get("volume", 0),
        "market_cap": info.get("marketCap"),
        "bid": info.get("bid", np.nan),
        "ask": info.get("ask", np.nan),
        "fifty_two_wk_high": info.get("fiftyTwoWeekHigh", np.nan),
        "fifty_two_wk_low": info.get("fiftyTwoWeekLow", np.nan),
        "currency": info.get("currency", "USD"),
        "exchange": info.get("exchange", ""),
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
    }


def _quote_from_history(tk, ticker: str, partial_info: dict) -> dict:
    """Build a quote-like dict from ``tk.history()`` when ``tk.info`` fails."""
    hist = tk.history(period="5d")
    if hist.empty:
        return partial_info

    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) >= 2 else last

    merged = dict(partial_info)
    merged.setdefault("symbol", ticker.upper())
    merged.setdefault("shortName", ticker.upper())
    merged["currentPrice"] = float(last["Close"])
    merged["previousClose"] = float(prev["Close"])
    merged.setdefault("dayHigh", float(last["High"]))
    merged.setdefault("dayLow", float(last["Low"]))
    merged.setdefault("volume", int(last["Volume"]))

    if len(hist) >= 252:
        merged.setdefault("fiftyTwoWeekHigh", float(hist["High"].max()))
        merged.setdefault("fiftyTwoWeekLow", float(hist["Low"].min()))

    return merged


# ---------------------------------------------------------------------------
# Historical price data
# ---------------------------------------------------------------------------

PERIOD_MAP = {
    "1D": ("1d", "5m"),
    "5D": ("5d", "15m"),
    "1M": ("1mo", "1h"),
    "3M": ("3mo", "1d"),
    "6M": ("6mo", "1d"),
    "YTD": ("ytd", "1d"),
    "1Y": ("1y", "1d"),
    "5Y": ("5y", "1wk"),
    "MAX": ("max", "1mo"),
}


@_ttl_cache(seconds=60)
def get_historical(ticker: str, period: str = "3M") -> pd.DataFrame:
    """
    Return OHLCV DataFrame for *ticker* over *period*.
    Valid periods: 1D, 5D, 1M, 3M, 6M, YTD, 1Y, 5Y, MAX.
    """
    _require_yfinance()
    yf_period, yf_interval = PERIOD_MAP.get(period, ("3mo", "1d"))
    tk = yf.Ticker(ticker)
    df = tk.history(period=yf_period, interval=yf_interval)
    if df.empty:
        return df
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Options data
# ---------------------------------------------------------------------------

@_ttl_cache(seconds=120)
def get_expiry_dates(ticker: str) -> tuple:
    """Return available options expiry dates as tuple of date-strings."""
    _require_yfinance()
    tk = yf.Ticker(ticker)
    return tk.options  # tuple of 'YYYY-MM-DD' strings


@_ttl_cache(seconds=60)
def get_options_chain(ticker: str, expiry: str) -> dict:
    """
    Fetch the full options chain for *ticker* at *expiry*.

    Returns dict with keys:
        calls  – DataFrame (strike, lastPrice, bid, ask, volume, openInterest, impliedVolatility, …)
        puts   – same structure
        expiry – expiry string
        spot   – current underlying price
        T      – time to expiry in years
    """
    _require_yfinance()
    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    info = tk.info or {}
    spot = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or np.nan
    )

    exp_date = _dt.datetime.strptime(expiry, "%Y-%m-%d")
    now = _dt.datetime.now()
    T = max((exp_date - now).days, 0) / 365.0

    calls = chain.calls.copy()
    puts = chain.puts.copy()

    for df in (calls, puts):
        if "impliedVolatility" in df.columns:
            df["iv"] = df["impliedVolatility"]
        if "lastPrice" not in df.columns:
            df["lastPrice"] = np.nan

    return {
        "calls": calls,
        "puts": puts,
        "expiry": expiry,
        "spot": spot,
        "T": T,
    }


# ---------------------------------------------------------------------------
# Implied Volatility Surface (from live options chains)
# ---------------------------------------------------------------------------

def get_iv_surface(ticker: str, max_expiries: int = 8,
                   moneyness_range: tuple = (0.75, 1.25),
                   r: float = 0.045) -> dict:
    """
    Build an implied-volatility surface from real options chain data.

    Returns dict with keys: strikes (1-D), expiries_years (1-D),
    iv_matrix (2-D, shape len(expiries) × len(strikes)), spot, expiry_labels.
    """
    _require_yfinance()
    dates = get_expiry_dates(ticker)
    if not dates:
        return {}

    dates = dates[:max_expiries]
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    spot = (
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
        or np.nan
    )
    if np.isnan(spot):
        return {}

    lo, hi = spot * moneyness_range[0], spot * moneyness_range[1]

    raw_rows = []
    for d in dates:
        try:
            chain_data = get_options_chain(ticker, d)
        except Exception:
            continue
        T = chain_data["T"]
        if T < 1 / 365:
            continue
        calls = chain_data["calls"]
        for _, row in calls.iterrows():
            K = row.get("strike", np.nan)
            iv_val = row.get("iv") or row.get("impliedVolatility", np.nan)
            if np.isnan(K) or np.isnan(iv_val) or iv_val <= 0.001:
                continue
            if lo <= K <= hi:
                raw_rows.append({"strike": K, "T": T, "iv": iv_val, "expiry": d})

    if not raw_rows:
        return {}

    raw = pd.DataFrame(raw_rows)

    all_strikes = np.sort(raw["strike"].unique())
    all_T = np.sort(raw["T"].unique())

    if len(all_strikes) < 3 or len(all_T) < 2:
        return {}

    iv_matrix = np.full((len(all_T), len(all_strikes)), np.nan)
    for i, T in enumerate(all_T):
        sub = raw[raw["T"] == T]
        for _, row in sub.iterrows():
            j = np.searchsorted(all_strikes, row["strike"])
            if j < len(all_strikes) and abs(all_strikes[j] - row["strike"]) < 0.01:
                iv_matrix[i, j] = row["iv"]

    for i in range(len(all_T)):
        row = iv_matrix[i]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            iv_matrix[i] = np.interp(
                all_strikes, all_strikes[valid], row[valid]
            )

    expiry_labels = []
    T_to_date = dict(zip(raw["T"], raw["expiry"]))
    for T in all_T:
        expiry_labels.append(T_to_date.get(T, f"{T:.3f}y"))

    return {
        "strikes": all_strikes,
        "expiries_years": all_T,
        "iv_matrix": iv_matrix,
        "spot": spot,
        "expiry_labels": expiry_labels,
    }


# ---------------------------------------------------------------------------
# Risk-free rate proxy (US Treasury)
# ---------------------------------------------------------------------------

@_ttl_cache(seconds=3600)
def get_risk_free_rate(maturity: str = "3m") -> float:
    """
    Estimate the risk-free rate from US Treasury ETF yields.

    maturity: '3m' (^IRX), '2y' (^FVX proxy), '10y' (^TNX), '30y' (^TYX).
    Falls back to 0.045 on failure.
    """
    _require_yfinance()
    symbol_map = {
        "3m": "^IRX",
        "10y": "^TNX",
        "30y": "^TYX",
    }
    sym = symbol_map.get(maturity, "^IRX")
    try:
        tk = yf.Ticker(sym)
        hist = tk.history(period="5d")
        if hist.empty:
            return 0.045
        return hist["Close"].iloc[-1] / 100.0
    except Exception:
        return 0.045


# ---------------------------------------------------------------------------
# Market overview (key indices + movers)
# ---------------------------------------------------------------------------

MAJOR_INDICES = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "VIX": "^VIX",
    "10Y Treasury": "^TNX",
}


@_ttl_cache(seconds=30)
def get_market_overview() -> pd.DataFrame:
    """Snapshot of major indices: name, symbol, price, change, change_pct."""
    _require_yfinance()
    rows = []
    for name, sym in MAJOR_INDICES.items():
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="2d")
            if hist.empty or len(hist) < 1:
                continue
            close = hist["Close"].iloc[-1]
            prev = hist["Close"].iloc[-2] if len(hist) >= 2 else close
            chg = close - prev
            chg_pct = chg / prev * 100 if prev else 0
            rows.append({
                "Index": name,
                "Symbol": sym,
                "Price": close,
                "Change": chg,
                "Change %": chg_pct,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: enrich options chain with BS-model Greeks
# ---------------------------------------------------------------------------

def enrich_chain_with_greeks(chain_df: pd.DataFrame, spot: float, T: float,
                              r: float, option_type: str = "call") -> pd.DataFrame:
    """
    Add columns for BS price, delta, gamma, vega, theta to an options chain DF.
    Requires columns: strike, iv (or impliedVolatility).
    """
    from core.greeks import delta, gamma, vega, theta

    df = chain_df.copy()
    iv_col = "iv" if "iv" in df.columns else "impliedVolatility"

    bs_prices, deltas, gammas, vegas, thetas = [], [], [], [], []
    for _, row in df.iterrows():
        K = row["strike"]
        sig = row.get(iv_col, np.nan)
        if np.isnan(sig) or sig <= 0 or T <= 0:
            bs_prices.append(np.nan)
            deltas.append(np.nan)
            gammas.append(np.nan)
            vegas.append(np.nan)
            thetas.append(np.nan)
        else:
            bs_prices.append(bs_price(spot, K, T, r, sig, option_type))
            deltas.append(delta(spot, K, T, r, sig, option_type))
            gammas.append(gamma(spot, K, T, r, sig))
            vegas.append(vega(spot, K, T, r, sig))
            thetas.append(theta(spot, K, T, r, sig, option_type))

    df["bs_price"] = bs_prices
    df["delta"] = deltas
    df["gamma"] = gammas
    df["vega"] = vegas
    df["theta"] = thetas
    return df


# ---------------------------------------------------------------------------
# Convenience: fetch live spot, ATM IV, and risk-free rate for a ticker
# ---------------------------------------------------------------------------

def get_live_params(ticker: str) -> dict:
    """
    Return a dict with live market parameters for use in pricing tools.

    Keys: ticker, spot, sigma (ATM implied vol), r (risk-free rate), name.
    Falls back gracefully on partial failures.
    """
    result = {"ticker": ticker, "spot": 100.0, "sigma": 0.20, "r": 0.05, "name": ticker}

    try:
        q = get_quote(ticker)
        if q and not np.isnan(q.get("price", np.nan)):
            result["spot"] = q["price"]
            result["name"] = q.get("name", ticker)
    except Exception:
        pass

    try:
        r = get_risk_free_rate("3m")
        if r and not np.isnan(r):
            result["r"] = round(r, 4)
    except Exception:
        pass

    try:
        dates = get_expiry_dates(ticker)
        if dates:
            spot = result["spot"]
            r_val = result["r"]
            for d in dates[:12]:
                chain = get_options_chain(ticker, d)
                T = chain["T"] if chain else 0
                if chain and T > 14 / 365:
                    calls = chain["calls"]
                    calls = calls.copy()
                    calls["dist"] = (calls["strike"] - spot).abs()
                    atm = calls.nsmallest(5, "dist")

                    iv_col = "iv" if "iv" in calls.columns else "impliedVolatility"
                    iv_vals = atm[iv_col].dropna()
                    iv_mean = float(iv_vals.mean()) if len(iv_vals) > 0 else 0

                    if iv_mean > 0.02:
                        result["sigma"] = round(iv_mean, 4)
                    else:
                        computed_ivs = []
                        for _, row in atm.iterrows():
                            lp = row.get("lastPrice", np.nan)
                            K = row["strike"]
                            if not np.isnan(lp) and lp > 0:
                                iv_c = implied_vol(lp, spot, K, T, r_val, "call")
                                if not np.isnan(iv_c) and 0.02 < iv_c < 3.0:
                                    computed_ivs.append(iv_c)
                        if computed_ivs:
                            result["sigma"] = round(float(np.mean(computed_ivs)), 4)
                    break
    except Exception:
        pass

    return result
