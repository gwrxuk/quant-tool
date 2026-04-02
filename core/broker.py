"""
Broker integration layer: unified interface for Interactive Brokers, Alpaca, and
Schwab with paper-trading support, order management, and position tracking.
"""
from __future__ import annotations

import abc
import datetime as _dt
import enum
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------

class OrderSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(enum.Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(enum.Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class AssetType(enum.Enum):
    STOCK = "STOCK"
    OPTION = "OPTION"
    INDEX_OPTION = "INDEX_OPTION"
    FUTURE = "FUTURE"


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float = 0.0
    asset_type: AssetType = AssetType.STOCK
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    market_value: float = 0.0

    def update_price(self, price: float):
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = (price - self.avg_cost) * self.quantity


@dataclass
class Order:
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    asset_type: AssetType = AssetType.STOCK
    timestamp: str = ""
    broker_order_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = _dt.datetime.now().isoformat(timespec="seconds")


@dataclass
class AccountInfo:
    account_id: str = ""
    buying_power: float = 0.0
    cash: float = 0.0
    portfolio_value: float = 0.0
    equity: float = 0.0
    margin_used: float = 0.0
    margin_available: float = 0.0
    day_trades_remaining: int = -1
    currency: str = "USD"
    broker: str = ""
    is_paper: bool = True


# ---------------------------------------------------------------------------
# Abstract broker interface
# ---------------------------------------------------------------------------

class BrokerBase(abc.ABC):
    """Unified broker interface all implementations must satisfy."""

    def __init__(self, paper: bool = True):
        self.paper = paper
        self._connected = False
        self._orders: list[Order] = []

    @property
    def connected(self) -> bool:
        return self._connected

    @abc.abstractmethod
    def connect(self, **credentials) -> bool:
        """Establish connection to broker. Returns True on success."""

    @abc.abstractmethod
    def disconnect(self):
        """Close broker connection."""

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """Fetch account summary."""

    @abc.abstractmethod
    def get_positions(self) -> list[Position]:
        """Fetch all open positions."""

    @abc.abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """Fetch real-time quote for symbol."""

    @abc.abstractmethod
    def place_order(self, order: Order) -> Order:
        """Submit an order. Returns the order with updated status/IDs."""

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order. Returns True on success."""

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> Order | None:
        """Check status of an order."""

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status is None:
            return list(self._orders)
        return [o for o in self._orders if o.status == status]

    @abc.abstractmethod
    def get_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        """Fetch options chain data."""


# ---------------------------------------------------------------------------
# Interactive Brokers (via ib_async / ib_insync)
# ---------------------------------------------------------------------------

class IBKRBroker(BrokerBase):
    """Interactive Brokers integration via ib_async (formerly ib_insync)."""

    BROKER_NAME = "IBKR"

    def __init__(self, paper: bool = True):
        super().__init__(paper)
        self._ib = None

    def connect(self, host="127.0.0.1", port=None, client_id=1, **_) -> bool:
        try:
            from ib_async import IB
        except ImportError:
            try:
                from ib_insync import IB
            except ImportError:
                raise ImportError(
                    "ib_async (or ib_insync) is required for IBKR integration. "
                    "Install with: pip install ib_async"
                )

        if port is None:
            port = 7497 if self.paper else 7496
        self._ib = IB()
        try:
            self._ib.connect(host, port, clientId=client_id)
            self._connected = True
            logger.info("Connected to IBKR %s on %s:%d", "paper" if self.paper else "live", host, port)
            return True
        except Exception as e:
            logger.error("IBKR connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        if self._ib:
            self._ib.disconnect()
        self._connected = False

    def get_account(self) -> AccountInfo:
        self._ensure_connected()
        summary = {tag.tag: tag.value for tag in self._ib.accountSummary()}
        return AccountInfo(
            account_id=summary.get("AccountCode", ""),
            buying_power=float(summary.get("BuyingPower", 0)),
            cash=float(summary.get("TotalCashValue", 0)),
            portfolio_value=float(summary.get("NetLiquidation", 0)),
            equity=float(summary.get("EquityWithLoanValue", 0)),
            margin_used=float(summary.get("InitMarginReq", 0)),
            margin_available=float(summary.get("AvailableFunds", 0)),
            currency="USD",
            broker=self.BROKER_NAME,
            is_paper=self.paper,
        )

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        positions = []
        for p in self._ib.positions():
            pos = Position(
                symbol=p.contract.symbol,
                quantity=float(p.position),
                avg_cost=float(p.avgCost),
                asset_type=AssetType.OPTION if p.contract.secType == "OPT" else AssetType.STOCK,
            )
            positions.append(pos)
        return positions

    def get_quote(self, symbol: str) -> dict:
        self._ensure_connected()
        from ib_async import Stock
        contract = Stock(symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)
        ticker = self._ib.reqMktData(contract, snapshot=True)
        self._ib.sleep(2)
        return {
            "symbol": symbol,
            "bid": float(ticker.bid) if ticker.bid else np.nan,
            "ask": float(ticker.ask) if ticker.ask else np.nan,
            "last": float(ticker.last) if ticker.last else np.nan,
            "volume": int(ticker.volume) if ticker.volume else 0,
        }

    def place_order(self, order: Order) -> Order:
        self._ensure_connected()
        from ib_async import Stock, LimitOrder, MarketOrder

        contract = Stock(order.symbol, "SMART", "USD")
        self._ib.qualifyContracts(contract)

        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        if order.order_type == OrderType.MARKET:
            ib_order = MarketOrder(action, order.quantity)
        elif order.order_type == OrderType.LIMIT:
            ib_order = LimitOrder(action, order.quantity, order.limit_price)
        else:
            raise ValueError(f"Order type {order.order_type} not yet supported for IBKR")

        trade = self._ib.placeOrder(contract, ib_order)
        order.broker_order_id = str(trade.order.orderId)
        order.status = OrderStatus.SUBMITTED
        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        for trade in self._ib.openTrades():
            if str(trade.order.orderId) == order_id:
                self._ib.cancelOrder(trade.order)
                return True
        return False

    def get_order_status(self, order_id: str) -> Order | None:
        for o in self._orders:
            if o.broker_order_id == order_id or o.order_id == order_id:
                return o
        return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        self._ensure_connected()
        chains = self._ib.reqSecDefOptParams(symbol, "", "STK", 0)
        if not chains:
            return pd.DataFrame()
        chain = chains[0]
        rows = []
        for exp in chain.expirations[:5] if expiry is None else [expiry]:
            for strike in chain.strikes:
                rows.append({"expiry": exp, "strike": strike, "exchange": chain.exchange})
        return pd.DataFrame(rows)

    def _ensure_connected(self):
        if not self._connected or not self._ib:
            raise ConnectionError("Not connected to IBKR. Call connect() first.")


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

class AlpacaBroker(BrokerBase):
    """Alpaca Markets integration for equities and options."""

    BROKER_NAME = "Alpaca"
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"

    def __init__(self, paper: bool = True):
        super().__init__(paper)
        self._api = None

    def connect(self, api_key="", api_secret="", **_) -> bool:
        try:
            from alpaca.trading.client import TradingClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for Alpaca integration. "
                "Install with: pip install alpaca-py"
            )

        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required for Alpaca")

        try:
            self._api = TradingClient(api_key, api_secret, paper=self.paper)
            _ = self._api.get_account()
            self._connected = True
            logger.info("Connected to Alpaca %s", "paper" if self.paper else "live")
            return True
        except Exception as e:
            logger.error("Alpaca connection failed: %s", e)
            self._connected = False
            return False

    def disconnect(self):
        self._api = None
        self._connected = False

    def get_account(self) -> AccountInfo:
        self._ensure_connected()
        acct = self._api.get_account()
        return AccountInfo(
            account_id=str(acct.id),
            buying_power=float(acct.buying_power),
            cash=float(acct.cash),
            portfolio_value=float(acct.portfolio_value),
            equity=float(acct.equity),
            margin_used=float(acct.initial_margin) if hasattr(acct, "initial_margin") else 0,
            margin_available=float(acct.buying_power),
            day_trades_remaining=int(acct.daytrade_count) if hasattr(acct, "daytrade_count") else -1,
            currency="USD",
            broker=self.BROKER_NAME,
            is_paper=self.paper,
        )

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        positions = []
        for p in self._api.get_all_positions():
            pos = Position(
                symbol=p.symbol,
                quantity=float(p.qty),
                avg_cost=float(p.avg_entry_price),
                current_price=float(p.current_price),
                unrealized_pnl=float(p.unrealized_pl),
                market_value=float(p.market_value),
            )
            positions.append(pos)
        return positions

    def get_quote(self, symbol: str) -> dict:
        self._ensure_connected()
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            from alpaca.data.historical import StockHistoricalDataClient
            data_client = StockHistoricalDataClient(None, None)
            quote = data_client.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=symbol))
            q = quote[symbol]
            return {
                "symbol": symbol,
                "bid": float(q.bid_price),
                "ask": float(q.ask_price),
                "last": float(q.ask_price),
                "volume": int(q.bid_size + q.ask_size),
            }
        except Exception:
            return {"symbol": symbol, "bid": np.nan, "ask": np.nan, "last": np.nan, "volume": 0}

    def place_order(self, order: Order) -> Order:
        self._ensure_connected()
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide as AOS, TimeInForce

        side = AOS.BUY if order.side == OrderSide.BUY else AOS.SELL

        if order.order_type == OrderType.MARKET:
            req = MarketOrderRequest(
                symbol=order.symbol, qty=order.quantity,
                side=side, time_in_force=TimeInForce.DAY,
            )
        elif order.order_type == OrderType.LIMIT:
            req = LimitOrderRequest(
                symbol=order.symbol, qty=order.quantity,
                side=side, time_in_force=TimeInForce.DAY,
                limit_price=order.limit_price,
            )
        else:
            raise ValueError(f"Order type {order.order_type} not yet supported for Alpaca")

        result = self._api.submit_order(req)
        order.broker_order_id = str(result.id)
        order.status = OrderStatus.SUBMITTED
        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        try:
            self._api.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        self._ensure_connected()
        try:
            o = self._api.get_order_by_id(order_id)
            for order in self._orders:
                if order.broker_order_id == order_id:
                    status_map = {
                        "new": OrderStatus.SUBMITTED,
                        "partially_filled": OrderStatus.PARTIAL,
                        "filled": OrderStatus.FILLED,
                        "canceled": OrderStatus.CANCELLED,
                        "rejected": OrderStatus.REJECTED,
                    }
                    order.status = status_map.get(str(o.status), OrderStatus.PENDING)
                    order.filled_quantity = float(o.filled_qty) if o.filled_qty else 0
                    order.filled_price = float(o.filled_avg_price) if o.filled_avg_price else 0
                    return order
        except Exception:
            pass
        return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        logger.warning("Alpaca options chain not available via standard API")
        return pd.DataFrame()

    def _ensure_connected(self):
        if not self._connected or not self._api:
            raise ConnectionError("Not connected to Alpaca. Call connect() first.")


# ---------------------------------------------------------------------------
# Schwab (Charles Schwab / TD Ameritrade)
# ---------------------------------------------------------------------------

class SchwabBroker(BrokerBase):
    """Charles Schwab integration via schwab-py."""

    BROKER_NAME = "Schwab"

    def __init__(self, paper: bool = True):
        super().__init__(paper)
        self._client = None
        self._account_hash = None

    def connect(self, app_key="", app_secret="", token_path="schwab_token.json",
                callback_url="https://127.0.0.1", **_) -> bool:
        try:
            import schwab
        except ImportError:
            raise ImportError(
                "schwab-py is required for Schwab integration. "
                "Install with: pip install schwab-py"
            )

        if not app_key or not app_secret:
            raise ValueError("app_key and app_secret are required for Schwab")

        try:
            self._client = schwab.auth.client_from_token_file(token_path, app_key, app_secret)
        except FileNotFoundError:
            logger.info("No token file found — initiating Schwab OAuth flow")
            try:
                self._client = schwab.auth.client_from_manual_flow(
                    app_key, app_secret, callback_url, token_path
                )
            except Exception as e:
                logger.error("Schwab OAuth failed: %s", e)
                return False

        try:
            resp = self._client.get_account_numbers()
            accounts = resp.json()
            if accounts:
                self._account_hash = accounts[0]["hashValue"]
            self._connected = True
            logger.info("Connected to Schwab")
            return True
        except Exception as e:
            logger.error("Schwab connection failed: %s", e)
            return False

    def disconnect(self):
        self._client = None
        self._connected = False

    def get_account(self) -> AccountInfo:
        self._ensure_connected()
        resp = self._client.get_account(self._account_hash)
        data = resp.json()
        bal = data.get("securitiesAccount", {}).get("currentBalances", {})
        return AccountInfo(
            account_id=self._account_hash or "",
            buying_power=float(bal.get("buyingPower", 0)),
            cash=float(bal.get("cashBalance", 0)),
            portfolio_value=float(bal.get("liquidationValue", 0)),
            equity=float(bal.get("equity", 0)),
            currency="USD",
            broker=self.BROKER_NAME,
            is_paper=self.paper,
        )

    def get_positions(self) -> list[Position]:
        self._ensure_connected()
        resp = self._client.get_account(self._account_hash, fields=["positions"])
        data = resp.json()
        positions = []
        for p in data.get("securitiesAccount", {}).get("positions", []):
            inst = p.get("instrument", {})
            positions.append(Position(
                symbol=inst.get("symbol", ""),
                quantity=float(p.get("longQuantity", 0)) - float(p.get("shortQuantity", 0)),
                avg_cost=float(p.get("averagePrice", 0)),
                current_price=float(p.get("marketValue", 0)) / max(
                    float(p.get("longQuantity", 0)) + float(p.get("shortQuantity", 0)), 1
                ),
                asset_type=AssetType.OPTION if inst.get("assetType") == "OPTION" else AssetType.STOCK,
            ))
        return positions

    def get_quote(self, symbol: str) -> dict:
        self._ensure_connected()
        try:
            resp = self._client.get_quote(symbol)
            data = resp.json()
            q = data.get(symbol, {}).get("quote", {})
            return {
                "symbol": symbol,
                "bid": float(q.get("bidPrice", 0)),
                "ask": float(q.get("askPrice", 0)),
                "last": float(q.get("lastPrice", 0)),
                "volume": int(q.get("totalVolume", 0)),
            }
        except Exception:
            return {"symbol": symbol, "bid": np.nan, "ask": np.nan, "last": np.nan, "volume": 0}

    def place_order(self, order: Order) -> Order:
        self._ensure_connected()
        order_body = {
            "orderType": order.order_type.value,
            "session": "NORMAL",
            "duration": "DAY",
            "orderStrategyType": "SINGLE",
            "orderLegCollection": [{
                "instruction": order.side.value,
                "quantity": order.quantity,
                "instrument": {
                    "symbol": order.symbol,
                    "assetType": "EQUITY",
                },
            }],
        }
        if order.order_type == OrderType.LIMIT:
            order_body["price"] = str(order.limit_price)

        try:
            resp = self._client.place_order(self._account_hash, order_body)
            order.status = OrderStatus.SUBMITTED
            location = resp.headers.get("Location", "")
            if location:
                order.broker_order_id = location.split("/")[-1]
        except Exception as e:
            logger.error("Schwab order failed: %s", e)
            order.status = OrderStatus.REJECTED

        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        self._ensure_connected()
        try:
            self._client.cancel_order(order_id, self._account_hash)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> Order | None:
        for o in self._orders:
            if o.broker_order_id == order_id or o.order_id == order_id:
                return o
        return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        self._ensure_connected()
        try:
            kwargs = {"symbol": symbol}
            if expiry:
                kwargs["from_date"] = _dt.datetime.strptime(expiry, "%Y-%m-%d")
                kwargs["to_date"] = kwargs["from_date"]
            resp = self._client.get_option_chain(symbol)
            data = resp.json()
            rows = []
            for exp_map_key in ("callExpDateMap", "putExpDateMap"):
                opt_type = "call" if "call" in exp_map_key else "put"
                for exp_key, strikes in data.get(exp_map_key, {}).items():
                    for strike_key, contracts in strikes.items():
                        for c in contracts:
                            rows.append({
                                "type": opt_type,
                                "expiry": exp_key.split(":")[0],
                                "strike": float(strike_key),
                                "bid": c.get("bid", 0),
                                "ask": c.get("ask", 0),
                                "last": c.get("last", 0),
                                "volume": c.get("totalVolume", 0),
                                "openInterest": c.get("openInterest", 0),
                                "iv": c.get("volatility", 0) / 100,
                                "delta": c.get("delta", 0),
                                "gamma": c.get("gamma", 0),
                                "theta": c.get("theta", 0),
                                "vega": c.get("vega", 0),
                            })
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    def _ensure_connected(self):
        if not self._connected or not self._client:
            raise ConnectionError("Not connected to Schwab. Call connect() first.")


# ---------------------------------------------------------------------------
# Paper Trading Broker (local simulation, no external dependency)
# ---------------------------------------------------------------------------

class PaperBroker(BrokerBase):
    """
    In-memory paper trading broker for testing strategies without any
    external API. Uses yfinance for price data.
    """

    BROKER_NAME = "Paper"

    def __init__(self, initial_cash: float = 100_000.0, paper: bool = True):
        super().__init__(paper=True)
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._positions: dict[str, Position] = {}
        self._order_counter = 0
        self._trade_log: list[dict] = []

    def connect(self, **_) -> bool:
        self._connected = True
        logger.info("Paper broker ready with $%,.2f", self._cash)
        return True

    def disconnect(self):
        self._connected = False

    def get_account(self) -> AccountInfo:
        portfolio_value = self._cash
        for pos in self._positions.values():
            portfolio_value += pos.market_value
        return AccountInfo(
            account_id="PAPER-001",
            buying_power=self._cash,
            cash=self._cash,
            portfolio_value=portfolio_value,
            equity=portfolio_value,
            currency="USD",
            broker=self.BROKER_NAME,
            is_paper=True,
        )

    def get_positions(self) -> list[Position]:
        return list(self._positions.values())

    def get_quote(self, symbol: str) -> dict:
        try:
            from core.market_data import get_quote
            q = get_quote(symbol)
            return {
                "symbol": symbol,
                "bid": q.get("price", np.nan) * 0.9999,
                "ask": q.get("price", np.nan) * 1.0001,
                "last": q.get("price", np.nan),
                "volume": q.get("volume", 0),
            }
        except Exception:
            return {"symbol": symbol, "bid": np.nan, "ask": np.nan, "last": np.nan, "volume": 0}

    def place_order(self, order: Order) -> Order:
        self._order_counter += 1
        order.order_id = f"PAPER-{self._order_counter:06d}"
        order.broker_order_id = order.order_id

        quote = self.get_quote(order.symbol)
        fill_price = quote.get("last", np.nan)

        if np.isnan(fill_price):
            order.status = OrderStatus.REJECTED
            self._orders.append(order)
            return order

        if order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and order.limit_price < fill_price:
                order.status = OrderStatus.PENDING
                self._orders.append(order)
                return order
            if order.side == OrderSide.SELL and order.limit_price > fill_price:
                order.status = OrderStatus.PENDING
                self._orders.append(order)
                return order
            fill_price = order.limit_price

        cost = fill_price * order.quantity
        if order.side == OrderSide.BUY:
            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                self._orders.append(order)
                return order
            self._cash -= cost
            self._update_position(order.symbol, order.quantity, fill_price, order.asset_type)
        else:
            self._cash += cost
            self._update_position(order.symbol, -order.quantity, fill_price, order.asset_type)

        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price

        self._trade_log.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "quantity": order.quantity,
            "price": fill_price,
            "timestamp": order.timestamp,
        })

        self._orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        for o in self._orders:
            if o.order_id == order_id and o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED):
                o.status = OrderStatus.CANCELLED
                return True
        return False

    def get_order_status(self, order_id: str) -> Order | None:
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return None

    def get_option_chain(self, symbol: str, expiry: str | None = None) -> pd.DataFrame:
        try:
            from core.market_data import get_options_chain, get_expiry_dates
            if expiry is None:
                dates = get_expiry_dates(symbol)
                if not dates:
                    return pd.DataFrame()
                expiry = dates[0]
            chain = get_options_chain(symbol, expiry)
            return pd.concat([chain["calls"], chain["puts"]], ignore_index=True)
        except Exception:
            return pd.DataFrame()

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self._trade_log) if self._trade_log else pd.DataFrame(
            columns=["order_id", "symbol", "side", "quantity", "price", "timestamp"]
        )

    def reset(self):
        """Reset paper account to initial state."""
        self._cash = self._initial_cash
        self._positions.clear()
        self._orders.clear()
        self._trade_log.clear()
        self._order_counter = 0

    def _update_position(self, symbol: str, qty_delta: float, price: float, asset_type: AssetType):
        if symbol in self._positions:
            pos = self._positions[symbol]
            old_qty = pos.quantity
            new_qty = old_qty + qty_delta
            if abs(new_qty) < 1e-10:
                del self._positions[symbol]
                return
            if (old_qty > 0 and qty_delta > 0) or (old_qty < 0 and qty_delta < 0):
                pos.avg_cost = (pos.avg_cost * old_qty + price * qty_delta) / new_qty
            pos.quantity = new_qty
            pos.update_price(price)
        else:
            if abs(qty_delta) > 1e-10:
                pos = Position(
                    symbol=symbol,
                    quantity=qty_delta,
                    avg_cost=price,
                    current_price=price,
                    asset_type=asset_type,
                    market_value=qty_delta * price,
                )
                self._positions[symbol] = pos


# ---------------------------------------------------------------------------
# Broker factory
# ---------------------------------------------------------------------------

BROKER_REGISTRY = {
    "ibkr": IBKRBroker,
    "interactive_brokers": IBKRBroker,
    "alpaca": AlpacaBroker,
    "schwab": SchwabBroker,
    "td_ameritrade": SchwabBroker,
    "paper": PaperBroker,
}


def create_broker(name: str, paper: bool = True, **kwargs) -> BrokerBase:
    """
    Factory function to create a broker instance.

    Supported names: ibkr, alpaca, schwab, paper.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    cls = BROKER_REGISTRY.get(key)
    if cls is None:
        raise ValueError(
            f"Unknown broker '{name}'. Supported: {', '.join(BROKER_REGISTRY.keys())}"
        )
    if key == "paper":
        return cls(initial_cash=kwargs.pop("initial_cash", 100_000), **kwargs)
    return cls(paper=paper)


def list_brokers() -> list[str]:
    return ["IBKR (Interactive Brokers)", "Alpaca", "Schwab", "Paper Trading"]
