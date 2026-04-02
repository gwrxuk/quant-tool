import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime as _dt

from core.broker import (
    create_broker, list_brokers, BrokerBase, PaperBroker,
    Order, OrderSide, OrderType, OrderStatus, AssetType,
)

st.set_page_config(page_title="Broker", layout="wide")
st.title("Broker Integration & Trading")

# ── Sidebar: Broker Connection ───────────────────────────────────────────
st.sidebar.header("Broker Connection")

broker_options = ["Paper Trading", "IBKR (Interactive Brokers)", "Alpaca", "Schwab"]
selected_broker = st.sidebar.selectbox("Broker", broker_options, index=0)
paper_mode = st.sidebar.checkbox("Paper Trading", value=True, key="broker_paper")

broker: BrokerBase | None = st.session_state.get("broker_instance")

# Clear stale broker when user switches broker type
_prev_broker_sel = st.session_state.get("_broker_selection")
if _prev_broker_sel and _prev_broker_sel != selected_broker:
    st.session_state.pop("broker_instance", None)
    st.session_state.pop("broker_name", None)
    broker = None
st.session_state["_broker_selection"] = selected_broker

if selected_broker == "Paper Trading":
    initial_cash = st.sidebar.number_input("Initial Cash ($)", value=100_000.0, step=10_000.0, key="paper_cash")
    if st.sidebar.button("Connect", key="connect_paper"):
        broker = create_broker("paper", initial_cash=initial_cash)
        broker.connect()
        st.session_state["broker_instance"] = broker
        st.session_state["broker_name"] = "Paper Trading"
        st.rerun()

elif selected_broker == "IBKR (Interactive Brokers)":
    ib_host = st.sidebar.text_input("Host", value="127.0.0.1", key="ib_host")
    ib_port = st.sidebar.number_input("Port", value=7497 if paper_mode else 7496, step=1, key="ib_port")
    ib_client = st.sidebar.number_input("Client ID", value=1, step=1, key="ib_client")
    if st.sidebar.button("Connect", key="connect_ibkr"):
        try:
            broker = create_broker("ibkr", paper=paper_mode)
            success = broker.connect(host=ib_host, port=int(ib_port), client_id=int(ib_client))
            if success:
                st.session_state["broker_instance"] = broker
                st.session_state["broker_name"] = "IBKR"
                st.rerun()
            else:
                st.sidebar.error("Connection failed. Is TWS/Gateway running?")
        except ImportError as e:
            st.sidebar.error(str(e))
        except Exception as e:
            st.sidebar.error(f"Connection error: {e}")

elif selected_broker == "Alpaca":
    alp_key = st.sidebar.text_input("API Key", type="password", key="alp_key")
    alp_secret = st.sidebar.text_input("API Secret", type="password", key="alp_secret")
    if st.sidebar.button("Connect", key="connect_alpaca"):
        if not alp_key or not alp_secret:
            st.sidebar.error("API Key and Secret are required.")
        else:
            try:
                broker = create_broker("alpaca", paper=paper_mode)
                success = broker.connect(api_key=alp_key, api_secret=alp_secret)
                if success:
                    st.session_state["broker_instance"] = broker
                    st.session_state["broker_name"] = "Alpaca"
                    st.rerun()
                else:
                    st.sidebar.error("Connection failed. Check credentials.")
            except ImportError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Connection error: {e}")

elif selected_broker == "Schwab":
    sch_key = st.sidebar.text_input("App Key", type="password", key="sch_key")
    sch_secret = st.sidebar.text_input("App Secret", type="password", key="sch_secret")
    if st.sidebar.button("Connect", key="connect_schwab"):
        if not sch_key or not sch_secret:
            st.sidebar.error("App Key and Secret are required.")
        else:
            try:
                broker = create_broker("schwab", paper=paper_mode)
                success = broker.connect(app_key=sch_key, app_secret=sch_secret)
                if success:
                    st.session_state["broker_instance"] = broker
                    st.session_state["broker_name"] = "Schwab"
                    st.rerun()
                else:
                    st.sidebar.error("Connection failed.")
            except ImportError as e:
                st.sidebar.error(str(e))
            except Exception as e:
                st.sidebar.error(f"Connection error: {e}")

# Connection status
broker = st.session_state.get("broker_instance")
broker_name = st.session_state.get("broker_name", "None")

if broker and broker.connected:
    st.sidebar.success(f"Connected: **{broker_name}**")
    if st.sidebar.button("Disconnect", key="broker_disconnect"):
        broker.disconnect()
        st.session_state.pop("broker_instance", None)
        st.session_state.pop("broker_name", None)
        st.rerun()
else:
    st.sidebar.warning("Not connected")

st.sidebar.divider()

# ── Main Content ─────────────────────────────────────────────────────────
if not broker or not broker.connected:
    st.info("Connect to a broker using the sidebar to access trading features.")
    st.markdown("""
    ### Supported Brokers

    | Broker | Features | Requirements |
    |--------|----------|-------------|
    | **Paper Trading** | Full simulation, no API needed | None |
    | **IBKR** | Equities, options, futures | TWS or IB Gateway + `ib_async` |
    | **Alpaca** | Equities, crypto | API key + `alpaca-py` |
    | **Schwab** | Equities, options | OAuth app + `schwab-py` |

    ### Quick Start
    1. Select **Paper Trading** to get started immediately
    2. Click **Connect** to initialize the paper broker
    3. Place orders, track positions, and monitor P&L

    ### External Broker Setup

    **IBKR**: Install TWS or IB Gateway, enable API connections (port 7496 live / 7497 paper),
    then `pip install ib_async`.

    **Alpaca**: Create an account at alpaca.markets, generate API keys, then `pip install alpaca-py`.

    **Schwab**: Register a developer app at developer.schwab.com, then `pip install schwab-py`.
    """)
    st.stop()

# ── Connected: Show trading interface ────────────────────────────────────
tab_account, tab_positions, tab_orders, tab_trade, tab_log = st.tabs([
    "Account", "Positions", "Orders", "Place Order", "Trade Log",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 – Account Overview
# ═══════════════════════════════════════════════════════════════════════════
with tab_account:
    st.subheader("Account Summary")

    try:
        acct = broker.get_account()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Portfolio Value", f"${acct.portfolio_value:,.2f}")
        c2.metric("Cash", f"${acct.cash:,.2f}")
        c3.metric("Buying Power", f"${acct.buying_power:,.2f}")
        c4.metric("Equity", f"${acct.equity:,.2f}")

        st.divider()
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Broker", acct.broker)
        c6.metric("Account", acct.account_id[:12] + "…" if len(acct.account_id) > 12 else acct.account_id)
        c7.metric("Currency", acct.currency)
        c8.metric("Mode", "Paper" if acct.is_paper else "Live")

        if isinstance(broker, PaperBroker):
            initial = broker._initial_cash
            total_pnl = acct.portfolio_value - initial
            pnl_pct = (total_pnl / initial * 100) if initial > 0 else 0

            st.divider()
            st.subheader("Paper Trading Performance")
            p1, p2, p3 = st.columns(3)
            p1.metric("Initial Capital", f"${initial:,.2f}")
            p2.metric("Total P&L", f"${total_pnl:,.2f}", delta=f"{pnl_pct:+.2f}%")
            p3.metric("Total Trades", len(broker._trade_log))

            if acct.portfolio_value > 0 and acct.cash < acct.portfolio_value:
                fig_alloc = go.Figure(data=[go.Pie(
                    labels=["Cash", "Positions"],
                    values=[acct.cash, acct.portfolio_value - acct.cash],
                    hole=0.4,
                    marker_colors=["#2962FF", "#FF6D00"],
                )])
                fig_alloc.update_layout(title="Capital Allocation", height=350)
                st.plotly_chart(fig_alloc, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch account data: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 – Positions
# ═══════════════════════════════════════════════════════════════════════════
with tab_positions:
    st.subheader("Open Positions")

    try:
        positions = broker.get_positions()

        if not positions:
            st.info("No open positions.")
        else:
            pos_data = []
            for pos in positions:
                pos_data.append({
                    "Symbol": pos.symbol,
                    "Quantity": pos.quantity,
                    "Avg Cost": f"${pos.avg_cost:.2f}",
                    "Current": f"${pos.current_price:.2f}",
                    "Market Value": f"${pos.market_value:,.2f}",
                    "Unrealized P&L": f"${pos.unrealized_pnl:,.2f}",
                    "Type": pos.asset_type.value,
                })
            st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)

            total_mv = sum(p.market_value for p in positions)
            total_pnl = sum(p.unrealized_pnl for p in positions)
            c1, c2 = st.columns(2)
            c1.metric("Total Market Value", f"${total_mv:,.2f}")
            c2.metric("Total Unrealized P&L", f"${total_pnl:,.2f}")

            if len(positions) > 1:
                fig_pos = go.Figure(data=[go.Bar(
                    x=[p.symbol for p in positions],
                    y=[p.unrealized_pnl for p in positions],
                    marker_color=["#00C853" if p.unrealized_pnl >= 0 else "#D50000" for p in positions],
                )])
                fig_pos.update_layout(
                    title="Unrealized P&L by Position",
                    xaxis_title="Symbol", yaxis_title="P&L ($)",
                    height=400,
                )
                st.plotly_chart(fig_pos, use_container_width=True)

    except Exception as e:
        st.error(f"Could not fetch positions: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 – Orders
# ═══════════════════════════════════════════════════════════════════════════
with tab_orders:
    st.subheader("Order History")

    orders = broker.get_orders()
    if not orders:
        st.info("No orders placed yet.")
    else:
        order_data = []
        for o in orders:
            order_data.append({
                "Order ID": o.order_id or o.broker_order_id,
                "Symbol": o.symbol,
                "Side": o.side.value,
                "Type": o.order_type.value,
                "Qty": o.quantity,
                "Limit": f"${o.limit_price:.2f}" if o.limit_price else "—",
                "Filled Qty": o.filled_quantity,
                "Fill Price": f"${o.filled_price:.2f}" if o.filled_price else "—",
                "Status": o.status.value,
                "Time": o.timestamp,
            })
        st.dataframe(pd.DataFrame(order_data), use_container_width=True, hide_index=True)

        status_counts = {}
        for o in orders:
            s = o.status.value
            status_counts[s] = status_counts.get(s, 0) + 1

        status_colors = {
            "FILLED": "#00C853", "PENDING": "#FF6D00", "SUBMITTED": "#2962FF",
            "CANCELLED": "#757575", "REJECTED": "#D50000", "PARTIAL": "#AA00FF",
        }
        fig_status = go.Figure(data=[go.Pie(
            labels=list(status_counts.keys()),
            values=list(status_counts.values()),
            marker_colors=[status_colors.get(s, "#999") for s in status_counts],
            hole=0.4,
        )])
        fig_status.update_layout(title="Order Status Distribution", height=300)
        st.plotly_chart(fig_status, use_container_width=True)

        pending = [o for o in orders if o.status in (OrderStatus.PENDING, OrderStatus.SUBMITTED)]
        if pending:
            st.subheader("Cancel Open Orders")
            for o in pending:
                col1, col2 = st.columns([3, 1])
                col1.text(f"{o.side.value} {o.quantity} {o.symbol} @ {o.limit_price or 'MKT'}")
                if col2.button("Cancel", key=f"cancel_{o.order_id}"):
                    success = broker.cancel_order(o.broker_order_id or o.order_id)
                    if success:
                        st.success(f"Cancelled order {o.order_id}")
                    else:
                        st.error("Cancel failed")
                    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 – Place Order
# ═══════════════════════════════════════════════════════════════════════════
with tab_trade:
    st.subheader("Place New Order")

    col_o1, col_o2 = st.columns(2)

    with col_o1:
        order_symbol = st.text_input("Symbol", value="AAPL", key="order_sym").upper()
        order_side = st.selectbox("Side", ["BUY", "SELL"], key="order_side")
        order_type = st.selectbox("Order Type", ["MARKET", "LIMIT", "STOP", "STOP_LIMIT"], key="order_type")
        order_qty = st.number_input("Quantity", value=10, step=1, min_value=1, key="order_qty")

    with col_o2:
        order_limit = None
        order_stop = None

        if order_type in ("LIMIT", "STOP_LIMIT"):
            order_limit = st.number_input("Limit Price", value=150.0, step=0.50, key="order_limit")
        if order_type in ("STOP", "STOP_LIMIT"):
            order_stop = st.number_input("Stop Price", value=145.0, step=0.50, key="order_stop")

        order_asset = st.selectbox("Asset Type", ["STOCK", "OPTION", "INDEX_OPTION"], key="order_asset")

        if st.button("Get Quote", key="order_quote"):
            try:
                q = broker.get_quote(order_symbol)
                st.metric("Last Price", f"${q.get('last', 0):.2f}")
                q1, q2 = st.columns(2)
                q1.metric("Bid", f"${q.get('bid', 0):.2f}")
                q2.metric("Ask", f"${q.get('ask', 0):.2f}")
            except Exception as e:
                st.error(f"Quote error: {e}")

    st.divider()

    est_cost = (order_limit or 150.0) * order_qty
    st.caption(f"Estimated cost: **${est_cost:,.2f}** (limit price × qty)")

    col_confirm, col_warn = st.columns([1, 2])
    with col_confirm:
        if st.button("Submit Order", key="submit_order", type="primary"):
            order = Order(
                order_id="",
                symbol=order_symbol,
                side=OrderSide[order_side],
                order_type=OrderType[order_type],
                quantity=float(order_qty),
                limit_price=order_limit,
                stop_price=order_stop,
                asset_type=AssetType[order_asset],
            )

            try:
                result = broker.place_order(order)
                if result.status == OrderStatus.FILLED:
                    st.success(f"Order FILLED: {result.side.value} {result.filled_quantity} "
                               f"{result.symbol} @ ${result.filled_price:.2f}")
                elif result.status == OrderStatus.SUBMITTED:
                    st.info(f"Order submitted: {result.order_id}")
                elif result.status == OrderStatus.REJECTED:
                    st.error(f"Order REJECTED: insufficient funds or invalid parameters")
                else:
                    st.info(f"Order status: {result.status.value}")
            except Exception as e:
                st.error(f"Order failed: {e}")

    with col_warn:
        if not paper_mode and broker_name != "Paper Trading":
            st.warning("**LIVE TRADING MODE** — Real money orders will be submitted!")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 – Trade Log (Paper Broker)
# ═══════════════════════════════════════════════════════════════════════════
with tab_log:
    st.subheader("Trade Log")

    if isinstance(broker, PaperBroker):
        log = broker.get_trade_log()
        if log.empty:
            st.info("No trades executed yet.")
        else:
            st.dataframe(log, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Trade Analytics")

            buys = log[log["side"] == "BUY"]
            sells = log[log["side"] == "SELL"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Trades", len(log))
            c2.metric("Buy Trades", len(buys))
            c3.metric("Sell Trades", len(sells))
            c4.metric("Unique Symbols", log["symbol"].nunique())

            if len(log) > 1:
                symbol_volume = log.groupby("symbol")["quantity"].sum().sort_values(ascending=False)
                fig_vol = go.Figure(data=[go.Bar(
                    x=symbol_volume.index, y=symbol_volume.values,
                    marker_color="#2962FF",
                )])
                fig_vol.update_layout(
                    title="Trading Volume by Symbol",
                    xaxis_title="Symbol", yaxis_title="Total Shares",
                    height=350,
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        st.divider()
        if st.button("Reset Paper Account", type="secondary", key="reset_paper"):
            broker.reset()
            st.success("Paper account reset to initial state.")
            st.rerun()
    else:
        orders = broker.get_orders()
        filled = [o for o in orders if o.status == OrderStatus.FILLED]
        if not filled:
            st.info("No filled orders yet.")
        else:
            log_data = [{
                "Order ID": o.order_id,
                "Symbol": o.symbol,
                "Side": o.side.value,
                "Qty": o.filled_quantity,
                "Price": f"${o.filled_price:.2f}",
                "Time": o.timestamp,
            } for o in filled]
            st.dataframe(pd.DataFrame(log_data), use_container_width=True, hide_index=True)
