import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import random

# Auto-refresh every 10 seconds
st_autorefresh(interval=10000, key="datarefresh")

st.set_page_config(layout="wide", page_title="RL Trading Bot Dashboard")

# Styling
st.markdown("""
    <style>
    .small-font { font-size:14px !important; }
    .bot-face { margin-bottom: 0; }
    .heartbeat {
        position: absolute;
        top: 15px;
        right: 25px;
        font-size: 16px;
        color: #00cc66;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
      0% { opacity: 0.3; }
      50% { opacity: 1; }
      100% { opacity: 0.3; }
    }
    </style>
""", unsafe_allow_html=True)

# Heartbeat Indicator (Bot Alive)
st.markdown("<div class='heartbeat'>🧠 Bot Thinking...</div>", unsafe_allow_html=True)

# Bot Avatar and Title
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.freepik.com/free-vector/graident-ai-robot-vectorart_78370-4114.jpg", width=100, caption="RL Bot")
with col2:
    st.markdown("<h2 class='bot-face'>🤖 RL Trading Bot</h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-font'>Live market analysis and trade execution updates</div>", unsafe_allow_html=True)

# Bot Activity Feed
st.markdown("---")
st.subheader("📡 Bot Activity Feed")

status_messages = [
    "🧠 Analyzing market trends...",
    "📉 Evaluating SELL signals...",
    "📈 Checking for BUY opportunity...",
    "📊 Monitoring volatility regime...",
    "📦 Holding position...",
    "💰 Updating PnL...",
    "🔍 Watching price movement...",
    "🧠 Processing tick data...",
    "✅ No action — HOLD",
    "🚀 Momentum increasing...",
    "🔄 Preparing observation vector...",
]

current_time = datetime.now().strftime('%H:%M:%S')
random.seed(current_time)
bot_message = f"**[{current_time}]** {random.choice(status_messages)}"
st.markdown(f"<div class='small-font'>{bot_message}</div>", unsafe_allow_html=True)

# Connect to DB
DB_PATH = "trading_bot.db"
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Load data
positions = pd.read_sql_query("SELECT * FROM positions ORDER BY timestamp DESC LIMIT 1", conn)
orders = pd.read_sql_query("SELECT * FROM orders ORDER BY timestamp DESC LIMIT 10", conn)
account = pd.read_sql_query("SELECT * FROM account_snapshots ORDER BY timestamp DESC LIMIT 1000", conn)

# Account Summary
st.markdown("---")
st.subheader("📊 Account Summary")

if not account.empty:
    latest = account.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("💼 Balance", f"₹{latest['balance']:.2f}")
    col2.metric("📈 Equity", f"₹{latest['equity']:.2f}")
    col3.metric("📦 Position Value", f"₹{latest['position_value']:.2f}")
else:
    st.markdown("**Bot initializing — no account snapshot available yet.**")

# Current Position
st.markdown("---")
st.subheader("🧾 Current Position")

if not positions.empty:
    pos = positions.iloc[0]
    st.markdown(f"""
    <div class='small-font'>
    - Symbol: `{pos['symbol']}`  
    - Quantity: `{pos['quantity']}`  
    - Avg Price: ₹{pos['avg_price']:.2f}  
    - Last Price: ₹{pos['last_price']:.2f}  
    - PnL: ₹{pos['pnl']:.2f}
    </div>
    """, unsafe_allow_html=True)

    # Position PnL Chart
    st.subheader("📊 Position PnL")
    df_pnl = pd.DataFrame({
        'Symbol': [pos['symbol']],
        'PnL': [pos['pnl']]
    })
    st.bar_chart(df_pnl.set_index('Symbol'))
else:
    idle_messages = [
        "🔍 No trade yet — market unclear.",
        "📉 Waiting for better entry...",
        "🧠 Holding back — no strong signals.",
        "🕵️‍♂️ Watching patiently for volatility...",
        "🧘 Bot is calm. No rush trades."
    ]
    st.markdown(f"<div class='small-font'><i>{random.choice(idle_messages)}</i></div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='small-font'>
    - Symbol: `--`  
    - Quantity: `0`  
    - Avg Price: ₹0.00  
    - Last Price: ₹0.00  
    - PnL: ₹0.00  
    </div>
    """, unsafe_allow_html=True)

# Recent Orders
st.markdown("---")
st.subheader("📄 Recent Orders")

if not orders.empty:
    st.dataframe(orders[['timestamp', 'symbol', 'order_type', 'quantity', 'price', 'status']], use_container_width=True)
else:
    st.markdown(f"<div class='small-font'><i>🤖 No orders yet. Bot is waiting for a signal.</i></div>", unsafe_allow_html=True)
    dummy_orders = pd.DataFrame([{
        'timestamp': current_time,
        'symbol': '—',
        'order_type': '—',
        'quantity': 0,
        'price': 0.00,
        'status': '—'
    }] * 5)
    st.dataframe(dummy_orders, use_container_width=True)

# Equity Curve
st.markdown("---")
st.subheader("📈 Equity Curve")

if not account.empty:
    account["timestamp"] = pd.to_datetime(account["timestamp"])
    st.line_chart(account.set_index("timestamp")[['balance', 'equity', 'position_value']])
else:
    st.markdown("📉 No equity data yet.")

conn.close()


# streamlit run dashboard.py