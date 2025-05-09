import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API credentials
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")

# Default trading parameters
DEFAULT_INITIAL_BALANCE = float(os.getenv("DEFAULT_INITIAL_BALANCE", "100000"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "saved_models")
DATA_PATH = os.getenv("DATA_PATH", "data/historical/{}.csv")
RESULTS_SAVE_PATH = os.getenv("RESULTS_SAVE_PATH", "results")

# Transaction cost parameters
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY", "0.0003"))  # 0.03%
BROKERAGE_DELIVERY = float(os.getenv("BROKERAGE_DELIVERY", "0.0"))     # 0% for delivery
STT_INTRADAY = float(os.getenv("STT_INTRADAY", "0.00025"))            # 0.025%
STT_DELIVERY = float(os.getenv("STT_DELIVERY", "0.001"))              # 0.1%
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE", "0.0000345"))  # 0.00345%
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES", "0.000001"))           # 0.0001%
STAMP_DUTY = float(os.getenv("STAMP_DUTY", "0.00015"))                # 0.015%
GST = float(os.getenv("GST", "0.18"))                                 # 18%

# Tax Settings
SHORT_TERM_CAPITAL_GAINS_TAX = float(os.getenv("SHORT_TERM_CAPITAL_GAINS_TAX", "0.15"))  # 15%
LONG_TERM_CAPITAL_GAINS_TAX = float(os.getenv("LONG_TERM_CAPITAL_GAINS_TAX", "0.10"))   # 10%
LONG_TERM_THRESHOLD_DAYS = float(os.getenv("LONG_TERM_THRESHOLD_DAYS", "365"))         # 1 year

# Position and risk management parameters
MAX_POSITION_SIZE_PCT = float(os.getenv("MAX_POSITION_SIZE_PCT", "0.40"))       # 40% of account
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "8"))                  # 8 positions max
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "10"))                 # 10 trades per day
DAILY_LOSS_LIMIT_PCT = float(os.getenv("DAILY_LOSS_LIMIT_PCT", "0.035"))        # 3.5% daily loss limit
DAILY_PROFIT_TARGET_PCT = float(os.getenv("DAILY_PROFIT_TARGET_PCT", "0.10"))   # 10% daily profit target
MAX_DRAWDOWN_ALLOWED = float(os.getenv("MAX_DRAWDOWN_ALLOWED", "0.15"))        # 15% max drawdown

# Trading mode configurations
SWING_TRADING_MODE = os.getenv("SWING_TRADING_MODE", "True").lower() in ['true', 'yes', '1']
USE_MARKET_REGIME = os.getenv("USE_MARKET_REGIME", "True").lower() in ['true', 'yes', '1']
USE_BOLLINGER_BREAKOUTS = os.getenv("USE_BOLLINGER_BREAKOUTS", "True").lower() in ['true', 'yes', '1']
USE_RSI_FILTER = os.getenv("USE_RSI_FILTER", "True").lower() in ['true', 'yes', '1'] 
USE_VOLUME_FILTER = os.getenv("USE_VOLUME_FILTER", "True").lower() in ['true', 'yes', '1']
USE_PROFIT_TRAILING = os.getenv("USE_PROFIT_TRAILING", "True").lower() in ['true', 'yes', '1']
USE_MULTIPLE_TIMEFRAMES = os.getenv("USE_MULTIPLE_TIMEFRAMES", "True").lower() in ['true', 'yes', '1']

# Default symbols list
DEFAULT_SYMBOLS = [
    "ASIANPAINT", "BAJFINANCE", "BHARTIARTL", "BRITANNIA", "TATAMOTORS", 
    "HCLTECH", "TITAN", "CIPLA", "IOC", "SUNPHARMA", "DRREDDY", "KOTAKBANK", 
    "AXISBANK", "ITC", "SBIN", "INFY", "TCS", "RELIANCE", "LT", "HINDUNILVR", 
    "MARUTI", "ICICIBANK", "TECHM", "BPCL", "TATASTEEL", "HEROMOTOCO", "ONGC", "WIPRO"
]

# Data fetching parameters
HISTORICAL_LOOKBACK_DAYS = int(os.getenv("HISTORICAL_LOOKBACK_DAYS", "365"))  # 1 year
TICKS_LOOKBACK_MINUTES = int(os.getenv("TICKS_LOOKBACK_MINUTES", "30"))       # 30 minutes
REALTIME_UPDATE_INTERVAL = int(os.getenv("REALTIME_UPDATE_INTERVAL", "60"))   # 60 seconds
CANDLE_TIMEFRAME = os.getenv("CANDLE_TIMEFRAME", "5minute")                  # 5-minute candles

# System operation settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DB_PATH = os.getenv("DB_PATH", "data/trading.db")
ENABLE_NOTIFICATIONS = os.getenv("ENABLE_NOTIFICATIONS", "True").lower() in ['true', 'yes', '1']
ENABLE_DASHBOARD = os.getenv("ENABLE_DASHBOARD", "True").lower() in ['true', 'yes', '1']
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8050"))

# Reinforcement Learning model parameters
RL_ENABLED = os.getenv("RL_ENABLED", "True").lower() in ['true', 'yes', '1']
RL_MODEL_UPDATE_FREQUENCY = int(os.getenv("RL_MODEL_UPDATE_FREQUENCY", "7"))  # Days
RL_TRAINING_TIMESTEPS = int(os.getenv("RL_TRAINING_TIMESTEPS", "300000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "2048"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "3e-5"))