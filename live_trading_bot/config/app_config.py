"""
Application configuration for the RL Trading Bot.
"""
import os
from dotenv import load_dotenv
from pytz import timezone

# Load environment variables from .env file
load_dotenv()

class AppConfig:
    """Application configuration class"""
    
    def __init__(self, config_path=None):
        # If a specific config path is provided, load it
        if config_path and config_path != 'config/app_config.py':
            # Here you could load a different config file
            pass
            
        # Base directories
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.LOG_DIR = os.path.join(self.BASE_DIR, 'logs')
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'models/saved')
        self.DB_DIR = os.path.join(self.BASE_DIR, 'db')
        
        # Ensure directories exist
        for directory in [self.LOG_DIR, self.DATA_DIR, self.MODEL_DIR, self.DB_DIR]:
            os.makedirs(directory, exist_ok=True)
        
        # Database settings
        self.DB_PATH = os.path.join(self.DB_DIR, 'trading.db')
        
        # Zerodha API credentials
        self.ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY')
        self.ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET')
        self.ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN', None)
        self.ZERODHA_TOTP_SECRET = os.getenv('ZERODHA_TOTP_SECRET', None)
        self.ZERODHA_REDIRECT_URL = os.getenv('ZERODHA_REDIRECT_URL', 'https://127.0.0.1/login')
        
        # Trading settings
        self.DEFAULT_INITIAL_BALANCE = float(os.getenv('DEFAULT_INITIAL_BALANCE', '100000'))
        self.BROKERAGE_INTRADAY = float(os.getenv('BROKERAGE_INTRADAY', '0.0003'))
        self.BROKERAGE_DELIVERY = float(os.getenv('BROKERAGE_DELIVERY', '0.0'))
        self.STT_INTRADAY = float(os.getenv('STT_INTRADAY', '0.00025'))
        self.STT_DELIVERY = float(os.getenv('STT_DELIVERY', '0.001'))
        self.EXCHANGE_TXN_CHARGE = float(os.getenv('EXCHANGE_TXN_CHARGE', '0.0000345'))
        self.SEBI_CHARGES = float(os.getenv('SEBI_CHARGES', '0.000001'))
        self.STAMP_DUTY = float(os.getenv('STAMP_DUTY', '0.00015'))
        self.GST = float(os.getenv('GST', '0.18'))
        
        # Risk management
        self.MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '0.40'))
        self.MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '8'))
        self.MAX_TRADES_PER_DAY = int(os.getenv('MAX_TRADES_PER_DAY', '10'))
        self.DAILY_LOSS_LIMIT_PCT = float(os.getenv('DAILY_LOSS_LIMIT_PCT', '0.035'))
        self.DAILY_PROFIT_TARGET_PCT = float(os.getenv('DAILY_PROFIT_TARGET_PCT', '0.10'))
        self.MAX_DRAWDOWN_ALLOWED = float(os.getenv('MAX_DRAWDOWN_ALLOWED', '0.15'))
        
        # Trading modes
        self.SWING_TRADING_MODE = os.getenv('SWING_TRADING_MODE', 'True').lower() == 'true'
        self.USE_MARKET_REGIME = os.getenv('USE_MARKET_REGIME', 'True').lower() == 'true'
        self.USE_BOLLINGER_BREAKOUTS = os.getenv('USE_BOLLINGER_BREAKOUTS', 'True').lower() == 'true'
        self.USE_RSI_FILTER = os.getenv('USE_RSI_FILTER', 'True').lower() == 'true'
        self.USE_VOLUME_FILTER = os.getenv('USE_VOLUME_FILTER', 'True').lower() == 'true'
        self.USE_PROFIT_TRAILING = os.getenv('USE_PROFIT_TRAILING', 'True').lower() == 'true'
        self.USE_MULTIPLE_TIMEFRAMES = os.getenv('USE_MULTIPLE_TIMEFRAMES', 'True').lower() == 'true'
        
        # Market timing settings
        self.MARKET_OPEN_HOUR = int(os.getenv('MARKET_OPEN_HOUR', '9'))
        self.MARKET_OPEN_MINUTE = int(os.getenv('MARKET_OPEN_MINUTE', '15'))
        self.MARKET_CLOSE_HOUR = int(os.getenv('MARKET_CLOSE_HOUR', '15'))
        self.MARKET_CLOSE_MINUTE = int(os.getenv('MARKET_CLOSE_MINUTE', '30'))
        self.AVOID_FIRST_MINUTES = int(os.getenv('AVOID_FIRST_MINUTES', '15'))
        self.AVOID_LAST_MINUTES = int(os.getenv('AVOID_LAST_MINUTES', '15'))
        
        # Timezone
        self.TIMEZONE = timezone(os.getenv('TIMEZONE', 'Asia/Kolkata'))
        
        # Symbols
        self.SYMBOLS = os.getenv('SYMBOLS', 'ASIANPAINT,BAJFINANCE').split(',')
        
        # Technical parameters
        self.WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '20'))
        self.REGIME_LOOKBACK = int(os.getenv('REGIME_LOOKBACK', '20'))
        self.ATR_PERIODS = int(os.getenv('ATR_PERIODS', '14'))
        
        # Scheduler settings
        self.DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', '60'))  # seconds
        self.TRADING_DECISION_INTERVAL = int(os.getenv('TRADING_DECISION_INTERVAL', '300'))  # seconds
        self.POSITION_CHECK_INTERVAL = int(os.getenv('POSITION_CHECK_INTERVAL', '60'))  # seconds
        
        # Notification settings
        self.ENABLE_EMAIL_NOTIFICATIONS = os.getenv('ENABLE_EMAIL_NOTIFICATIONS', 'False').lower() == 'true'
        self.EMAIL_SERVER = os.getenv('EMAIL_SERVER', 'smtp.gmail.com')
        self.EMAIL_PORT = int(os.getenv('EMAIL_PORT', '587'))
        self.EMAIL_USE_TLS = os.getenv('EMAIL_USE_TLS', 'True').lower() == 'true'
        self.EMAIL_USERNAME = os.getenv('EMAIL_USERNAME', '')
        self.EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', '')
        self.EMAIL_RECIPIENTS = os.getenv('EMAIL_RECIPIENTS', '').split(',')
        
        # Telegram notifications
        self.ENABLE_TELEGRAM_NOTIFICATIONS = os.getenv('ENABLE_TELEGRAM_NOTIFICATIONS', 'False').lower() == 'true'
        self.TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')