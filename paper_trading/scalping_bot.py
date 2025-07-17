import numpy as np
import pandas as pd
import os
import logging
import pytz
import sqlite3
from stable_baselines3 import PPO
from datetime import datetime, timedelta
from kiteconnect import KiteConnect, KiteTicker
import time
from kiteconnect import exceptions
import json
from finta import TA
import argparse
from dotenv import load_dotenv
import queue
import threading

# Selenium imports for token management
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import pyotp
import traceback

# Email imports
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scalping_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Scalping_Bot")

# Environment variables
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "scalping_models")  # Updated path

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

# Scalping specific parameters (matching training code)
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "100000"))
SCALP_TARGET_PROFIT = float(os.getenv("SCALP_TARGET_PROFIT", "0.5"))  # 0.5% target profit
SCALP_STOP_LOSS = float(os.getenv("SCALP_STOP_LOSS", "0.3"))  # 0.3% stop loss
MAX_POSITION_TIME_BARS = int(os.getenv("MAX_POSITION_TIME_BARS", "12"))  # 12 bars = 1 hour
POSITION_SIZE_PERCENT = float(os.getenv("SCALP_POSITION_SIZE_PCT", "50"))  # 50% of balance
PREDICTION_INTERVAL = int(os.getenv("PREDICTION_INTERVAL", "30"))  # 30 seconds for scalping

# Trading parameters
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY", "0.0003"))
STT_INTRADAY = float(os.getenv("STT_INTRADAY", "0.00025"))
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE", "0.0000345"))
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES", "0.000001"))
STAMP_DUTY = float(os.getenv("STAMP_DUTY", "0.00015"))
GST = float(os.getenv("GST", "0.18"))

# Constants
TIMEZONE = pytz.timezone('Asia/Kolkata')
DB_PATH = "scalping_bot.db"

class EmailNotificationManager:
    """Handles all email notifications for scalping"""
    
    def __init__(self):
        self.email_host = EMAIL_HOST
        self.email_port = EMAIL_PORT
        self.email_user = EMAIL_USER
        self.email_password = EMAIL_PASSWORD
        self.email_to = EMAIL_TO
        
        if not all([self.email_user, self.email_password, self.email_to]):
            logger.warning("Email configuration incomplete. Email notifications disabled.")
            self.email_enabled = False
        else:
            self.email_enabled = True
            logger.info("✅ Email notifications enabled")
    
    def send_email(self, subject, body, is_html=False):
        """Send email notification"""
        if not self.email_enabled:
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.email_to
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'html' if is_html else 'plain'))
            
            server = smtplib.SMTP(self.email_host, self.email_port)
            server.starttls()
            server.login(self.email_user, self.email_password)
            
            text = msg.as_string()
            server.sendmail(self.email_user, self.email_to, text)
            server.quit()
            
            logger.info(f"✅ Email sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            return False
    
    def send_login_success(self, symbol, trading_mode):
        """Send login success notification"""
        subject = f"🔥 Scalping Bot - Login Successful ({symbol})"
        body = f"""
        <h2>🔥 Scalping Bot - Login Successful</h2>
        <p><strong>Symbol:</strong> {symbol}</p>
        <p><strong>Trading Mode:</strong> {'Live Trading' if not trading_mode else 'Paper Trading'}</p>
        <p><strong>Timeframe:</strong> 5-minute charts</p>
        <p><strong>Strategy:</strong> High-frequency scalping</p>
        <p><strong>Login Time:</strong> {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</p>
        
        <hr>
        <p><strong>Scalping Parameters:</strong></p>
        <p>• Target Profit: {SCALP_TARGET_PROFIT}%</p>
        <p>• Stop Loss: {SCALP_STOP_LOSS}%</p>
        <p>• Max Hold Time: {MAX_POSITION_TIME_BARS} bars (1 hour)</p>
        <p>• Decision Interval: {PREDICTION_INTERVAL} seconds</p>
        <p>• Position Size: {POSITION_SIZE_PERCENT}% of balance</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_scalp_order_notification(self, order, symbol, trading_mode, entry_type="NEW"):
        """Send scalping order notification"""
        order_type = order['type']
        emoji = "🟢" if order_type == "BUY" else "🔴"
        
        subject = f"{emoji} SCALP {entry_type} {order_type} - {symbol}"
        
        body = f"""
        <h2>{emoji} Scalp {entry_type} {order_type} Order</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Action</strong></td><td>SCALP {entry_type} {order_type}</td></tr>
            <tr><td><strong>Quantity</strong></td><td>{order['quantity']}</td></tr>
            <tr><td><strong>Price</strong></td><td>₹{order['price']:.2f}</td></tr>
            <tr><td><strong>Total Value</strong></td><td>₹{order['quantity'] * order['price']:.2f}</td></tr>
            <tr><td><strong>Status</strong></td><td>{order['status']}</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%H:%M:%S IST')}</td></tr>
        </table>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_scalp_exit_notification(self, exit_reason, pnl, duration, symbol, trading_mode):
        """Send scalping exit notification"""
        emoji = "💰" if pnl > 0 else "💸"
        
        subject = f"{emoji} Scalp Exit: {exit_reason} - {symbol}"
        
        body = f"""
        <h2>{emoji} Scalp Trade Completed</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Exit Reason</strong></td><td>{exit_reason}</td></tr>
            <tr><td><strong>Trade PnL</strong></td><td style="color: {'green' if pnl > 0 else 'red'}">₹{pnl:.2f}</td></tr>
            <tr><td><strong>Duration</strong></td><td>{duration:.1f} bars</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%H:%M:%S IST')}</td></tr>
        </table>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_scalping_summary(self, symbol, trades_count, total_pnl, win_rate, avg_duration, trading_mode):
        """Send scalping session summary"""
        emoji = "📊"
        
        subject = f"{emoji} Scalping Summary - {symbol}"
        
        body = f"""
        <h2>{emoji} Scalping Session Summary</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Total Trades</strong></td><td>{trades_count}</td></tr>
            <tr><td><strong>Total PnL</strong></td><td style="color: {'green' if total_pnl > 0 else 'red'}">₹{total_pnl:.2f}</td></tr>
            <tr><td><strong>Win Rate</strong></td><td>{win_rate:.1f}%</td></tr>
            <tr><td><strong>Avg Duration</strong></td><td>{avg_duration:.1f} bars</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Paper Trading' if trading_mode else 'Live Trading'}</td></tr>
        </table>
        """
        self.send_email(subject, body, is_html=True)

class ZerodhaTokenManager:
    """Token management for scalping bot"""
    
    def __init__(self, email_manager):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.username = os.getenv("ZERODHA_USER_ID")
        self.password = os.getenv("ZERODHA_PASSWORD")
        self.totp_secret = os.getenv("ZERODHA_TOTP_SECRET")
        self.email_manager = email_manager
        
        if not all([self.api_key, self.api_secret, self.username, self.password, self.totp_secret]):
            raise ValueError("Missing required Zerodha credentials")
        
        logger.info("✅ ZerodhaTokenManager initialized for scalping")
    
    def generate_totp(self):
        """Generate TOTP code"""
        try:
            totp = pyotp.TOTP(self.totp_secret)
            code = totp.now()
            return code
        except Exception as e:
            logger.error(f"TOTP generation failed: {e}")
            raise
    
    def setup_chrome_driver(self):
        """Setup headless Chrome driver"""
        chrome_options = Options()
        chrome_options.binary_location = "/usr/bin/chromium-browser"
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            logger.error(f"Chrome driver setup failed: {e}")
            raise
    
    def auto_login_and_get_token(self):
        """Automated login for scalping bot"""
        driver = None
        
        try:
            driver = self.setup_chrome_driver()
            login_url = f"https://kite.zerodha.com/connect/login?api_key={self.api_key}&v=3"
            driver.get(login_url)
            time.sleep(3)
            
            # Login process
            wait = WebDriverWait(driver, 20)
            username_field = wait.until(EC.presence_of_element_located((By.ID, "userid")))
            username_field.send_keys(self.username)
            
            password_field = driver.find_element(By.ID, "password")
            password_field.send_keys(self.password)
            
            login_button = driver.find_element(By.XPATH, "//button[contains(text(),'Login')]")
            login_button.click()
            time.sleep(5)
            
            # Handle 2FA if needed
            current_url = driver.current_url
            if "request_token=" in current_url:
                request_token = current_url.split("request_token=")[1].split("&")[0]
            else:
                # Handle TOTP
                totp_field = driver.find_element(By.XPATH, "//input[@type='number' and @maxlength='6']")
                totp_code = self.generate_totp()
                totp_field.send_keys(totp_code)
                
                continue_button = driver.find_element(By.XPATH, "//button[@type='submit']")
                continue_button.click()
                
                wait.until(lambda d: "request_token=" in d.current_url)
                current_url = driver.current_url
                request_token = current_url.split("request_token=")[1].split("&")[0]
            
            # Generate access token
            kite = KiteConnect(api_key=self.api_key)
            data = kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            
            with open("access_token.txt", "w") as f:
                f.write(access_token)
            
            return access_token
            
        except Exception as e:
            logger.error(f"Auto-login failed: {e}")
            return None
        finally:
            if driver:
                driver.quit()
    
    def refresh_token_if_needed(self):
        """Get valid access token"""
        if os.path.exists("access_token.txt"):
            try:
                with open("access_token.txt", "r") as f:
                    token = f.read().strip()
                
                # Test token validity
                kite = KiteConnect(api_key=self.api_key)
                kite.set_access_token(token)
                kite.profile()
                
                logger.info("✅ Existing token valid")
                return token
                
            except exceptions.TokenException:
                logger.info("Token expired, refreshing...")
            except Exception as e:
                logger.error(f"Error checking token: {e}")
        
        return self.auto_login_and_get_token()

class ScalpingBot:
    def __init__(self, model, symbol, paper_trading=True):
        self.model = model
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.kite = None
        self.ticker = None
        self.token = None
        
        # Scalping state - matching training environment
        self.balance = INITIAL_BALANCE
        self.initial_balance = INITIAL_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.position_entry_time = 0  # Track in bars like training
        self.current_price = 0
        self.current_bar = 0  # Track current bar number
        
        # Scalping tracking - matching training environment
        self.total_trades = 0
        self.scalp_wins = 0
        self.scalp_losses = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Scalping data
        self.candle_data = pd.DataFrame()
        self.data_queue = queue.Queue()
        self.last_prediction_time = None
        
        # Trade tracking
        self.trades_history = []
        self.daily_trades = 0
        self.daily_pnl = 0
        
        # Initialize managers
        self.email_manager = EmailNotificationManager()
        self.token_manager = ZerodhaTokenManager(self.email_manager)
        
        # Database
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()
        
        # Setup connection
        self.setup_zerodha()
        
        logger.info(f"🔥 Scalping Bot initialized for {symbol}")
        logger.info(f"📊 5-minute timeframe scalping strategy")
        logger.info(f"💰 Target: {SCALP_TARGET_PROFIT}% | Stop: {SCALP_STOP_LOSS}%")

    def create_tables(self):
        """Create database tables for scalping"""
        cursor = self.db_conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scalp_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl REAL,
                duration_bars INTEGER,
                exit_reason TEXT,
                paper_trading BOOLEAN
            )''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candle_data_5m (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )''')
        
        self.db_conn.commit()
        logger.info("✅ Scalping database tables created")

    def setup_zerodha(self):
        """Setup Zerodha for scalping"""
        try:
            self.kite = KiteConnect(api_key=API_KEY)
            access_token = self.token_manager.refresh_token_if_needed()
            
            if not access_token:
                raise ValueError("Failed to get access token")
            
            self.kite.set_access_token(access_token)
            
            # Get instrument token
            instruments = self.kite.instruments("NSE")
            for inst in instruments:
                if inst['tradingsymbol'] == self.symbol:
                    self.token = inst['instrument_token']
                    break
            
            if not self.token:
                raise ValueError(f"Token not found for {self.symbol}")
            
            # Setup WebSocket
            self.ticker = KiteTicker(API_KEY, access_token)
            self.ticker.on_ticks = self.on_ticks
            self.ticker.on_connect = self.on_connect
            
            logger.info(f"✅ Scalping bot setup complete. Token: {self.token}")
            self.email_manager.send_login_success(self.symbol, self.paper_trading)
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            if not self.paper_trading:
                raise

    def on_connect(self, ws, response):
        """WebSocket connection handler"""
        logger.info("🔌 WebSocket connected for scalping")
        ws.subscribe([self.token])
        ws.set_mode(ws.MODE_QUOTE, [self.token])

    def on_ticks(self, ws, ticks):
        """Process real-time ticks for 5-minute candles"""
        for tick in ticks:
            if tick['instrument_token'] == self.token:
                self.current_price = tick['last_price']
                
                tick_data = {
                    'timestamp': datetime.now(TIMEZONE),
                    'price': tick['last_price'],
                    'volume': tick.get('volume_today', 0),
                    'high': tick['ohlc']['high'],
                    'low': tick['ohlc']['low'],
                    'open': tick['ohlc']['open']
                }
                self.data_queue.put(tick_data)

    def create_5min_candles(self):
        """Create 5-minute OHLCV candles from tick data"""
        try:
            if self.data_queue.empty():
                return
            
            # Process queued tick data
            tick_data = []
            while not self.data_queue.empty():
                tick_data.append(self.data_queue.get())
            
            if not tick_data:
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(tick_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create 5-minute intervals
            df['5min_interval'] = df['timestamp'].dt.floor('5min')
            
            # Group by 5-minute intervals to create candles
            candles = df.groupby('5min_interval').agg({
                'price': ['first', 'max', 'min', 'last'],
                'volume': 'last'
            }).reset_index()
            
            # Flatten columns
            candles.columns = ['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Update candle data and increment bar count
            if not candles.empty:
                # Calculate indicators before adding new data
                if len(self.candle_data) > 0:
                    self.candle_data = pd.concat([self.candle_data, candles], ignore_index=True)
                else:
                    self.candle_data = candles.copy()
                
                # Calculate scalping indicators
                self.candle_data = self.calculate_scalping_indicators(self.candle_data.copy())
                
                # Keep only last 100 bars for performance
                self.candle_data = self.candle_data.drop_duplicates(subset=['timestamp']).tail(100).reset_index(drop=True)
                
                # Update current bar number
                self.current_bar = len(self.candle_data) - 1
                
                # Save to database
                self.save_candle_data(candles)
                
        except Exception as e:
            logger.error(f"Error creating 5min candles: {e}")

    def save_candle_data(self, candles):
        """Save 5-minute candle data"""
        try:
            cursor = self.db_conn.cursor()
            for _, row in candles.iterrows():
                cursor.execute('''
                    INSERT OR REPLACE INTO candle_data_5m 
                    (timestamp, symbol, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    row['timestamp'].isoformat(), self.symbol,
                    row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
                ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving candle data: {e}")

    def calculate_scalping_indicators(self, df):
        """Calculate indicators exactly matching training code"""
        try:
            if len(df) < 30:
                return df
            
            # Rename columns to match training code expectations
            df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
            
            # Fast EMAs for scalping (matching training code)
            for period in [3, 5, 8, 13, 21]:
                df[f'EMA{period}'] = df['close'].ewm(span=period).mean()
                df[f'SMA{period}'] = df['close'].rolling(window=period).mean()
            
            # Fast RSI (5 and 14 period) - matching training
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
            rs = gain / loss
            df['RSI5'] = 100 - (100 / (1 + rs))
            
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Fast Stochastic
            high_5 = df['high'].rolling(5).max()
            low_5 = df['low'].rolling(5).min()
            df['STOCH_K'] = ((df['close'] - low_5) / (high_5 - low_5)) * 100
            df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
            
            # MACD with faster settings
            ema_fast = df['close'].ewm(span=5).mean()
            ema_slow = df['close'].ewm(span=13).mean()
            df['MACD'] = ema_fast - ema_slow
            df['Signal'] = df['MACD'].ewm(span=3).mean()
            
            # Bollinger Bands with tighter settings
            sma_bb = df['close'].rolling(10).mean()
            std_bb = df['close'].rolling(10).std()
            df['BB_Upper'] = sma_bb + 1.5 * std_bb
            df['BB_Middle'] = sma_bb
            df['BB_Lower'] = sma_bb - 1.5 * std_bb
            
            # Volume indicators
            df['OBV'] = df['volume'].cumsum()  # Simplified OBV
            df['Volume_SMA'] = df['volume'].rolling(20).mean()
            
            # Price action
            df['High_Low_Pct'] = (df['high'] - df['low']) / df['close'] * 100
            df['Body_Size'] = abs(df['close'] - df['open']) / df['close'] * 100
            
            # Support/Resistance
            df['Resistance_5'] = df['high'].rolling(5).max()
            df['Support_5'] = df['low'].rolling(5).min()
            
            # Volatility
            df['Volatility'] = df['close'].pct_change().rolling(10).std() * 100
            
            # EMA crossover signals
            df['ema_bullish'] = (df['EMA_3'] > df['EMA_8']) & (df['EMA_8'] > df['EMA_13'])
            df['ema_bearish'] = (df['EMA_3'] < df['EMA_8']) & (df['EMA_8'] < df['EMA_13'])
            
            # Fill NaN values
            df = df.bfill().ffill().fillna(0)
            
            # Convert back to uppercase columns for consistency
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df

    def prepare_scalping_observation(self):
        """Prepare observation matching exactly the training code (65 features)"""
        try:
            if len(self.candle_data) < 30:
                return None
            
            window_size = 20
            if self.current_bar < window_size:
                return None
            
            current_price = self.current_price
            
            # Basic account state (8 features) - exactly matching training
            obs = np.array([
                self.balance / self.initial_balance,
                self.shares_held / 100,
                self.cost_basis / current_price if self.cost_basis > 0 else 0,
                current_price / self.candle_data["Close"].iloc[max(0, self.current_bar-10):self.current_bar+1].mean(),
                1 if self.shares_held > 0 else 0,  # In position
                (self.current_bar - self.position_entry_time) / MAX_POSITION_TIME_BARS if self.shares_held > 0 else 0,
                self.consecutive_losses / 10,  # Normalize consecutive losses
                self.scalp_wins / max(1, self.scalp_wins + self.scalp_losses),  # Win rate
            ])
            
            # Get window data
            window_data = self.candle_data.iloc[self.current_bar-window_size+1:self.current_bar+1]
            
            # Price and volume data (5 features)
            close_mean = window_data["Close"].mean()
            volume_mean = window_data["Volume"].mean()
            
            obs = np.append(obs, [
                window_data["Open"].iloc[-1] / close_mean,
                window_data["High"].iloc[-1] / close_mean,
                window_data["Low"].iloc[-1] / close_mean,
                window_data["Close"].iloc[-1] / close_mean,
                window_data["Volume"].iloc[-1] / volume_mean if volume_mean > 0 else 1,
            ])
            
            # Fast EMAs for scalping (4 features)
            for period in [3, 5, 8, 13]:
                ema_col = f'EMA{period}'
                if ema_col in window_data.columns:
                    ema = window_data[ema_col].iloc[-1] / close_mean
                else:
                    ema = window_data["Close"].ewm(span=period).mean().iloc[-1] / close_mean
                obs = np.append(obs, [ema])
            
            # Short-term SMAs (4 features)
            for period in [3, 5, 8, 13]:
                sma_col = f'SMA{period}'
                if sma_col in window_data.columns:
                    sma = window_data[sma_col].iloc[-1] / close_mean
                else:
                    sma = window_data["Close"].rolling(period).mean().iloc[-1] / close_mean
                obs = np.append(obs, [sma])
            
            # Momentum indicators (6 features)
            price_changes = window_data["Close"].pct_change()
            obs = np.append(obs, [
                price_changes.iloc[-1],  # Last 1-bar change
                price_changes.iloc[-3:].mean(),  # 3-bar average change
                price_changes.iloc[-5:].mean(),  # 5-bar average change
                price_changes.iloc[-3:].std(),  # 3-bar volatility
                (window_data["High"].iloc[-1] - window_data["Low"].iloc[-1]) / close_mean,  # Current bar range
                window_data["Close"].iloc[-5:].std() / close_mean,  # 5-bar price volatility
            ])
            
            # Volume indicators (4 features)
            volume_changes = window_data["Volume"].pct_change()
            obs = np.append(obs, [
                volume_changes.iloc[-1],  # Volume change
                window_data["Volume"].iloc[-3:].mean() / volume_mean if volume_mean > 0 else 1,
                (window_data["Volume"].iloc[-1] > window_data["Volume"].rolling(5).mean().iloc[-1]) * 1.0,
                window_data["Volume"].iloc[-3:].std() / volume_mean if volume_mean > 0 else 0,
            ])
            
            # RSI and fast oscillators (3 features)
            if 'RSI' in window_data.columns:
                rsi = window_data['RSI'].iloc[-1] / 100
            else:
                rsi = 0.5
            
            if 'STOCH_K' in window_data.columns:
                stoch_k = window_data['STOCH_K'].iloc[-1] / 100
                stoch_d = window_data['STOCH_D'].iloc[-1] / 100
            else:
                stoch_k = 0.5
                stoch_d = 0.5
            
            obs = np.append(obs, [rsi, stoch_k, stoch_d])
            
            # MACD for trend (2 features)
            if 'MACD' in window_data.columns:
                macd = window_data['MACD'].iloc[-1] / close_mean
                signal = window_data['Signal'].iloc[-1] / close_mean
            else:
                macd = 0.5
                signal = 0.5
            
            obs = np.append(obs, [macd, signal])
            
            # Bollinger Bands (3 features)
            if all(x in window_data.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                bb_upper = window_data['BB_Upper'].iloc[-1] / close_mean
                bb_middle = window_data['BB_Middle'].iloc[-1] / close_mean
                bb_lower = window_data['BB_Lower'].iloc[-1] / close_mean
            else:
                bb_upper = 1.02
                bb_middle = 1.0
                bb_lower = 0.98
            
            obs = np.append(obs, [bb_upper, bb_middle, bb_lower])
            
            # Support/Resistance levels (4 features)
            if 'Resistance_5' in window_data.columns and 'Support_5' in window_data.columns:
                resistance = window_data['Resistance_5'].iloc[-1] / close_mean
                support = window_data['Support_5'].iloc[-1] / close_mean
            else:
                resistance = 1.01
                support = 0.99
            
            # Distance to support/resistance
            dist_to_resistance = (resistance * close_mean - current_price) / current_price
            dist_to_support = (current_price - support * close_mean) / current_price
            
            obs = np.append(obs, [resistance, support, dist_to_resistance, dist_to_support])
            
            # Time-based features (3 features)
            bars_from_open = self.current_bar % 75  # 375 minutes / 5 = 75 bars per day
            session_progress = bars_from_open / 75
            
            # Volatility regime
            if 'Volatility' in window_data.columns:
                current_volatility = window_data['Volatility'].iloc[-1]
                avg_volatility = self.candle_data['Volatility'].rolling(100).mean().iloc[self.current_bar] if len(self.candle_data) > 100 else 1
                volatility_regime = current_volatility / avg_volatility if avg_volatility > 0 else 1
            else:
                volatility_regime = 1.0
            
            obs = np.append(obs, [session_progress, volatility_regime, bars_from_open / 75])
            
            # EMA crossover signals (4 features)
            if all(f'EMA{p}' in window_data.columns for p in [3, 5, 8, 13]):
                ema3 = window_data['EMA3'].iloc[-1]
                ema5 = window_data['EMA5'].iloc[-1]
                ema8 = window_data['EMA8'].iloc[-1]
                ema13 = window_data['EMA13'].iloc[-1]
                
                bullish_cross = (ema3 > ema5 > ema8) * 1.0
                bearish_cross = (ema3 < ema5 < ema8) * 1.0
                trend_strength = abs(ema3 - ema13) / close_mean
                trend_direction = (ema3 > ema13) * 1.0
            else:
                bullish_cross = 0.0
                bearish_cross = 0.0
                trend_strength = 0.0
                trend_direction = 0.5
            
            obs = np.append(obs, [bullish_cross, bearish_cross, trend_strength, trend_direction])
            
            # Price action patterns (3 features)
            body_size = abs(window_data['Close'].iloc[-1] - window_data['Open'].iloc[-1]) / close_mean
            upper_shadow = (window_data['High'].iloc[-1] - max(window_data['Close'].iloc[-1], window_data['Open'].iloc[-1])) / close_mean
            lower_shadow = (min(window_data['Close'].iloc[-1], window_data['Open'].iloc[-1]) - window_data['Low'].iloc[-1]) / close_mean
            
            obs = np.append(obs, [body_size, upper_shadow, lower_shadow])
            
            # Ensure exactly 65 features
            if len(obs) < 65:
                obs = np.append(obs, np.full(65 - len(obs), 0.5))
            elif len(obs) > 65:
                obs = obs[:65]
            
            return np.nan_to_num(obs).astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            return None

    def calculate_position_size(self):
        """Calculate position size for scalping"""
        if self.current_price <= 0:
            return 0
        
        # Use percentage of capital for scalping (matching training)
        risk_amount = self.balance * (POSITION_SIZE_PERCENT / 100)
        position_size = int(risk_amount / self.current_price)
        
        return max(position_size, 1) if position_size > 0 else 0

    def place_scalp_order(self, order_type, quantity):
        """Place scalping order"""
        if quantity <= 0:
            return None
        
        price = self.current_price
        trade_value = quantity * price
        
        order = {
            'symbol': self.symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(TIMEZONE),
            'status': 'PENDING'
        }
        
        if self.paper_trading:
            # Paper trading execution (matching training environment logic)
            if order_type == "BUY" and self.shares_held == 0:
                total_cost = trade_value
                if self.balance >= total_cost:
                    self.balance -= total_cost
                    self.shares_held = quantity
                    self.cost_basis = price
                    self.position_entry_time = self.current_bar  # Track in bars
                    self.total_trades += 1
                    order['status'] = 'COMPLETED'
                    logger.info(f"🟢 SCALP BUY: {quantity} @ ₹{price:.2f}")
                else:
                    order['status'] = 'REJECTED'
                    
            elif order_type == "SELL" and self.shares_held > 0:
                self.balance += trade_value
                
                # Calculate trade PnL (matching training environment)
                profit = trade_value - (self.shares_held * self.cost_basis)
                profit_pct = profit / (self.shares_held * self.cost_basis) * 100 if self.cost_basis > 0 else 0
                holding_time_bars = self.current_bar - self.position_entry_time
                
                # Update win/loss tracking
                if profit > 0:
                    self.scalp_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.scalp_losses += 1
                    self.consecutive_losses += 1
                
                # Record trade
                self.record_scalp_trade(profit, holding_time_bars, "MANUAL_EXIT")
                
                self.shares_held = 0
                self.cost_basis = 0
                self.position_entry_time = 0
                self.total_trades += 1
                order['status'] = 'COMPLETED'
                logger.info(f"🔴 SCALP SELL: {quantity} @ ₹{price:.2f} | PnL: ₹{profit:.2f}")
        
        else:
            # Live trading implementation
            try:
                order_params = {
                    'variety': 'regular',
                    'exchange': 'NSE',
                    'tradingsymbol': self.symbol,
                    'transaction_type': order_type,
                    'quantity': quantity,
                    'order_type': 'MARKET',
                    'product': 'MIS',
                    'validity': 'DAY'
                }
                
                order_id = self.kite.place_order(**order_params)
                order['order_id'] = order_id
                order['status'] = 'PLACED'
                logger.info(f"✅ Live scalp order placed: {order_id}")
                
            except Exception as e:
                order['status'] = f'FAILED: {str(e)}'
                logger.error(f"❌ Live order failed: {e}")
        
        # Send notification
        if order['status'] in ['COMPLETED', 'PLACED']:
            entry_type = "ENTRY" if order_type == "BUY" else "EXIT"
            self.email_manager.send_scalp_order_notification(
                order, self.symbol, self.paper_trading, entry_type
            )
        
        return order

    def record_scalp_trade(self, pnl, duration_bars, exit_reason):
        """Record completed scalp trade"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO scalp_trades 
                (timestamp, symbol, entry_price, exit_price, quantity, pnl, duration_bars, exit_reason, paper_trading)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(), self.symbol,
                self.cost_basis, self.current_price, self.shares_held,
                pnl, duration_bars, exit_reason, self.paper_trading
            ))
            self.db_conn.commit()
            
            # Update daily stats
            self.daily_trades += 1
            self.daily_pnl += pnl
            
            # Send exit notification
            self.email_manager.send_scalp_exit_notification(
                exit_reason, pnl, duration_bars, self.symbol, self.paper_trading
            )
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")

    def check_exit_conditions(self):
        """Check if position should be exited (matching training environment)"""
        if self.shares_held == 0 or self.position_entry_time == 0:
            return False, None
        
        current_pnl_pct = ((self.current_price - self.cost_basis) / self.cost_basis) * 100
        holding_time_bars = self.current_bar - self.position_entry_time
        
        # Convert percentage thresholds
        scalp_target = SCALP_TARGET_PROFIT
        scalp_stop = SCALP_STOP_LOSS
        
        # Target profit reached
        if current_pnl_pct >= scalp_target:
            return True, "TARGET_PROFIT"
        
        # Stop loss hit
        if current_pnl_pct <= -scalp_stop:
            return True, "STOP_LOSS"
        
        # Maximum time exceeded
        if holding_time_bars >= MAX_POSITION_TIME_BARS:
            return True, "TIME_LIMIT"
        
        return False, None

    def should_make_prediction(self):
        """Check if should make new prediction"""
        if self.last_prediction_time is None:
            return True
        
        time_diff = (datetime.now(TIMEZONE) - self.last_prediction_time).total_seconds()
        return time_diff >= PREDICTION_INTERVAL

    def make_scalping_decision(self):
        """Make scalping trading decision"""
        try:
            # Check exit conditions first (matching training environment)
            should_exit, exit_reason = self.check_exit_conditions()
            if should_exit and self.shares_held > 0:
                self.place_scalp_order("SELL", self.shares_held)
                return
            
            # Check if should make new prediction
            if not self.should_make_prediction():
                return
            
            # Don't enter new position if already holding
            if self.shares_held > 0:
                return
            
            # Prepare observation (65 features)
            obs = self.prepare_scalping_observation()
            if obs is None:
                return
            
            # Get model prediction
            action, _states = self.model.predict(obs, deterministic=True)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            predicted_action = action_names[int(action)]
            
            logger.info(f"🤖 Scalp Decision: {predicted_action} @ ₹{self.current_price:.2f}")
            
            # Execute only BUY orders for new positions (matching training environment)
            if predicted_action == "BUY":
                position_size = self.calculate_position_size()
                if position_size > 0:
                    self.place_scalp_order("BUY", position_size)
            
            self.last_prediction_time = datetime.now(TIMEZONE)
            
        except Exception as e:
            logger.error(f"Error in scalping decision: {e}")

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now(TIMEZONE)
        if now.weekday() >= 5:  # Weekend
            return False
        
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close

    def close_all_positions_at_market_close(self):
        """Close all positions at market close"""
        if self.shares_held > 0:
            self.place_scalp_order("SELL", self.shares_held)
            logger.info("🔚 Closed all positions at market close")

    def send_daily_summary(self):
        """Send daily scalping summary"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT COUNT(*), AVG(pnl), AVG(duration_bars),
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)
                FROM scalp_trades 
                WHERE DATE(timestamp) = DATE('now', 'localtime')
                AND paper_trading = ?
            ''', (self.paper_trading,))
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                trades_count, avg_pnl, avg_duration, win_rate = result
                
                self.email_manager.send_scalping_summary(
                    self.symbol, trades_count or 0, self.daily_pnl,
                    win_rate or 0, avg_duration or 0, self.paper_trading
                )
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")

    def run_scalping_strategy(self):
        """Main scalping loop"""
        logger.info("🔥 Starting 5-minute scalping strategy")
        
        # Start WebSocket
        if self.ticker:
            self.ticker.connect(threaded=True)
            time.sleep(2)
        
        last_summary_time = datetime.now(TIMEZONE)
        
        try:
            while True:
                now = datetime.now(TIMEZONE)
                
                # Check market hours
                if not self.is_market_open():
                    # Close positions at market close
                    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
                    after_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
                    
                    if market_close <= now <= after_close:
                        self.close_all_positions_at_market_close()
                        self.send_daily_summary()
                    
                    if now.hour < 9 or now.hour > 16:
                        logger.info("💤 Market closed - sleeping")
                        time.sleep(1800)  # 30 minutes
                        continue
                
                # Create 5-minute candles
                self.create_5min_candles()
                
                # Make scalping decisions
                self.make_scalping_decision()
                
                # Send hourly summary
                if (now - last_summary_time).total_seconds() >= 3600:
                    logger.info(f"📊 Hourly Update - Trades: {self.daily_trades}, PnL: ₹{self.daily_pnl:.2f}")
                    last_summary_time = now
                
                # High frequency checking for scalping
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Scalping bot stopped")
        except Exception as e:
            logger.error(f"❌ Error in scalping strategy: {e}")
        finally:
            if self.ticker:
                self.ticker.close()
            self.db_conn.close()
            logger.info("✅ Scalping session ended")

def start_scalping(symbol, model_path, paper_trading=True):
    """Start scalping bot"""
    full_model_path = f"{MODEL_SAVE_PATH}/{symbol}_scalping.zip"  # Updated path to match training
    if not os.path.exists(full_model_path):
        logger.error(f"❌ Model not found: {full_model_path}")
        return
    
    try:
        model = PPO.load(full_model_path)
        logger.info(f"✅ Scalping model loaded: {full_model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    try:
        bot = ScalpingBot(model, symbol, paper_trading)
        bot.run_scalping_strategy()
    except Exception as e:
        logger.error(f"❌ Scalping bot failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='5-Minute Scalping Bot')
    parser.add_argument('--symbol', type=str, default="ADANIPORTS", help='Stock symbol')
    parser.add_argument('--live', action='store_true', help='Enable live trading')
    
    args = parser.parse_args()
    
    start_scalping(
        symbol=args.symbol,
        model_path=args.symbol,
        paper_trading=not args.live
    )


# Paper scalping
# python scalping_bot.py --symbol ADANIPORTS

# # Live scalping  
# python scalping_bot.py --symbol ADANIPORTS --live
