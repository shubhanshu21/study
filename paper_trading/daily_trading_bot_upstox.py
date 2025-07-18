import numpy as np
import pandas as pd
import os
import logging
import pytz
import sqlite3
from stable_baselines3 import PPO
from datetime import datetime, timedelta
from upstox_client import ApiClient, Configuration, OrderApi, MarketDataApi
from upstox_client.models import PlaceOrderRequest
import time
import json
from finta import TA
import argparse
from dotenv import load_dotenv
import requests
import urllib.parse

# Selenium imports for token management
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
import pyotp

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
        logging.FileHandler("upstox_trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Daily_Trading_Bot")

# Environment variables
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

# Upstox credentials for auto-login
UPSTOX_MOBILE = os.getenv("UPSTOX_MOBILE")
UPSTOX_PASSWORD = os.getenv("UPSTOX_PASSWORD")
UPSTOX_PIN = os.getenv("UPSTOX_PIN")
UPSTOX_TOTP_SECRET = os.getenv("UPSTOX_TOTP_SECRET")

# Upstox Sandbox URLs
SANDBOX_BASE_URL = os.getenv("SANDBOX_BASE_URL")
PROD_BASE_URL = os.getenv("PROD_BASE_URL")

# Trading mode - True for sandbox, False for live
SANDBOX_MODE = True

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST")
EMAIL_PORT = int(os.getenv("EMAIL_PORT"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

# Trading parameters
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE"))
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY"))
BROKERAGE_DELIVERY = float(os.getenv("BROKERAGE_DELIVERY"))
STT_INTRADAY = float(os.getenv("STT_INTRADAY"))
STT_DELIVERY = float(os.getenv("STT_DELIVERY"))
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE"))
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES"))
STAMP_DUTY = float(os.getenv("STAMP_DUTY"))
GST = float(os.getenv("GST"))

# Constants
TIMEZONE = pytz.timezone('Asia/Kolkata')
DB_PATH = "upstox_trading_bot.db"

class EmailNotificationManager:
    """Handles all email notifications"""
    
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
            logger.warning("Email not configured. Skipping notification.")
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
    
    def send_login_success(self, symbol, sandbox_mode):
        """Send login success notification"""
        subject = f"🟢 RL Trading Bot - Login Successful ({symbol})"
        body = f"""
        <h2>🤖 RL Trading Bot - Login Successful</h2>
        <p><strong>Symbol:</strong> {symbol}</p>
        <p><strong>Trading Mode:</strong> {'Sandbox Mode' if sandbox_mode else 'Live Trading'}</p>
        <p><strong>Login Time:</strong> {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</p>
        <p><strong>Status:</strong> Bot is now active and monitoring markets</p>
        
        <hr>
        <p>The bot will perform daily analysis after market close and place orders accordingly.</p>
        <p>You will receive notifications for all trading activities.</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_order_notification(self, order, symbol, sandbox_mode):
        """Send order placed notification"""
        order_type = order['type']
        emoji = "🟢" if order_type == "BUY" else "🔴"
        
        subject = f"{emoji} {order_type} Order - {symbol} ({'Sandbox' if sandbox_mode else 'Live'})"
        
        body = f"""
        <h2>{emoji} {order_type} Order Notification</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Action</strong></td><td>{order_type}</td></tr>
            <tr><td><strong>Quantity</strong></td><td>{order['quantity']}</td></tr>
            <tr><td><strong>Price</strong></td><td>₹{order['price']:.2f}</td></tr>
            <tr><td><strong>Total Value</strong></td><td>₹{order['quantity'] * order['price']:.2f}</td></tr>
            <tr><td><strong>Transaction Cost</strong></td><td>₹{order['transaction_cost']:.2f}</td></tr>
            <tr><td><strong>Status</strong></td><td>{order['status']}</td></tr>
            <tr><td><strong>Order ID</strong></td><td>{order.get('order_id', 'N/A')}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Sandbox Trading' if sandbox_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</td></tr>
        </table>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_pnl_report(self, symbol, balance, equity, position_value, total_pnl, shares_held, cost_basis, current_price, sandbox_mode):
        """Send daily PnL report"""
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        subject = f"{pnl_emoji} Daily PnL Report - {symbol} ({'Sandbox' if sandbox_mode else 'Live'})"
        
        pnl_percentage = (total_pnl / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0
        
        body = f"""
        <h2>{pnl_emoji} Daily PnL Report</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Sandbox Trading' if sandbox_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Date</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d')}</td></tr>
            <tr><td><strong>Cash Balance</strong></td><td>₹{balance:.2f}</td></tr>
            <tr><td><strong>Shares Held</strong></td><td>{shares_held}</td></tr>
            <tr><td><strong>Cost Basis</strong></td><td>₹{cost_basis:.2f}</td></tr>
            <tr><td><strong>Current Price</strong></td><td>₹{current_price:.2f}</td></tr>
            <tr><td><strong>Position Value</strong></td><td>₹{position_value:.2f}</td></tr>
            <tr><td><strong>Total Equity</strong></td><td>₹{equity:.2f}</td></tr>
            <tr><td><strong>Total PnL</strong></td><td style="color: {'green' if total_pnl >= 0 else 'red'}">₹{total_pnl:.2f} ({pnl_percentage:.2f}%)</td></tr>
            <tr><td><strong>Initial Balance</strong></td><td>₹{INITIAL_BALANCE:.2f}</td></tr>
        </table>
        
        <hr>
        <p><strong>Performance Summary:</strong></p>
        <p>{'📈 Positive performance!' if total_pnl >= 0 else '📉 Negative performance - monitor closely'}</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_error_notification(self, error_type, error_message, symbol, sandbox_mode):
        """Send error notification"""
        subject = f"🚨 RL Trading Bot Error - {error_type} ({symbol})"
        
        body = f"""
        <h2>🚨 RL Trading Bot Error Alert</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Error Type</strong></td><td>{error_type}</td></tr>
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Sandbox Trading' if sandbox_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</td></tr>
            <tr><td><strong>Error Message</strong></td><td>{error_message}</td></tr>
        </table>
        
        <hr>
        <p><strong>Action Required:</strong> Please check the trading bot logs and resolve the issue.</p>
        <p>The bot may have stopped or be running in degraded mode.</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_daily_summary(self, symbol, analysis_result, sandbox_mode):
        """Send daily analysis summary"""
        subject = f"📊 Daily Analysis Summary - {symbol} ({'Sandbox' if sandbox_mode else 'Live'})"
        
        body = f"""
        <h2>📊 Daily Analysis Summary</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Analysis Date</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d')}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Sandbox Trading' if sandbox_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Model Prediction</strong></td><td>{analysis_result['action']}</td></tr>
            <tr><td><strong>Closing Price</strong></td><td>₹{analysis_result['price']:.2f}</td></tr>
            <tr><td><strong>Analysis Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%H:%M:%S IST')}</td></tr>
        </table>
        
        <hr>
        <p>Daily analysis completed successfully. {'Order placed' if analysis_result['action'] != 'HOLD' else 'No action taken (HOLD signal)'}</p>
        """
        self.send_email(subject, body, is_html=True)

class UpstoxTokenManager:
    """Handles automatic token generation and refresh for Upstox with full Selenium automation"""
    
    def __init__(self, email_manager):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.redirect_uri = REDIRECT_URI
        self.mobile = UPSTOX_MOBILE
        self.password = UPSTOX_PASSWORD
        self.pin = UPSTOX_PIN
        self.totp_secret = UPSTOX_TOTP_SECRET
        self.email_manager = email_manager
        
        # Validate all required credentials
        if not all([self.api_key, self.api_secret, self.redirect_uri]):
            raise ValueError("Missing required Upstox API credentials in environment variables")
        
        if not all([self.mobile, self.password, self.pin, self.totp_secret]):
            logger.warning("⚠️ Missing login credentials. Auto-login will require manual intervention.")
            self.auto_login_enabled = False
        else:
            self.auto_login_enabled = True
            logger.info("✅ Auto-login enabled with full credentials")
        
        logger.info("✅ UpstoxTokenManager initialized")
    
    def generate_auth_url(self):
        """Generate authorization URL for Upstox login"""
        auth_url = f"https://api.upstox.com/v2/login/authorization/dialog"
        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri,
            'state': 'state_parameter'
        }
        
        auth_url_with_params = f"{auth_url}?{urllib.parse.urlencode(params)}"
        return auth_url_with_params
    
    def generate_totp(self):
        """Generate TOTP code for 2FA"""
        try:
            totp = pyotp.TOTP(self.totp_secret)
            code = totp.now()
            logger.info(f"🔑 TOTP generated: {code}")
            return code
        except Exception as e:
            logger.error(f"❌ Error generating TOTP: {e}")
            return None
    
    def setup_chrome_driver(self, headless=True):
        """Setup Chrome WebDriver with optimal settings"""
        try:
            chrome_options = Options()
            
            # Basic options
            if headless:
                chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # User agent to avoid detection
            chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            
            # Disable images and CSS for faster loading
            prefs = {
                "profile.managed_default_content_settings.images": 2,
                "profile.managed_default_content_settings.stylesheets": 2,
            }
            chrome_options.add_experimental_option("prefs", prefs)
            
            # Create driver
            driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return driver
            
        except Exception as e:
            logger.error(f"❌ Error setting up Chrome driver: {e}")
            return None
    
    def wait_and_find_element(self, driver, by, value, timeout=30, retry_count=3):
        """Wait for element with retry logic"""
        for attempt in range(retry_count):
            try:
                wait = WebDriverWait(driver, timeout)
                element = wait.until(EC.presence_of_element_located((by, value)))
                return element
            except (TimeoutException, NoSuchElementException) as e:
                if attempt < retry_count - 1:
                    logger.warning(f"⚠️ Element not found (attempt {attempt + 1}), retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Element not found after {retry_count} attempts: {value}")
                    raise e
    
    def wait_and_click_element(self, driver, by, value, timeout=30, retry_count=3):
        """Wait for element and click with retry logic"""
        for attempt in range(retry_count):
            try:
                wait = WebDriverWait(driver, timeout)
                element = wait.until(EC.element_to_be_clickable((by, value)))
                
                # Scroll to element if needed
                driver.execute_script("arguments[0].scrollIntoView();", element)
                time.sleep(1)
                
                # Try normal click first
                try:
                    element.click()
                    return True
                except:
                    # If normal click fails, try JavaScript click
                    driver.execute_script("arguments[0].click();", element)
                    return True
                    
            except (TimeoutException, NoSuchElementException, StaleElementReferenceException) as e:
                if attempt < retry_count - 1:
                    logger.warning(f"⚠️ Click failed (attempt {attempt + 1}), retrying...")
                    time.sleep(2)
                else:
                    logger.error(f"❌ Click failed after {retry_count} attempts: {value}")
                    raise e
    
    def fill_input_field(self, driver, by, value, text, clear_first=True):
        """Fill input field with text"""
        try:
            element = self.wait_and_find_element(driver, by, value)
            
            if clear_first:
                element.clear()
            
            # Type text slowly to avoid detection
            for char in text:
                element.send_keys(char)
                time.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error filling input field: {e}")
            return False
    
    def auto_login_and_get_token(self):
        """Automatically login and get access token using Selenium with full automation"""
        print("\n🚀 Starting fully automated login process for Upstox...")
        driver = None
        
        try:
            # Setup Chrome driver
            driver = self.setup_chrome_driver(headless=True)
            if not driver:
                return None
            
            # Navigate to authorization URL
            auth_url = self.generate_auth_url()
            print(f"🌐 Navigating to: {auth_url}")
            driver.get(auth_url)
            
            # Wait for page to load
            time.sleep(3)
            
            if self.auto_login_enabled:
                # Step 1: Enter mobile number
                print("📱 Entering mobile number...")
                mobile_input = self.wait_and_find_element(driver, By.XPATH, "//input[@type='text']")
                if mobile_input:
                    self.fill_input_field(driver, By.XPATH, "//input[@type='text']", self.mobile)
                    time.sleep(2)
                
                # Click Get OTP button
                print("🔘 Clicking Get OTP button...")
                get_otp_selectors = [
                    "//button[contains(text(), 'Get OTP')]",
                    "//button[@id='getOtp']",
                    "//button[contains(@class, 'otp')]",
                    "//input[@type='submit']"
                ]
                
                for selector in get_otp_selectors:
                    try:
                        if self.wait_and_click_element(driver, By.XPATH, selector, timeout=10):
                            print("✅ Get OTP button clicked")
                            break
                    except:
                        continue
                
                time.sleep(5)
                
                # Step 2: Enter TOTP
                print("🔑 Generating and entering TOTP...")
                totp_code = self.generate_totp()
                if totp_code:
                    otp_selectors = [
                        "//input[@id='otpNum']",
                        "//input[@type='text'][contains(@placeholder, 'OTP')]",
                        "//input[@type='text'][contains(@name, 'otp')]",
                        "//input[@type='password'][contains(@placeholder, 'OTP')]"
                    ]
                    
                    for selector in otp_selectors:
                        try:
                            if self.fill_input_field(driver, By.XPATH, selector, totp_code):
                                print("✅ TOTP entered successfully")
                                break
                        except:
                            continue
                    
                    time.sleep(2)
                    
                    # Click Continue/Verify button
                    print("🔘 Clicking Continue button...")
                    continue_selectors = [
                        "//button[@id='continueBtn']",
                        "//button[contains(text(), 'Continue')]",
                        "//button[contains(text(), 'Verify')]",
                        "//button[@type='submit']"
                    ]
                    
                    for selector in continue_selectors:
                        try:
                            if self.wait_and_click_element(driver, By.XPATH, selector, timeout=10):
                                print("✅ Continue button clicked")
                                break
                        except:
                            continue
                    
                    time.sleep(5)
                
                # Step 3: Enter PIN
                print("🔐 Entering PIN...")
                pin_selectors = [
                    "//input[@id='pinCode']",
                    "//input[@type='password'][contains(@placeholder, 'PIN')]",
                    "//input[@type='password'][contains(@name, 'pin')]",
                    "//input[@type='password']"
                ]
                
                for selector in pin_selectors:
                    try:
                        if self.fill_input_field(driver, By.XPATH, selector, self.pin):
                            print("✅ PIN entered successfully")
                            break
                    except:
                        continue
                
                time.sleep(2)
                
                # Click final Continue button
                print("🔘 Clicking final Continue button...")
                final_continue_selectors = [
                    "//button[@id='pinContinueBtn']",
                    "//button[contains(text(), 'Continue')]",
                    "//button[contains(text(), 'Submit')]",
                    "//button[@type='submit']"
                ]
                
                for selector in final_continue_selectors:
                    try:
                        if self.wait_and_click_element(driver, By.XPATH, selector, timeout=10):
                            print("✅ Final Continue button clicked")
                            break
                    except:
                        continue
                
                time.sleep(5)
                
            else:
                # Manual login fallback
                print("🔐 Please complete the login process manually...")
                print("The bot will wait for the redirect and extract the authorization code")
            
            # Wait for redirect with authorization code
            print("⏳ Waiting for redirect with authorization code...")
            wait = WebDriverWait(driver, 300)  # 5 minutes timeout
            
            def check_redirect(driver):
                current_url = driver.current_url
                return 'code=' in current_url or self.redirect_uri in current_url
            
            wait.until(check_redirect)
            
            # Extract authorization code
            current_url = driver.current_url
            print(f"🔍 Current URL: {current_url}")
            
            if 'code=' in current_url:
                # Extract code from URL
                parsed_url = urllib.parse.urlparse(current_url)
                query_params = urllib.parse.parse_qs(parsed_url.query)
                
                if 'code' in query_params:
                    auth_code = query_params['code'][0]
                    print(f"✅ Authorization code extracted: {auth_code[:20]}...")
                    
                    # Get access token
                    access_token = self.get_access_token(auth_code)
                    return access_token
                else:
                    print("❌ No authorization code found in URL parameters")
                    return None
            else:
                print("❌ No authorization code found in redirect URL")
                return None
                
        except Exception as e:
            print(f"❌ Auto-login failed: {e}")
            logger.error(f"Auto-login failed: {e}")
            self.email_manager.send_error_notification("Auto-Login", str(e), "UPSTOX", True)
            return None
            
        finally:
            if driver:
                driver.quit()
                print("🔒 Browser closed")
    
    def get_access_token(self, auth_code):
        """Get access token using authorization code"""
        try:
            url = "https://api.upstox.com/v2/login/authorization/token"
            
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Accept': 'application/json'
            }
            
            data = {
                'code': auth_code,
                'client_id': self.api_key,
                'client_secret': self.api_secret,
                'redirect_uri': self.redirect_uri,
                'grant_type': 'authorization_code'
            }
            
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                token_data = response.json()
                access_token = token_data['access_token']
                
                # Save token with timestamp
                token_info = {
                    'access_token': access_token,
                    'created_at': datetime.now(TIMEZONE).isoformat(),
                    'expires_at': (datetime.now(TIMEZONE) + timedelta(hours=24)).isoformat()
                }
                
                with open("upstox_access_token.json", "w") as f:
                    json.dump(token_info, f, indent=2)
                
                logger.info("✅ Access token generated and saved")
                return access_token
            else:
                logger.error(f"❌ Failed to get access token: {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting access token: {e}")
            return None
    
    def check_existing_token(self):
        """Check if existing token is valid"""
        try:
            if os.path.exists("upstox_access_token.json"):
                with open("upstox_access_token.json", "r") as f:
                    token_info = json.load(f)
                
                access_token = token_info.get('access_token')
                expires_at = token_info.get('expires_at')
                
                if not access_token or not expires_at:
                    return None
                
                # Check if token is expired
                expires_datetime = datetime.fromisoformat(expires_at)
                if datetime.now(TIMEZONE) >= expires_datetime:
                    print("❌ Existing token is expired")
                    return None
                
                # Test token validity with API call
                headers = {
                    'Authorization': f'Bearer {access_token}',
                    'Accept': 'application/json'
                }
                
                base_url = SANDBOX_BASE_URL if SANDBOX_MODE else PROD_BASE_URL
                response = requests.get(f"{base_url}/user/profile", headers=headers)
                
                if response.status_code == 200:
                    print("✅ Existing token is valid")
                    return access_token
                else:
                    print("❌ Existing token is invalid")
                    return None
            
            return None
            
        except Exception as e:
            print(f"❌ Error checking existing token: {e}")
            return None
    
    def refresh_token_if_needed(self):
        """Check if token needs refresh and do it automatically"""
        existing_token = self.check_existing_token()
        if existing_token:
            return existing_token
        
        print("🆕 Getting new token...")
        return self.auto_login_and_get_token()

# [Rest of the code remains the same - DailyTradingAgent class and other functions]
class DailyTradingAgent:
    def __init__(self, model, symbol, sandbox_mode=True):
        self.model = model
        self.symbol = symbol
        self.sandbox_mode = sandbox_mode
        self.api_client = None
        self.access_token = None
        self.df = pd.DataFrame()
        self.current_step = 0
        self.daily_analysis_completed = False
        self.last_trading_date = None
        
        # Portfolio state
        self.initial_balance = INITIAL_BALANCE
        self.balance = INITIAL_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.position = {'quantity': 0, 'avg_price': 0, 'last_price': 0, 'pnl': 0}
        self.order_history = []
        
        # Initialize managers
        self.email_manager = EmailNotificationManager()
        self.token_manager = UpstoxTokenManager(self.email_manager)
        
        # Database connection
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()
        
        # Setup Upstox connection
        self.setup_upstox()
        
        # Load balance from database
        self.load_balance_from_db()
        
        logger.info(f"Daily Trading Agent initialized for {symbol}")
        logger.info(f"Sandbox Mode: {sandbox_mode}")
        logger.info(f"Initial Balance: ₹{self.initial_balance:.2f}")
        logger.info(f"Current Balance: ₹{self.balance:.2f}")
    
    # [Rest of the methods remain the same as in your original code]
    def create_tables(self):
        """Create database tables for storing trading data"""
        cursor = self.db_conn.cursor()
        
        # Orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                order_type TEXT,
                quantity INTEGER,
                price REAL,
                status TEXT,
                transaction_cost REAL,
                order_id TEXT,
                sandbox_mode BOOLEAN,
                order_variety TEXT DEFAULT 'regular'
            )''')
        
        # Positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                quantity INTEGER,
                avg_price REAL,
                last_price REAL,
                pnl REAL
            )''')
        
        # Account snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                balance REAL,
                equity REAL,
                position_value REAL,
                total_pnl REAL
            )''')
        
        # Daily analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                action TEXT,
                confidence REAL,
                closing_price REAL,
                indicators TEXT
            )''')
        
        # Balance tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS balance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                cash_balance REAL,
                shares_held INTEGER,
                cost_basis REAL,
                last_synced DATETIME,
                sandbox_mode BOOLEAN,
                UNIQUE(symbol, sandbox_mode)
            )''')
        
        self.db_conn.commit()
        logger.info("Database tables created/verified")
    
    def setup_upstox(self):
        """Setup Upstox API client"""
        try:
            print("🔧 Setting up Upstox connection...")
            
            # Get valid access token
            access_token = self.token_manager.refresh_token_if_needed()
            
            if not access_token:
                error_msg = "Failed to get valid access token"
                logger.error(error_msg)
                self.email_manager.send_error_notification("Upstox Setup", error_msg, self.symbol, self.sandbox_mode)
                return
            
            # Setup API client
            configuration = Configuration()
            configuration.access_token = access_token
            
            if self.sandbox_mode:
                configuration.host = SANDBOX_BASE_URL
            else:
                configuration.host = PROD_BASE_URL
            
            self.api_client = ApiClient(configuration)
            self.access_token = access_token
            
            logger.info("✅ Upstox setup complete")
            print("✅ Upstox setup complete")
            
            # Send login success notification
            self.email_manager.send_login_success(self.symbol, self.sandbox_mode)
            
        except Exception as e:
            logger.error(f"Upstox setup failed: {e}")
            print(f"❌ Upstox setup failed: {e}")
            self.email_manager.send_error_notification("Upstox Setup", str(e), self.symbol, self.sandbox_mode)
    
    def get_current_price(self):
        """Get current market price from Upstox"""
        try:
            if not self.api_client:
                return None
            
            market_data_api = MarketDataApi(self.api_client)
            
            # Get instrument key (NSE_EQ for equity)
            instrument_key = f"NSE_EQ|{self.symbol}"
            
            # Get LTP (Last Traded Price)
            response = market_data_api.get_market_data_quotes([instrument_key])
            
            if response.status == 'success' and response.data:
                ltp = response.data[instrument_key]['last_price']
                return float(ltp)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return None
    
    def fetch_historical_data(self, interval='1day', days=365):
        """Fetch historical data from Upstox"""
        try:
            if not self.api_client:
                logger.error("API client not available")
                return pd.DataFrame()
            
            market_data_api = MarketDataApi(self.api_client)
            
            # Get instrument key
            instrument_key = f"NSE_EQ|{self.symbol}"
            
            # Calculate date range
            to_date = datetime.now(TIMEZONE).date()
            from_date = to_date - timedelta(days=days)
            
            # Get historical data
            response = market_data_api.get_historical_candle_data(
                instrument_key=instrument_key,
                interval=interval,
                to_date=to_date.isoformat(),
                from_date=from_date.isoformat()
            )
            
            if response.status == 'success' and response.data:
                # Convert to DataFrame
                data = response.data['candles']
                df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date').reset_index(drop=True)
                
                logger.info(f"✅ Fetched {len(df)} historical data points")
                return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            return pd.DataFrame()
    
    def calculate_transaction_costs(self, trade_value, is_delivery=True):
        """Calculate transaction costs for a trade"""
        brokerage_rate = BROKERAGE_DELIVERY if is_delivery else BROKERAGE_INTRADAY
        stt_rate = STT_DELIVERY if is_delivery else STT_INTRADAY
        
        brokerage = min(trade_value * brokerage_rate, 20)
        stt = trade_value * stt_rate
        exchange_charge = trade_value * EXCHANGE_TXN_CHARGE
        sebi_charges = trade_value * SEBI_CHARGES
        stamp_duty = trade_value * STAMP_DUTY
        gst = (brokerage + exchange_charge) * GST
        
        total_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst
        return round(total_cost, 2)
    
    def calculate_position_size(self, current_price):
        """Calculate position size based on available balance"""
        if current_price <= 0:
            return 0
        
        # Use 10% of available balance for each trade
        available_amount = self.balance * 0.1
        position_size = int(available_amount / current_price)
        
        return max(position_size, 1) if position_size > 0 else 0
    
    def place_order(self, order_type, quantity, price, order_variety="regular"):
        """Place an order through Upstox"""
        if quantity <= 0 or price <= 0:
            logger.warning(f"Invalid order parameters: quantity={quantity}, price={price}")
            return None
        
        trade_value = quantity * price
        transaction_cost = self.calculate_transaction_costs(trade_value)
        
        order = {
            'symbol': self.symbol,
            'type': order_type,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(TIMEZONE).isoformat(),
            'transaction_cost': transaction_cost,
            'status': 'PENDING',
            'order_id': None,
            'order_variety': order_variety
        }
        
        try:
            if self.api_client:
                order_api = OrderApi(self.api_client)
                
                # Create order request
                order_request = PlaceOrderRequest(
                    quantity=quantity,
                    product='D',  # Delivery
                    validity='DAY',
                    price=price,
                    tag='RL_BOT',
                    instrument_token=f"NSE_EQ|{self.symbol}",
                    order_type='MARKET',
                    transaction_type='BUY' if order_type == 'BUY' else 'SELL',
                    disclosed_quantity=0,
                    trigger_price=0,
                    is_amo=order_variety == 'amo'
                )
                
                # Place order
                response = order_api.place_order(order_request)
                
                if response.status == 'success':
                    order['order_id'] = response.data.get('order_id')
                    order['status'] = 'PLACED'
                    
                    # Update local position for sandbox mode
                    if self.sandbox_mode:
                        self.update_position_after_order(order_type, quantity, price, transaction_cost)
                    
                    logger.info(f"✅ Order placed: {order_type} {quantity} shares at ₹{price:.2f}")
                else:
                    order['status'] = f'FAILED: {response.errors}'
                    logger.error(f"❌ Order failed: {response.errors}")
            
            else:
                # Fallback for sandbox mode without API client
                if self.sandbox_mode:
                    order['status'] = 'COMPLETED'
                    order['order_id'] = f"SANDBOX_{int(time.time())}"
                    self.update_position_after_order(order_type, quantity, price, transaction_cost)
                    logger.info(f"✅ Sandbox order executed: {order_type} {quantity} shares at ₹{price:.2f}")
                else:
                    order['status'] = 'FAILED: No API client'
                    logger.error("❌ No API client available for live trading")
        
        except Exception as e:
            order['status'] = f'FAILED: {str(e)}'
            logger.error(f"❌ Order failed: {e}")
            self.email_manager.send_error_notification("Order Placement", str(e), self.symbol, self.sandbox_mode)
        
        # Update position tracking
        self.position['quantity'] = self.shares_held
        self.position['avg_price'] = self.cost_basis
        self.position['last_price'] = price
        self.position['pnl'] = self.shares_held * (price - self.cost_basis) if self.cost_basis > 0 else 0
        
        # Save order and send notification
        self.save_order(order)
        self.order_history.append(order)
        
        if order['status'] not in ['PENDING', 'FAILED']:
            self.email_manager.send_order_notification(order, self.symbol, self.sandbox_mode)
        
        return order
    
    def update_position_after_order(self, order_type, quantity, price, transaction_cost):
        """Update position after successful order execution"""
        trade_value = quantity * price
        
        if order_type == "BUY":
            total_cost = trade_value + transaction_cost
            if self.balance >= total_cost:
                self.balance -= total_cost
                
                if self.shares_held >= 0:
                    total_value = self.shares_held * self.cost_basis + trade_value
                    self.shares_held += quantity
                    self.cost_basis = total_value / self.shares_held if self.shares_held > 0 else 0
                else:
                    self.shares_held += quantity
                
                self.save_balance_to_db()
        
        elif order_type == "SELL":
            if self.shares_held >= quantity:
                self.balance += (trade_value - transaction_cost)
                self.shares_held -= quantity
                
                if self.shares_held == 0:
                    self.cost_basis = 0
                
                self.save_balance_to_db()
    
    def save_order(self, order):
        """Save order to database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO orders (timestamp, symbol, order_type, quantity, price, 
                                  status, transaction_cost, order_id, sandbox_mode, order_variety)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order['timestamp'], order['symbol'], order['type'], order['quantity'],
                order['price'], order['status'], order['transaction_cost'],
                order.get('order_id', ''), self.sandbox_mode, order.get('order_variety', 'regular')
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving order: {e}")
    
    def save_balance_to_db(self):
        """Save balance to database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO balance_tracking (
                    timestamp, symbol, cash_balance, shares_held, 
                    cost_basis, last_synced, sandbox_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(),
                self.symbol,
                self.balance,
                self.shares_held,
                self.cost_basis,
                datetime.now(TIMEZONE).isoformat(),
                self.sandbox_mode
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving balance to DB: {e}")
    
    def load_balance_from_db(self):
        """Load balance from database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                SELECT cash_balance, shares_held, cost_basis, last_synced 
                FROM balance_tracking 
                WHERE symbol = ? AND sandbox_mode = ?
                ORDER BY timestamp DESC LIMIT 1
            ''', (self.symbol, self.sandbox_mode))
            
            result = cursor.fetchone()
            if result:
                self.balance, self.shares_held, self.cost_basis, last_synced = result
                logger.info(f"📁 Balance loaded from DB: ₹{self.balance:.2f}")
                return True
        except Exception as e:
            logger.error(f"Error loading balance from DB: {e}")
        return False
    
    def save_daily_snapshot(self):
        """Save daily account snapshot and send PnL report"""
        try:
            position_value = self.shares_held * self.position['last_price']
            equity = self.balance + position_value
            total_pnl = equity - self.initial_balance
            
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO account_snapshots (timestamp, balance, equity, position_value, total_pnl)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(), self.balance, equity, position_value, total_pnl
            ))
            self.db_conn.commit()
            
            # Save balance
            self.save_balance_to_db()
            
            logger.info(f"📊 Daily Snapshot - Balance: ₹{self.balance:.2f}, Equity: ₹{equity:.2f}, PnL: ₹{total_pnl:.2f}")
            
            # Send PnL report
            self.email_manager.send_pnl_report(
                self.symbol, self.balance, equity, position_value, total_pnl,
                self.shares_held, self.cost_basis, self.position['last_price'], self.sandbox_mode
            )
            
        except Exception as e:
            logger.error(f"Error saving daily snapshot: {e}")
    
    def prepare_observation(self):
        """Prepare observation vector for the RL model"""
        if self.df.empty or self.current_step >= len(self.df):
            logger.error("No data available for observation")
            return None
        
        try:
            row = self.df.iloc[self.current_step]
            current_price = row["Close"]
            
            # Get window data for calculations
            window_start = max(0, self.current_step - 20 + 1)
            window_data = self.df.iloc[window_start:self.current_step + 1]
            close_mean = window_data["Close"].mean()
            
            # Portfolio state
            obs = [
                self.balance / self.initial_balance,
                self.shares_held / 100,
                self.cost_basis / current_price if self.cost_basis > 0 else 0,
                current_price / close_mean if close_mean > 0 else 1,
                1 if self.shares_held > 0 else 0,
                1 if self.shares_held < 0 else 0,
                0,  # Trailing stop (placeholder)
                0,  # Target price (placeholder)
            ]
            
            # OHLCV data (normalized)
            for col in ["Open", "High", "Low", "Close"]:
                obs.append(row[col] / close_mean if close_mean > 0 else 1)
            
            volume_mean = window_data["Volume"].mean()
            obs.append(row["Volume"] / volume_mean if volume_mean > 0 else 1)
            
            # Technical indicators (same as original)
            indicators = [
                'SMA5', 'SMA10', 'SMA20', 'SMA50',
                'EMA5', 'EMA10', 'EMA20', 'EMA50',
                'RSI', 'MACD', 'Signal',
                'BB_Upper', 'BB_Middle', 'BB_Lower',
                'OBV', 'CMF', 'ADX', 'SAR',
                'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A', 'ICHIMOKU_SENKOU_B',
                'STOCH_K', 'STOCH_D', 'CCI', 'WILLIAMS', 'STD20',
                'KC_UPPER', 'KC_MIDDLE', 'KC_LOWER',
                'GOLDEN_CROSS', 'DEATH_CROSS', 'ROC5', 'ROC20', 'VOL_REGIME'
            ]
            
            for indicator in indicators:
                if indicator in row:
                    if indicator == 'RSI':
                        obs.append(row[indicator] / 100)
                    elif indicator in ['GOLDEN_CROSS', 'DEATH_CROSS']:
                        obs.append(row[indicator])
                    elif indicator in ['CCI']:
                        obs.append((min(max(row[indicator], -200), 200) + 200) / 400)
                    elif indicator in ['WILLIAMS']:
                        obs.append((row[indicator] + 100) / 100)
                    elif indicator in ['STOCH_K', 'STOCH_D']:
                        obs.append(row[indicator] / 100)
                    elif indicator in ['CMF']:
                        obs.append((row[indicator] + 1) / 2)
                    elif indicator in ['ADX']:
                        obs.append(row[indicator] / 100)
                    elif indicator in ['VOL_REGIME']:
                        obs.append(min(row[indicator], 3) / 3)
                    elif indicator in ['ROC5', 'ROC20']:
                        obs.append(min(max((row[indicator] + 20) / 40, 0), 1))
                    else:
                        obs.append(row[indicator] / close_mean if close_mean > 0 else 1)
                else:
                    obs.append(0.5)
            
            # Ensure exactly 53 features
            while len(obs) < 53:
                obs.append(0.5)
            
            obs = obs[:53]
            
            return np.nan_to_num(np.array(obs, dtype=np.float32))
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            return None
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators (same as original)"""
        try:
            logger.info("📊 Calculating technical indicators...")
            
            # Basic Moving Averages
            for period in [5, 10, 20, 50]:
                df[f'SMA{period}'] = TA.SMA(df, period=period)
                df[f'EMA{period}'] = TA.EMA(df, period=period)
            
            # RSI
            df['RSI'] = TA.RSI(df, period=14)
            
            # MACD
            macd_data = TA.MACD(df)
            df['MACD'] = macd_data['MACD']
            df['Signal'] = macd_data['SIGNAL']
            
            # Bollinger Bands
            bb_data = TA.BBANDS(df)
            df['BB_Upper'] = bb_data['BB_UPPER']
            df['BB_Middle'] = bb_data['BB_MIDDLE']
            df['BB_Lower'] = bb_data['BB_LOWER']
            
            # Volume indicators
            df['OBV'] = TA.OBV(df)
            
            # Chaikin Money Flow
            mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
            mfv = mfm * df['Volume']
            df['CMF'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()
            
            # ADX
            df['ADX'] = TA.ADX(df)
            
            # Parabolic SAR
            df['SAR'] = TA.SAR(df)
            
            # Ichimoku
            ichimoku = TA.ICHIMOKU(df)
            df['ICHIMOKU_TENKAN'] = ichimoku['TENKAN']
            df['ICHIMOKU_KIJUN'] = ichimoku['KIJUN']
            df['ICHIMOKU_SENKOU_A'] = ichimoku['senkou_span_a']
            df['ICHIMOKU_SENKOU_B'] = ichimoku['SENKOU']
            
            # Stochastic
            df['STOCH_K'] = TA.STOCH(df)
            df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
            
            # CCI
            df['CCI'] = TA.CCI(df)
            
            # Williams %R
            df['WILLIAMS'] = TA.WILLIAMS(df)
            
            # Standard Deviation
            df['STD20'] = df['Close'].rolling(20).std()
            
            # ATR and Keltner Channels
            df['ATR'] = TA.ATR(df)
            df['KC_MIDDLE'] = df['Close'].ewm(span=20).mean()
            df['KC_UPPER'] = df['KC_MIDDLE'] + 2 * df['ATR']
            df['KC_LOWER'] = df['KC_MIDDLE'] - 2 * df['ATR']
            
            # Moving Average Crosses
            df['GOLDEN_CROSS'] = np.where(
                (df['SMA10'].shift(1) < df['SMA50'].shift(1)) & 
                (df['SMA10'] > df['SMA50']), 1, 0
            )
            df['DEATH_CROSS'] = np.where(
                (df['SMA10'].shift(1) > df['SMA50'].shift(1)) & 
                (df['SMA10'] < df['SMA50']), 1, 0
            )
            
            # Rate of Change
            df['ROC5'] = TA.ROC(df, period=5)
            df['ROC20'] = TA.ROC(df, period=20)
            
            # Volatility Regime
            df['VOL_REGIME'] = df['ATR'].rolling(5).mean() / df['ATR'].rolling(20).mean()
            
            # Fill NaN values
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)
            df.fillna(0, inplace=True)
            
            logger.info("✅ Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            return pd.DataFrame()
    
    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now(TIMEZONE)
        
        # Check if it's a trading day (Monday-Friday)
        if now.weekday() >= 5:
            return False
        
        # Market hours: 9:15 AM to 3:30 PM
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return market_open <= now <= market_close
    
    def is_trading_day(self):
        """Check if today is a trading day"""
        now = datetime.now(TIMEZONE)
        return now.weekday() < 5
    
    def has_traded_today(self):
        """Check if any successful trade happened today"""
        today = datetime.now(TIMEZONE).strftime('%Y-%m-%d')
        cursor = self.db_conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM orders 
            WHERE DATE(timestamp) = ? 
            AND status IN ('COMPLETED', 'PLACED')
        ''', (today,))
        return cursor.fetchone()[0] > 0
    
    def get_trading_windows(self):
        """Get trading time windows"""
        now = datetime.now(TIMEZONE)
        current_date = now.date()
        
        # Evening analysis windows  
        evening_windows = [
            (16, 0), (16, 30), (17, 0), (17, 30),
            (18, 0), (19, 0), (20, 0), (21, 0), (22, 0), (23, 0),
        ]
        
        windows = []
        for hour, minute in evening_windows:
            window_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            if window_time.date() == current_date and window_time >= now:
                windows.append(window_time)
        
        # If no windows left for today, add tomorrow's windows
        if not windows:
            tomorrow = now + timedelta(days=1)
            while tomorrow.weekday() >= 5:
                tomorrow += timedelta(days=1)
            
            for hour, minute in evening_windows:
                window_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                windows.append(window_time)
        
        return windows
    
    def daily_analysis_and_trading(self):
        """Perform daily analysis and trading"""
        try:
            logger.info("🔍 Starting daily analysis...")
            
            # Check if already traded today
            if self.has_traded_today():
                logger.info("✅ Already traded today, skipping analysis")
                return True
            
            # Fetch latest historical data
            df = self.fetch_historical_data(interval='1day', days=365)
            if df.empty:
                logger.error("❌ Failed to fetch historical data")
                return False
            
            # Calculate technical indicators
            df = self.calculate_all_indicators(df)
            if df.empty:
                logger.error("❌ Failed to calculate indicators")
                return False
            
            # Update internal state
            self.df = df
            self.current_step = len(df) - 1
            
            # Get current price
            current_price = df.iloc[-1]['Close']
            self.position['last_price'] = current_price
            
            # Generate observation
            obs = self.prepare_observation()
            if obs is None:
                logger.error("❌ Failed to prepare observation")
                return False
            
            # Get model prediction
            action, _states = self.model.predict(obs, deterministic=True)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            predicted_action = action_names[int(action)]
            
            logger.info(f"🤖 Model Prediction: {predicted_action} at ₹{current_price:.2f}")
            
            # Calculate position size
            position_size = self.calculate_position_size(current_price)
            
            # Execute trade based on prediction
            order = None
            if predicted_action == "BUY" and position_size > 0:
                order = self.place_order("BUY", position_size, current_price)
            elif predicted_action == "SELL" and self.shares_held > 0:
                sell_quantity = min(position_size, self.shares_held)
                order = self.place_order("SELL", sell_quantity, current_price)
            
            # Save analysis and snapshot
            self.save_daily_analysis(predicted_action, current_price, obs)
            self.save_daily_snapshot()
            
            # Send notifications
            analysis_result = {'action': predicted_action, 'price': current_price}
            self.email_manager.send_daily_summary(self.symbol, analysis_result, self.sandbox_mode)
            
            logger.info("✅ Daily analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error in daily analysis: {e}")
            self.email_manager.send_error_notification("Daily Analysis", str(e), self.symbol, self.sandbox_mode)
            return False
    
    def save_daily_analysis(self, action, price, indicators):
        """Save daily analysis to database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO daily_analysis (timestamp, symbol, action, confidence, closing_price, indicators)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(TIMEZONE).isoformat(), self.symbol, action, 0.0, price, json.dumps(indicators.tolist())
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving daily analysis: {e}")
    
    def run_daily_strategy(self):
        """Run daily trading strategy"""
        logger.info("🚀 Starting Daily Trading Strategy with Upstox Sandbox")
        
        while True:
            try:
                now = datetime.now(TIMEZONE)
                current_date = now.date()
                
                # Reset daily flag for new trading day
                if self.last_trading_date != current_date and self.is_trading_day():
                    self.daily_analysis_completed = False
                    self.last_trading_date = current_date
                    logger.info(f"📅 New trading day: {current_date}")
                
                # Check if it's a trading day
                if not self.is_trading_day():
                    logger.info("📅 Weekend/Holiday - No trading")
                    time.sleep(3600)
                    continue
                
                # If analysis already completed today, wait for next day
                if self.daily_analysis_completed:
                    next_day = now + timedelta(days=1)
                    while next_day.weekday() >= 5:
                        next_day += timedelta(days=1)
                    
                    next_trading_day = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
                    wait_seconds = (next_trading_day - now).total_seconds()
                    logger.info(f"✅ Daily analysis completed. Next trading day: {next_trading_day.strftime('%Y-%m-%d')}")
                    time.sleep(min(wait_seconds, 3600))
                    continue
                
                # Get trading windows
                trading_windows = self.get_trading_windows()
                
                # Check if we're in any trading window
                for window_time in trading_windows:
                    time_diff = (window_time - now).total_seconds()
                    
                    # If we're within 15 minutes of a trading window
                    if -900 <= time_diff <= 900:
                        logger.info(f"🎯 In trading window: {window_time.strftime('%H:%M')}")
                        
                        try:
                            success = self.daily_analysis_and_trading()
                            
                            if success:
                                self.daily_analysis_completed = True
                                logger.info("✅ Daily analysis completed successfully")
                                break
                            else:
                                logger.warning(f"⚠️ Analysis failed at {window_time.strftime('%H:%M')}")
                        except Exception as e:
                            logger.error(f"❌ Error during analysis: {e}")
                            continue
                    
                    # Wait for next window
                    elif time_diff > 0:
                        wait_minutes = time_diff / 60
                        logger.info(f"⏰ Next window at {window_time.strftime('%H:%M')} (in {wait_minutes:.1f} minutes)")
                        time.sleep(min(time_diff, 600))
                        break
                
                # If no windows matched, sleep and check again
                if not any(-900 <= (w - now).total_seconds() <= 900 for w in trading_windows):
                    time.sleep(300)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Critical error: {e}")
                self.email_manager.send_error_notification("Critical Error", str(e), self.symbol, self.sandbox_mode)
                time.sleep(300)
        
        self.db_conn.close()
        logger.info("✅ Daily trading strategy ended")

def start_daily_trading(symbol, model_path, sandbox_mode=True):
    """Start the daily trading strategy"""
    
    # Load the trained model
    full_model_path = f"{MODEL_SAVE_PATH}/{symbol}.zip"
    if not os.path.exists(full_model_path):
        logger.error(f"❌ Model not found: {full_model_path}")
        return
    
    try:
        model = PPO.load(full_model_path)
        logger.info(f"✅ Model loaded from {full_model_path}")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        return
    
    # Create and run trading agent
    try:
        agent = DailyTradingAgent(model, symbol, sandbox_mode)
        agent.run_daily_strategy()
        
    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
        
        # Save trading history
        if 'agent' in locals():
            history_file = f"{symbol}_trading_history.json"
            with open(history_file, "w") as f:
                json.dump(agent.order_history, f, indent=4, default=str)
            logger.info(f"📁 Trading history saved to {history_file}")
            
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        
    finally:
        logger.info("✅ Trading session ended")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Daily RL Trading Bot with Upstox Auto-Login')
    parser.add_argument('--symbol', type=str, default="ADANIPORTS", help='Stock symbol to trade')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: sandbox mode)')
    
    args = parser.parse_args()
    
    # Start daily trading
    start_daily_trading(
        symbol=args.symbol,
        model_path=args.symbol,
        sandbox_mode=not args.live
    )

#  python daily_trading_bot_upstox.py --symbol ADANIPORTS 
 
# python daily_trading_bot_upstox.py --symbol ADANIPORTS --live