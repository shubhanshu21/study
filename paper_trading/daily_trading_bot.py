import numpy as np
import pandas as pd
import os
import logging
import pytz
import sqlite3
from stable_baselines3 import PPO
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
import time
from kiteconnect import exceptions
import json
from finta import TA
import argparse
from dotenv import load_dotenv

# Selenium imports for token management
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
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
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Daily_Trading_Bot")

# Environment variables
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models")

# Email configuration
EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

# Trading parameters
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "100000"))
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY", "0.0003"))
BROKERAGE_DELIVERY = float(os.getenv("BROKERAGE_DELIVERY", "0.0"))
STT_INTRADAY = float(os.getenv("STT_INTRADAY", "0.00025"))
STT_DELIVERY = float(os.getenv("STT_DELIVERY", "0.001"))
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE", "0.0000345"))
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES", "0.000001"))
STAMP_DUTY = float(os.getenv("STAMP_DUTY", "0.00015"))
GST = float(os.getenv("GST", "0.18"))

# Constants
TIMEZONE = pytz.timezone('Asia/Kolkata')
DB_PATH = "trading_bot.db"

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
    
    def send_login_success(self, symbol, trading_mode):
        """Send login success notification"""
        subject = f"🟢 RL Trading Bot - Login Successful ({symbol})"
        body = f"""
        <h2>🤖 RL Trading Bot - Login Successful</h2>
        <p><strong>Symbol:</strong> {symbol}</p>
        <p><strong>Trading Mode:</strong> {'Live Trading' if not trading_mode else 'Paper Trading'}</p>
        <p><strong>Login Time:</strong> {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</p>
        <p><strong>Status:</strong> Bot is now active and monitoring markets</p>
        
        <hr>
        <p>The bot will perform daily analysis after market close and place orders accordingly.</p>
        <p>You will receive notifications for all trading activities.</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_order_notification(self, order, symbol, trading_mode):
        """Send order placed notification"""
        order_type = order['type']
        emoji = "🟢" if order_type == "BUY" else "🔴"
        
        subject = f"{emoji} {order_type} Order - {symbol} ({'Paper' if trading_mode else 'Live'})"
        
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
            <tr><td><strong>Trading Mode</strong></td><td>{'Paper Trading' if trading_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</td></tr>
        </table>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_pnl_report(self, symbol, balance, equity, position_value, total_pnl, shares_held, cost_basis, current_price, trading_mode):
        """Send daily PnL report"""
        pnl_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        subject = f"{pnl_emoji} Daily PnL Report - {symbol} ({'Paper' if trading_mode else 'Live'})"
        
        pnl_percentage = (total_pnl / INITIAL_BALANCE) * 100 if INITIAL_BALANCE > 0 else 0
        
        body = f"""
        <h2>{pnl_emoji} Daily PnL Report</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Paper Trading' if trading_mode else 'Live Trading'}</td></tr>
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
    
    def send_error_notification(self, error_type, error_message, symbol, trading_mode):
        """Send error notification"""
        subject = f"🚨 RL Trading Bot Error - {error_type} ({symbol})"
        
        body = f"""
        <h2>🚨 RL Trading Bot Error Alert</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Error Type</strong></td><td>{error_type}</td></tr>
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Paper Trading' if trading_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S IST')}</td></tr>
            <tr><td><strong>Error Message</strong></td><td>{error_message}</td></tr>
        </table>
        
        <hr>
        <p><strong>Action Required:</strong> Please check the trading bot logs and resolve the issue.</p>
        <p>The bot may have stopped or be running in degraded mode.</p>
        """
        self.send_email(subject, body, is_html=True)
    
    def send_daily_summary(self, symbol, analysis_result, trading_mode):
        """Send daily analysis summary"""
        subject = f"📊 Daily Analysis Summary - {symbol} ({'Paper' if trading_mode else 'Live'})"
        
        body = f"""
        <h2>📊 Daily Analysis Summary</h2>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr><td><strong>Symbol</strong></td><td>{symbol}</td></tr>
            <tr><td><strong>Analysis Date</strong></td><td>{datetime.now(TIMEZONE).strftime('%Y-%m-%d')}</td></tr>
            <tr><td><strong>Trading Mode</strong></td><td>{'Paper Trading' if trading_mode else 'Live Trading'}</td></tr>
            <tr><td><strong>Model Prediction</strong></td><td>{analysis_result['action']}</td></tr>
            <tr><td><strong>Closing Price</strong></td><td>₹{analysis_result['price']:.2f}</td></tr>
            <tr><td><strong>Analysis Time</strong></td><td>{datetime.now(TIMEZONE).strftime('%H:%M:%S IST')}</td></tr>
        </table>
        
        <hr>
        <p>Daily analysis completed successfully. {'Order placed' if analysis_result['action'] != 'HOLD' else 'No action taken (HOLD signal)'}</p>
        """
        self.send_email(subject, body, is_html=True)

class ZerodhaTokenManager:
    """Handles automatic token generation and refresh"""
    
    def __init__(self, email_manager):
        self.api_key = API_KEY
        self.api_secret = API_SECRET
        self.username = os.getenv("ZERODHA_USER_ID")
        self.password = os.getenv("ZERODHA_PASSWORD")
        self.totp_secret = os.getenv("ZERODHA_TOTP_SECRET")
        self.email_manager = email_manager
        
        if not all([self.api_key, self.api_secret, self.username, self.password, self.totp_secret]):
            raise ValueError("Missing required Zerodha credentials in environment variables")
        
        logger.info("✅ ZerodhaTokenManager initialized")
    
    def generate_totp(self):
        """Generate TOTP code for 2FA"""
        try:
            print("🔐 Generating TOTP code...")
            totp = pyotp.TOTP(self.totp_secret)
            code = totp.now()
            print(f"✅ TOTP generated: {code}")
            logger.info(f"TOTP generated: {code}")
            return code
        except Exception as e:
            print(f"❌ TOTP generation failed: {e}")
            logger.error(f"TOTP generation failed: {e}")
            raise
    
    def setup_chrome_driver(self):
        """Setup Chrome driver with Chromium"""
        print("🌐 Setting up Chrome driver with Chromium...")
        
        chrome_options = Options()
        chrome_options.binary_location = "/usr/bin/chromium-browser"
        
        # Server-optimized arguments
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--remote-debugging-port=9222")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-plugins")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        try:
            driver = webdriver.Chrome(options=chrome_options)
            print("✅ Chrome driver (Chromium) created successfully")
            logger.info("Chrome driver setup successful")
            return driver
        except Exception as e:
            print(f"❌ Chrome driver setup failed: {e}")
            logger.error(f"Chrome driver setup failed: {e}")
            self.email_manager.send_error_notification("Chrome Driver Setup", str(e), "SYSTEM", True)
            raise
    
    def find_totp_field(self, driver):
        """Find the TOTP field based on the actual HTML structure"""
        print("🔍 Looking for TOTP field...")
        
        # Wait for the 2FA form to load
        time.sleep(5)
        
        # Strategy 1: Look for form with class "twofa-form"
        try:
            twofa_form = driver.find_element(By.CLASS_NAME, "twofa-form")
            print("✅ Found 2FA form")
            
            # Look for input field within the 2FA form
            totp_input = twofa_form.find_element(By.TAG_NAME, "input")
            if totp_input.is_displayed():
                print("✅ Found TOTP input field in 2FA form")
                return totp_input
                
        except NoSuchElementException:
            print("❌ 2FA form not found")
        
        # Strategy 2: Look for input with type="number" and specific attributes
        try:
            totp_input = driver.find_element(By.XPATH, "//input[@type='number' and @maxlength='6' and @minlength='6']")
            if totp_input.is_displayed():
                print("✅ Found TOTP input by number type and length")
                return totp_input
        except NoSuchElementException:
            print("❌ TOTP input not found by number type")
        
        # Strategy 3: Look for input with placeholder containing dots
        try:
            totp_input = driver.find_element(By.XPATH, "//input[@placeholder='••••••']")
            if totp_input.is_displayed():
                print("✅ Found TOTP input by placeholder")
                return totp_input
        except NoSuchElementException:
            print("❌ TOTP input not found by placeholder")
        
        # Strategy 4: Look for any input field in a div with "twofa-value" class
        try:
            twofa_div = driver.find_element(By.CLASS_NAME, "twofa-value")
            totp_input = twofa_div.find_element(By.TAG_NAME, "input")
            if totp_input.is_displayed():
                print("✅ Found TOTP input in twofa-value div")
                return totp_input
        except NoSuchElementException:
            print("❌ TOTP input not found in twofa-value div")
        
        # Strategy 5: Look for any visible input field after login (since userid is reused)
        try:
            # Check if we're on a different page (not login page)
            if "External TOTP" in driver.page_source:
                print("✅ Detected TOTP page by text content")
                
                # Find any visible input field
                input_fields = driver.find_elements(By.TAG_NAME, "input")
                for field in input_fields:
                    if field.is_displayed() and field.get_attribute("type") in ["number", "text", "password"]:
                        print("✅ Found visible input field on TOTP page")
                        return field
                        
        except Exception as e:
            print(f"❌ Error in strategy 5: {e}")
        
        print("❌ No TOTP field found with any strategy")
        return None
    
    def click_continue_button(self, driver):
        """Click continue button using multiple strategies that handle the HTML comment issue"""
        print("🔘 Attempting to click continue button...")
        
        # Strategy 1: Direct click by class and type (most specific)
        try:
            button = driver.find_element(By.XPATH, "//button[@type='submit' and contains(@class, 'button-orange')]")
            if button.is_displayed() and button.is_enabled():
                button.click()
                print("✅ Clicked continue button by type and class")
                logger.info("Continue button clicked by type and class")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 1 failed: button by type and class")
        
        # Strategy 2: Find button by class only
        try:
            button = driver.find_element(By.CLASS_NAME, "button-orange")
            if button.is_displayed() and button.is_enabled():
                button.click()
                print("✅ Clicked continue button by class")
                logger.info("Continue button clicked by class")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 2 failed: button by class")
        
        # Strategy 3: Find submit button in 2FA form
        try:
            twofa_form = driver.find_element(By.CLASS_NAME, "twofa-form")
            button = twofa_form.find_element(By.XPATH, ".//button[@type='submit']")
            if button.is_displayed() and button.is_enabled():
                button.click()
                print("✅ Clicked submit button in 2FA form")
                logger.info("Submit button in 2FA form clicked")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 3 failed: submit button in 2FA form")
        
        # Strategy 4: Find button with 'wide' class
        try:
            button = driver.find_element(By.XPATH, "//button[contains(@class, 'wide')]")
            if button.is_displayed() and button.is_enabled():
                button.click()
                print("✅ Clicked button with 'wide' class")
                logger.info("Button with 'wide' class clicked")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 4 failed: button with 'wide' class")
        
        # Strategy 5: Find any button in actions div
        try:
            actions_div = driver.find_element(By.CLASS_NAME, "actions")
            button = actions_div.find_element(By.TAG_NAME, "button")
            if button.is_displayed() and button.is_enabled():
                button.click()
                print("✅ Clicked button in actions div")
                logger.info("Button in actions div clicked")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 5 failed: button in actions div")
        
        # Strategy 6: Use JavaScript to click the button
        try:
            button = driver.find_element(By.XPATH, "//button[@type='submit']")
            if button.is_displayed() and button.is_enabled():
                driver.execute_script("arguments[0].click();", button)
                print("✅ Clicked continue button using JavaScript")
                logger.info("Continue button clicked using JavaScript")
                return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 6 failed: JavaScript click")
        
        # Strategy 7: Press Enter on the TOTP field
        try:
            totp_field = self.find_totp_field(driver)
            if totp_field:
                totp_field.send_keys(Keys.RETURN)
                print("✅ Pressed Enter on TOTP field")
                logger.info("Enter key pressed on TOTP field")
                return True
        except Exception as e:
            print(f"⚠️ Strategy 7 failed: Enter key - {e}")
        
        # Strategy 8: Look for any visible button and click it
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            for button in buttons:
                if button.is_displayed() and button.is_enabled():
                    button_class = button.get_attribute("class")
                    button_type = button.get_attribute("type")
                    if "orange" in button_class or button_type == "submit":
                        button.click()
                        print(f"✅ Clicked button: class='{button_class}', type='{button_type}'")
                        logger.info(f"Button clicked: class='{button_class}', type='{button_type}'")
                        return True
        except (NoSuchElementException, StaleElementReferenceException):
            print("⚠️ Strategy 8 failed: any visible button")
        
        print("❌ No continue button found with any strategy")
        return False
    
    def auto_login_and_get_token(self):
        """Automatically login and get access token"""
        print("\n🚀 Starting auto-login process...")
        driver = None
        
        try:
            # Setup Chrome driver
            driver = self.setup_chrome_driver()
            
            # Navigate to login page
            login_url = f"https://kite.zerodha.com/connect/login?api_key={self.api_key}&v=3"
            print(f"🌐 Navigating to: {login_url}")
            logger.info(f"Opening login URL: {login_url}")
            
            driver.get(login_url)
            print("✅ Login page loaded")
            
            # Wait for page to load
            time.sleep(3)
            
            # Step 1: Fill username and password
            print("\n👤 Step 1: Filling credentials...")
            
            try:
                wait = WebDriverWait(driver, 20)
                username_field = wait.until(EC.presence_of_element_located((By.ID, "userid")))
                username_field.clear()
                username_field.send_keys(self.username)
                print("✅ Username entered")
                logger.info("Username field filled")
            except TimeoutException:
                print("❌ Username field not found")
                self.email_manager.send_error_notification("Login Error", "Username field not found", "SYSTEM", True)
                return None
            
            try:
                password_field = driver.find_element(By.ID, "password")
                password_field.clear()
                password_field.send_keys(self.password)
                print("✅ Password entered")
                logger.info("Password field filled")
            except NoSuchElementException:
                print("❌ Password field not found")
                self.email_manager.send_error_notification("Login Error", "Password field not found", "SYSTEM", True)
                return None
            
            # Click login button
            try:
                login_button = driver.find_element(By.XPATH, "//button[contains(text(),'Login')]")
                login_button.click()
                print("✅ Login button clicked")
                logger.info("Login button clicked")
            except NoSuchElementException:
                print("❌ Login button not found")
                self.email_manager.send_error_notification("Login Error", "Login button not found", "SYSTEM", True)
                return None
            
            # Step 2: Check if we're redirected directly (no 2FA needed)
            print("\n🔍 Step 2: Checking for direct redirect...")
            time.sleep(5)
            
            current_url = driver.current_url
            if "request_token=" in current_url:
                print("✅ Direct redirect! No 2FA needed.")
                request_token = current_url.split("request_token=")[1].split("&")[0]
                print(f"✅ Request token: {request_token}")
                
                # Generate access token
                kite = KiteConnect(api_key=self.api_key)
                data = kite.generate_session(request_token, api_secret=self.api_secret)
                access_token = data["access_token"]
                
                with open("access_token.txt", "w") as f:
                    f.write(access_token)
                
                print("✅ Access token saved!")
                return access_token
            
            # Step 3: Handle 2FA
            print("\n🔐 Step 3: Handling 2FA...")
            
            # Wait for 2FA page to load
            time.sleep(5)
            
            totp_field = self.find_totp_field(driver)
            if not totp_field:
                print("❌ Could not find TOTP field")
                print("📄 Current page source (first 500 chars):")
                print(driver.page_source[:500])
                self.email_manager.send_error_notification("2FA Error", "TOTP field not found", "SYSTEM", True)
                return None
            
            # Generate and enter TOTP
            totp_code = self.generate_totp()
            
            try:
                totp_field.clear()
                totp_field.send_keys(totp_code)
                print(f"✅ TOTP entered: {totp_code}")
                logger.info(f"TOTP entered: {totp_code}")
            except StaleElementReferenceException:
                print("⚠️ TOTP field became stale, refinding...")
                totp_field = self.find_totp_field(driver)
                if totp_field:
                    totp_field.clear()
                    totp_field.send_keys(totp_code)
                    print(f"✅ TOTP entered after refinding: {totp_code}")
                    logger.info(f"TOTP entered after refinding: {totp_code}")
                else:
                    print("❌ Could not refind TOTP field")
                    self.email_manager.send_error_notification("2FA Error", "TOTP field became stale", "SYSTEM", True)
                    return None
            
            # Click continue button
            if not self.click_continue_button(driver):
                print("❌ Failed to click continue button")
                self.email_manager.send_error_notification("2FA Error", "Failed to click continue button", "SYSTEM", True)
                return None
            
            # Step 4: Wait for redirect
            print("\n🔄 Step 4: Waiting for redirect...")
            
            try:
                wait = WebDriverWait(driver, 30)
                wait.until(lambda d: "request_token=" in d.current_url or "status=success" in d.current_url)
                current_url = driver.current_url
                print(f"✅ Redirected to: {current_url}")
                logger.info(f"Redirect URL: {current_url}")
                
            except TimeoutException:
                print("❌ Redirect timeout")
                current_url = driver.current_url
                print(f"Current URL: {current_url}")
                
                # Check if there's an error message
                if "invalid" in driver.page_source.lower() or "incorrect" in driver.page_source.lower():
                    print("🚨 Invalid TOTP code detected")
                    self.email_manager.send_error_notification("2FA Error", "Invalid TOTP code", "SYSTEM", True)
                    return None
                
                self.email_manager.send_error_notification("Login Error", "Redirect timeout", "SYSTEM", True)
                return None
            
            # Step 5: Extract request token
            print("\n🔑 Step 5: Extracting request token...")
            
            if "request_token=" not in current_url:
                print("❌ Request token not found in URL")
                print(f"URL: {current_url}")
                self.email_manager.send_error_notification("Login Error", "Request token not found in URL", "SYSTEM", True)
                return None
            
            request_token = current_url.split("request_token=")[1].split("&")[0]
            print(f"✅ Request token extracted: {request_token}")
            logger.info(f"Request token: {request_token}")
            
            # Step 6: Generate access token
            print("\n🎫 Step 6: Generating access token...")
            
            kite = KiteConnect(api_key=self.api_key)
            data = kite.generate_session(request_token, api_secret=self.api_secret)
            access_token = data["access_token"]
            
            print(f"✅ Access token generated: {access_token[:20]}...")
            logger.info(f"Access token generated: {access_token[:20]}...")
            
            # Step 7: Save token
            with open("access_token.txt", "w") as f:
                f.write(access_token)
            print("✅ Access token saved to file")
            logger.info("Access token saved to access_token.txt")
            
            return access_token
            
        except Exception as e:
            print(f"❌ Auto-login failed: {e}")
            logger.error(f"Auto-login failed: {e}", exc_info=True)
            self.email_manager.send_error_notification("Login Error", str(e), "SYSTEM", True)
            return None
            
        finally:
            if driver:
                print("🔄 Closing browser...")
                driver.quit()
    
    def check_existing_token(self):
        """Check if existing token is valid"""
        print("\n🔍 Checking existing token...")
        
        if not os.path.exists("access_token.txt"):
            print("❌ No existing token file found")
            return None
        
        try:
            with open("access_token.txt", "r") as f:
                token = f.read().strip()
            
            if not token:
                print("❌ Token file is empty")
                return None
            
            print(f"📝 Found existing token: {token[:20]}...")
            
            # Test token validity
            kite = KiteConnect(api_key=self.api_key)
            kite.set_access_token(token)
            profile = kite.profile()
            
            print(f"✅ Existing token is valid for: {profile['user_name']}")
            logger.info(f"Existing token valid for: {profile['user_name']}")
            return token
            
        except exceptions.TokenException:
            print("❌ Existing token is expired")
            logger.info("Existing token expired")
            return None
        except Exception as e:
            print(f"❌ Error checking existing token: {e}")
            logger.error(f"Error checking existing token: {e}")
            return None
    
    def refresh_token_if_needed(self):
        """Check if token needs refresh and do it automatically"""
        print("\n🔄 Starting token refresh process...")
        
        # First, check existing token
        existing_token = self.check_existing_token()
        if existing_token:
            return existing_token
        
        # If no valid token, get new one
        print("\n🆕 Getting new token...")
        return self.auto_login_and_get_token()

class DailyTradingAgent:
    def __init__(self, model, symbol, paper_trading=True):
        self.model = model
        self.symbol = symbol
        self.paper_trading = paper_trading
        self.kite = None
        self.token = None
        self.df = pd.DataFrame()
        self.current_step = 0
        
        # Portfolio state
        self.balance = INITIAL_BALANCE
        self.initial_balance = INITIAL_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.position = {'quantity': 0, 'avg_price': 0, 'last_price': 0, 'pnl': 0}
        self.order_history = []
        
        # Initialize email manager
        self.email_manager = EmailNotificationManager()
        
        # Initialize token manager
        self.token_manager = ZerodhaTokenManager(self.email_manager)
        
        # Database connection
        self.db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.create_tables()
        
        # Setup Zerodha connection
        self.setup_zerodha()
        
        logger.info(f"Daily Trading Agent initialized for {symbol}")
        logger.info(f"Paper Trading: {paper_trading}")
        logger.info(f"Initial Balance: ₹{self.balance:.2f}")

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
                paper_trading BOOLEAN,
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
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
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
        
        # Pending AMO orders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pending_amo_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                symbol TEXT,
                order_type TEXT,
                quantity INTEGER,
                price REAL,
                transaction_cost REAL,
                execution_date TEXT,
                status TEXT DEFAULT 'PENDING'
            )''')
        
        self.db_conn.commit()
        logger.info("Database tables created/verified")

    def setup_zerodha(self):
        """Setup Zerodha KiteConnect API with automatic token management"""
        try:
            print("🔧 Setting up Zerodha connection...")
            self.kite = KiteConnect(api_key=API_KEY)
            
            # Get valid access token using token manager
            access_token = self.token_manager.refresh_token_if_needed()
            
            if not access_token:
                error_msg = "Failed to get valid access token"
                if not self.paper_trading:
                    self.email_manager.send_error_notification("Zerodha Setup", error_msg, self.symbol, self.paper_trading)
                    raise ValueError(error_msg + " for live trading")
                else:
                    logger.warning(error_msg + ", continuing in paper trading mode")
                    return
            
            # Set access token
            self.kite.set_access_token(access_token)
            
            # Get instrument token
            instruments = self.kite.instruments("NSE")
            for inst in instruments:
                if inst['tradingsymbol'] == self.symbol:
                    self.token = inst['instrument_token']
                    break
            
            if not self.token:
                error_msg = f"Instrument token not found for {self.symbol}"
                self.email_manager.send_error_notification("Zerodha Setup", error_msg, self.symbol, self.paper_trading)
                raise ValueError(error_msg)
            
            logger.info(f"✅ Zerodha setup complete. Token: {self.token}")
            print(f"✅ Zerodha setup complete. Token: {self.token}")
            
            # Send login success notification
            self.email_manager.send_login_success(self.symbol, self.paper_trading)
            
        except Exception as e:
            logger.error(f"Zerodha setup failed: {e}")
            print(f"❌ Zerodha setup failed: {e}")
            self.email_manager.send_error_notification("Zerodha Setup", str(e), self.symbol, self.paper_trading)
            if not self.paper_trading:
                raise
            else:
                logger.warning("Continuing in paper trading mode without Zerodha connection")

    def refresh_zerodha_token(self):
        """Refresh Zerodha token if needed"""
        try:
            print("🔄 Refreshing Zerodha token...")
            new_token = self.token_manager.refresh_token_if_needed()
            if new_token and self.kite:
                self.kite.set_access_token(new_token)
                logger.info("✅ Zerodha token refreshed successfully")
                print("✅ Zerodha token refreshed successfully")
                return True
            print("❌ Failed to refresh Zerodha token")
            return False
        except Exception as e:
            logger.error(f"Failed to refresh Zerodha token: {e}")
            print(f"❌ Failed to refresh Zerodha token: {e}")
            self.email_manager.send_error_notification("Token Refresh", str(e), self.symbol, self.paper_trading)
            return False

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

    def get_current_market_price(self):
        """Get current market price (for AMO execution simulation)"""
        try:
            if self.kite and self.token:
                # Get current market price from Zerodha
                ltp = self.kite.ltp(f"NSE:{self.symbol}")
                return ltp[f"NSE:{self.symbol}"]['last_price']
            else:
                # Fallback to last known price
                return self.position['last_price']
        except Exception as e:
            logger.error(f"Error getting current market price: {e}")
            return self.position['last_price']

    def store_pending_amo_order(self, order):
        """Store AMO order for execution next trading day"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO pending_amo_orders (timestamp, symbol, order_type, quantity, 
                                              price, transaction_cost, execution_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order['timestamp'], order['symbol'], order['type'], order['quantity'],
                order['price'], order['transaction_cost'], order['execution_date'], 'PENDING'
            ))
            self.db_conn.commit()
            logger.info(f"AMO order stored for execution on {order['execution_date']}")
            
        except Exception as e:
            logger.error(f"Error storing AMO order: {e}")
            self.email_manager.send_error_notification("AMO Storage", str(e), self.symbol, self.paper_trading)

    def execute_pending_amo_orders(self):
        """Execute pending AMO orders at market open (for paper trading)"""
        if not self.paper_trading:
            return
        
        try:
            cursor = self.db_conn.cursor()
            today = datetime.now(TIMEZONE).strftime('%Y-%m-%d')
            
            # Get pending AMO orders for today
            cursor.execute('''
                SELECT * FROM pending_amo_orders 
                WHERE execution_date = ? AND status = 'PENDING'
            ''', (today,))
            
            pending_orders = cursor.fetchall()
            
            if not pending_orders:
                return
            
            logger.info(f"Executing {len(pending_orders)} pending AMO orders")
            
            for order_data in pending_orders:
                order_id, timestamp, symbol, order_type, quantity, price, transaction_cost, execution_date, status = order_data
                
                # Simulate order execution at market open price
                current_price = self.get_current_market_price() or price
                
                logger.info(f"📈 Executing AMO order: {order_type} {quantity} shares at ₹{current_price:.2f}")
                
                # Execute the order
                trade_value = quantity * current_price
                
                if order_type == "BUY":
                    total_cost = trade_value + transaction_cost
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        
                        # Update position
                        if self.shares_held >= 0:
                            total_value = self.shares_held * self.cost_basis + trade_value
                            self.shares_held += quantity
                            self.cost_basis = total_value / self.shares_held if self.shares_held > 0 else 0
                        else:
                            self.shares_held += quantity
                        
                        execution_status = 'COMPLETED'
                        logger.info(f"✅ AMO BUY executed: {quantity} shares at ₹{current_price:.2f}")
                    else:
                        execution_status = 'REJECTED - Insufficient funds'
                        logger.warning(f"❌ AMO BUY rejected: Insufficient funds")
                
                elif order_type == "SELL":
                    if self.shares_held >= quantity:
                        self.balance += (trade_value - transaction_cost)
                        self.shares_held -= quantity
                        
                        if self.shares_held == 0:
                            self.cost_basis = 0
                        
                        execution_status = 'COMPLETED'
                        logger.info(f"✅ AMO SELL executed: {quantity} shares at ₹{current_price:.2f}")
                    else:
                        execution_status = 'REJECTED - Insufficient shares'
                        logger.warning(f"❌ AMO SELL rejected: Insufficient shares")
                
                # Update order status
                cursor.execute('''
                    UPDATE pending_amo_orders 
                    SET status = ? 
                    WHERE id = ?
                ''', (execution_status, order_id))
                
                # Log to main orders table
                executed_order = {
                    'symbol': symbol,
                    'type': order_type,
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now(TIMEZONE).isoformat(),
                    'transaction_cost': transaction_cost,
                    'status': execution_status,
                    'order_id': f"AMO_{order_id}"
                }
                self.save_order(executed_order)
                
                # Send order notification
                self.email_manager.send_order_notification(executed_order, self.symbol, self.paper_trading)
            
            self.db_conn.commit()
            
        except Exception as e:
            logger.error(f"Error executing AMO orders: {e}")
            self.email_manager.send_error_notification("AMO Execution", str(e), self.symbol, self.paper_trading)

    def fetch_historical_data(self, interval='day', days=365):
        """Fetch historical data from Zerodha with token refresh"""
        try:
            if not self.kite or not self.token:
                logger.error("Kite client or token not available")
                return pd.DataFrame()
            
            to_date = datetime.now(TIMEZONE)
            from_date = to_date - timedelta(days=days)
            
            logger.info(f"Fetching historical data: {from_date.date()} to {to_date.date()}")
            
            try:
                data = self.kite.historical_data(
                    self.token, from_date, to_date, interval, continuous=False, oi=False
                )
            except exceptions.TokenException:
                logger.warning("Token expired, refreshing...")
                if self.refresh_zerodha_token():
                    data = self.kite.historical_data(
                        self.token, from_date, to_date, interval, continuous=False, oi=False
                    )
                else:
                    logger.error("Failed to refresh token")
                    return pd.DataFrame()
            
            if not data:
                logger.error("No historical data received")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df.rename(columns={
                'date': 'Date', 'open': 'Open', 'high': 'High',
                'low': 'Low', 'close': 'Close', 'volume': 'Volume'
            }, inplace=True)
            
            logger.info(f"✅ Fetched {len(df)} historical data points")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {e}")
            self.email_manager.send_error_notification("Data Fetch", str(e), self.symbol, self.paper_trading)
            return pd.DataFrame()

    def calculate_transaction_costs(self, trade_value, is_delivery=True):
        """Calculate transaction costs for a trade"""
        brokerage_rate = BROKERAGE_DELIVERY if is_delivery else BROKERAGE_INTRADAY
        stt_rate = STT_DELIVERY if is_delivery else STT_INTRADAY
        
        brokerage = min(trade_value * brokerage_rate, 20)  # Cap at ₹20
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
                self.balance / self.initial_balance,  # Normalized balance
                self.shares_held / 100,  # Normalized shares held
                self.cost_basis / current_price if self.cost_basis > 0 else 0,  # Relative cost basis
                current_price / close_mean if close_mean > 0 else 1,  # Price relative to mean
                1 if self.shares_held > 0 else 0,  # Long position
                1 if self.shares_held < 0 else 0,  # Short position
                0,  # Trailing stop (placeholder)
                0,  # Target price (placeholder)
            ]
            
            # OHLCV data (normalized)
            for col in ["Open", "High", "Low", "Close"]:
                obs.append(row[col] / close_mean if close_mean > 0 else 1)
            
            volume_mean = window_data["Volume"].mean()
            obs.append(row["Volume"] / volume_mean if volume_mean > 0 else 1)
            
            # Technical indicators
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
                    obs.append(0.5)  # Default value for missing indicators
            
            # Additional features to reach 53 features
            while len(obs) < 53:
                obs.append(0.5)
            
            # Ensure exactly 53 features
            obs = obs[:53]
            
            return np.nan_to_num(np.array(obs, dtype=np.float32))
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            self.email_manager.send_error_notification("Observation Preparation", str(e), self.symbol, self.paper_trading)
            return None

    def place_order(self, order_type, quantity, price, order_variety="regular"):
        """Place an order (paper trading now mimics live trading exactly)"""
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
            'order_id': f"PAPER_{int(time.time())}" if self.paper_trading else None,
            'order_variety': order_variety
        }
        
        if self.paper_trading:
            # Paper trading now mimics live trading exactly
            if order_variety == "amo":
                # Simulate AMO behavior - order queued for next day
                order['status'] = 'AMO_PLACED'
                order['execution_date'] = (datetime.now(TIMEZONE) + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"📝 Paper AMO queued: {order_type} {quantity} shares at ₹{price:.2f} for next trading day")
                
                # Store AMO order for execution simulation tomorrow
                self.store_pending_amo_order(order)
                
            else:
                # Regular order - immediate execution (only during market hours)
                if self.is_market_open():
                    if order_type == "BUY":
                        total_cost = trade_value + transaction_cost
                        if self.balance >= total_cost:
                            self.balance -= total_cost
                            
                            # Update position
                            if self.shares_held >= 0:
                                total_value = self.shares_held * self.cost_basis + trade_value
                                self.shares_held += quantity
                                self.cost_basis = total_value / self.shares_held if self.shares_held > 0 else 0
                            else:
                                self.shares_held += quantity
                            
                            order['status'] = 'COMPLETED'
                            logger.info(f"✅ Paper BUY executed: {quantity} shares at ₹{price:.2f}")
                        else:
                            order['status'] = 'REJECTED - Insufficient funds'
                            logger.warning(f"❌ Paper BUY rejected: Insufficient funds")
                    
                    elif order_type == "SELL":
                        if self.shares_held >= quantity:
                            self.balance += (trade_value - transaction_cost)
                            self.shares_held -= quantity
                            
                            if self.shares_held == 0:
                                self.cost_basis = 0
                            
                            order['status'] = 'COMPLETED'
                            logger.info(f"✅ Paper SELL executed: {quantity} shares at ₹{price:.2f}")
                        else:
                            order['status'] = 'REJECTED - Insufficient shares'
                            logger.warning(f"❌ Paper SELL rejected: Insufficient shares")
                else:
                    order['status'] = 'REJECTED - Market closed'
                    logger.warning(f"❌ Paper order rejected: Market is closed")
        
        else:
            # Live trading execution
            try:
                order_params = {
                    'variety': order_variety,
                    'exchange': 'NSE',
                    'tradingsymbol': self.symbol,
                    'transaction_type': order_type,
                    'quantity': quantity,
                    'order_type': 'MARKET',
                    'product': 'CNC',
                    'validity': 'DAY'
                }
                
                if order_variety == "amo":
                    order_params['validity'] = 'DAY'
                
                try:
                    order_id = self.kite.place_order(**order_params)
                except exceptions.TokenException:
                    logger.warning("Token expired during order placement, refreshing...")
                    if self.refresh_zerodha_token():
                        order_id = self.kite.place_order(**order_params)
                    else:
                        raise ValueError("Failed to refresh token for order placement")
                
                order['order_id'] = order_id
                order['status'] = 'PLACED'
                
                logger.info(f"✅ Live order placed: {order_id}")
                
            except Exception as e:
                order['status'] = f'FAILED: {str(e)}'
                logger.error(f"❌ Live order failed: {e}")
                self.email_manager.send_error_notification("Order Placement", str(e), self.symbol, self.paper_trading)
        
        # Update position tracking
        self.position['quantity'] = self.shares_held
        self.position['avg_price'] = self.cost_basis
        self.position['last_price'] = price
        self.position['pnl'] = self.shares_held * (price - self.cost_basis) if self.cost_basis > 0 else 0
        
        # Save order to database
        self.save_order(order)
        self.order_history.append(order)
        
        # Send order notification (only for immediate executions, not AMO queuing)
        if order['status'] not in ['AMO_PLACED', 'PENDING']:
            self.email_manager.send_order_notification(order, self.symbol, self.paper_trading)
        
        return order

    def save_order(self, order):
        """Save order to database"""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute('''
                INSERT INTO orders (timestamp, symbol, order_type, quantity, price, 
                                  status, transaction_cost, order_id, paper_trading, order_variety)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                order['timestamp'], order['symbol'], order['type'], order['quantity'],
                order['price'], order['status'], order['transaction_cost'],
                order.get('order_id', ''), self.paper_trading, order.get('order_variety', 'regular')
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Error saving order: {e}")
            self.email_manager.send_error_notification("Database Error", str(e), self.symbol, self.paper_trading)

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
            
            logger.info(f"📊 Daily Snapshot - Balance: ₹{self.balance:.2f}, Equity: ₹{equity:.2f}, PnL: ₹{total_pnl:.2f}")
            
            # Send PnL report
            self.email_manager.send_pnl_report(
                self.symbol, self.balance, equity, position_value, total_pnl,
                self.shares_held, self.cost_basis, self.position['last_price'], self.paper_trading
            )
            
        except Exception as e:
            logger.error(f"Error saving daily snapshot: {e}")
            self.email_manager.send_error_notification("Snapshot Error", str(e), self.symbol, self.paper_trading)

    def daily_analysis_and_trading(self):
        """Enhanced daily analysis with AMO execution check and email notifications"""
        try:
            logger.info("🔍 Starting daily analysis...")
            
            # 1. Execute any pending AMO orders first (paper trading)
            if self.paper_trading:
                self.execute_pending_amo_orders()
            
            # 2. Refresh token if needed
            if not self.paper_trading:
                print("🔄 Refreshing token before analysis...")
                self.refresh_zerodha_token()
            
            # 3. Fetch latest historical data
            df = self.fetch_historical_data(interval='day', days=365)
            if df.empty:
                logger.error("❌ Failed to fetch historical data")
                self.email_manager.send_error_notification("Data Fetch", "Failed to fetch historical data", self.symbol, self.paper_trading)
                return
            
            # 4. Calculate technical indicators
            df = self.calculate_all_indicators(df)
            if df.empty:
                logger.error("❌ Failed to calculate indicators")
                self.email_manager.send_error_notification("Indicator Calculation", "Failed to calculate indicators", self.symbol, self.paper_trading)
                return
            
            # 5. Update internal state
            self.df = df
            self.current_step = len(df) - 1  # Point to latest complete day
            
            # 6. Get current price (today's closing price)
            current_price = df.iloc[-1]['Close']
            self.position['last_price'] = current_price
            
            # 7. Generate observation
            obs = self.prepare_observation()
            if obs is None:
                logger.error("❌ Failed to prepare observation")
                self.email_manager.send_error_notification("Observation Error", "Failed to prepare observation", self.symbol, self.paper_trading)
                return
            
            # 8. Get model prediction
            action, _states = self.model.predict(obs, deterministic=True)
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            predicted_action = action_names[int(action)]
            
            logger.info(f"🤖 Model Prediction: {predicted_action} at ₹{current_price:.2f}")
            
            # 9. Calculate position size
            position_size = self.calculate_position_size(current_price)
            
            # 10. Execute trade based on prediction (now uses AMO for both paper and live)
            order = None
            if predicted_action == "BUY" and position_size > 0:
                order = self.place_order("BUY", position_size, current_price, "amo")
            elif predicted_action == "SELL" and self.shares_held > 0:
                sell_quantity = min(position_size, self.shares_held)
                order = self.place_order("SELL", sell_quantity, current_price, "amo")
            
            # 11. Save analysis to database
            self.save_daily_analysis(predicted_action, current_price, obs)
            
            # 12. Save daily snapshot and send PnL report
            self.save_daily_snapshot()
            
            # 13. Send daily analysis summary
            analysis_result = {
                'action': predicted_action,
                'price': current_price
            }
            self.email_manager.send_daily_summary(self.symbol, analysis_result, self.paper_trading)
            
            logger.info("✅ Daily analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Error in daily analysis: {e}")
            self.email_manager.send_error_notification("Daily Analysis", str(e), self.symbol, self.paper_trading)

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
            self.email_manager.send_error_notification("Database Error", str(e), self.symbol, self.paper_trading)

    def is_trading_day(self):
        """Check if today is a trading day"""
        now = datetime.now(TIMEZONE)
        return now.weekday() < 5  # Monday=0, Friday=4

    def get_next_analysis_time(self):
        """Get next analysis time (market close + 30 minutes)"""
        now = datetime.now(TIMEZONE)
        
        # Market closes at 3:30 PM, analysis at 4:00 PM
        analysis_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # If current time is past analysis time, move to next trading day
        if now >= analysis_time:
            analysis_time += timedelta(days=1)
            
        # Skip weekends
        while analysis_time.weekday() >= 5:
            analysis_time += timedelta(days=1)
            
        return analysis_time

    def run_daily_strategy(self):
        """Main daily trading loop with comprehensive error handling"""
        logger.info("🚀 Starting Daily Trading Strategy with Auto Token Management & Email Notifications")
        
        while True:
            try:
                now = datetime.now(TIMEZONE)
                
                # Check if it's a trading day
                if not self.is_trading_day():
                    logger.info("📅 Weekend/Holiday - No trading")
                    time.sleep(3600)  # Sleep for 1 hour
                    continue
                
                # Get next analysis time
                next_analysis = self.get_next_analysis_time()
                
                # Wait until analysis time
                wait_seconds = (next_analysis - now).total_seconds()
                if wait_seconds > 0:
                    logger.info(f"⏰ Next analysis at {next_analysis.strftime('%Y-%m-%d %H:%M:%S')} (in {wait_seconds/3600:.1f} hours)")
                    time.sleep(min(wait_seconds, 3600))  # Sleep max 1 hour at a time
                    continue
                
                # Perform daily analysis
                self.daily_analysis_and_trading()
                
                # Sleep for 2 hours after analysis
                time.sleep(7200)
                
            except KeyboardInterrupt:
                logger.info("🛑 Trading stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in daily strategy: {e}")
                self.email_manager.send_error_notification("Strategy Error", str(e), self.symbol, self.paper_trading)
                time.sleep(300)  # Sleep 5 minutes on error
        
        # Cleanup
        self.db_conn.close()
        logger.info("✅ Daily trading strategy ended")

    def calculate_all_indicators(self, df):
        """Calculate all technical indicators"""
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
            self.email_manager.send_error_notification("Indicator Calculation", str(e), self.symbol, self.paper_trading)
            return pd.DataFrame()

def start_daily_trading(symbol, model_path, paper_trading=True):
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
        agent = DailyTradingAgent(model, symbol, paper_trading)
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
    parser = argparse.ArgumentParser(description='Daily RL Trading Bot with Auto Token Management')
    parser.add_argument('--symbol', type=str, default="ADANIPORTS", help='Stock symbol to trade')
    parser.add_argument('--live', action='store_true', help='Enable live trading (default: paper trading)')
    
    args = parser.parse_args()
    
    # Start daily trading
    start_daily_trading(
        symbol=args.symbol,
        model_path=args.symbol,
        paper_trading=not args.live
    )

# python daily_trading_bot.py --symbol ADANIPORTS
# 


# sudo nano /etc/systemd/system/trading-bot.service

# [Unit]
# Description=RL Trading Bot Service
# After=network.target

# [Service]
# Type=simple
# User=root
# WorkingDirectory=/var/www/html/python/paper_trading
# ExecStart=/usr/bin/python3 /var/www/html/python/paper_trading/daily_trading_bot.py --symbol ADANIPORTS
# Restart=always
# RestartSec=10
# StandardOutput=journal
# StandardError=journal
# Environment=PYTHONPATH=/var/www/html/python/paper_trading

# [Install]
# WantedBy=multi-user.target

# sudo chmod 644 /etc/systemd/system/trading-bot.service


# # Reload systemd to recognize the new service
# sudo systemctl daemon-reload

# # Enable the service to start automatically on boot
# sudo systemctl enable trading-bot.service

# # Start the service immediately
# sudo systemctl start trading-bot.service

# # Check if the service is running
# sudo systemctl status trading-bot.service

# # View real-time logs
# sudo journalctl -u trading-bot.service -f

# # View logs since boot
# sudo journalctl -u trading-bot.service --since today