import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
import time
import logging
import traceback
from kiteconnect import KiteConnect
from kiteconnect import exceptions
from dotenv import load_dotenv
import pyotp

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("login_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("login_bot")

class ZerodhaTokenManager:
    def __init__(self):
        print("🔧 Initializing ZerodhaTokenManager...")
        
        self.api_key = os.getenv("ZERODHA_API_KEY")
        self.api_secret = os.getenv("ZERODHA_API_SECRET")
        self.username = os.getenv("ZERODHA_USER_ID")
        self.password = os.getenv("ZERODHA_PASSWORD")
        self.totp_secret = os.getenv("ZERODHA_TOTP_SECRET")
        
        if not all([self.api_key, self.api_secret, self.username, self.password, self.totp_secret]):
            raise ValueError("Missing required environment variables")
        
        logger.info("✅ ZerodhaTokenManager initialized successfully")
    
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
        """Click continue button with multiple strategies that handle the HTML comment issue"""
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
                return None
            
            try:
                password_field = driver.find_element(By.ID, "password")
                password_field.clear()
                password_field.send_keys(self.password)
                print("✅ Password entered")
                logger.info("Password field filled")
            except NoSuchElementException:
                print("❌ Password field not found")
                return None
            
            # Click login button
            try:
                login_button = driver.find_element(By.XPATH, "//button[contains(text(),'Login')]")
                login_button.click()
                print("✅ Login button clicked")
                logger.info("Login button clicked")
            except NoSuchElementException:
                print("❌ Login button not found")
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
                    return None
            
            # Click continue button
            if not self.click_continue_button(driver):
                print("❌ Failed to click continue button")
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
                    return None
                
                return None
            
            # Step 5: Extract request token
            print("\n🔑 Step 5: Extracting request token...")
            
            if "request_token=" not in current_url:
                print("❌ Request token not found in URL")
                print(f"URL: {current_url}")
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

def main():
    """Main function"""
    print("🚀 Zerodha Token Manager - Fixed Continue Button HTML Comment Issue")
    print("=" * 70)
    
    try:
        token_manager = ZerodhaTokenManager()
        token = token_manager.refresh_token_if_needed()
        
        if token:
            print(f"\n🎉 SUCCESS! Token: {token[:20]}...")
            
            # Verify token
            kite = KiteConnect(api_key=token_manager.api_key)
            kite.set_access_token(token)
            profile = kite.profile()
            print(f"✅ Valid for: {profile['user_name']} ({profile['user_id']})")
            print(f"📧 Email: {profile['email']}")
            
        else:
            print("\n💥 Failed to get token")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
