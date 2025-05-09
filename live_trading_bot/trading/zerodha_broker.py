"""
Zerodha broker integration for executing trades.
"""
import os
import time
import logging
import pyotp
from datetime import datetime
from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException, NetworkException

logger = logging.getLogger(__name__)

class ZerodhaBroker:
    """Zerodha broker interface for trading operations"""
    
    def __init__(self, api_key, api_secret, totp_secret=None, redirect_url=None, mode='paper'):
        """
        Initialize Zerodha broker
        
        Args:
            api_key (str): Zerodha API key
            api_secret (str): Zerodha API secret
            totp_secret (str, optional): TOTP secret for automated login
            redirect_url (str, optional): Redirect URL for authentication
            mode (str): Trading mode - 'live', 'paper', or 'backtest'
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.totp_secret = totp_secret
        self.redirect_url = redirect_url
        self.mode = mode
        self.kite = KiteConnect(api_key=api_key)
        self.authenticated = False
        self.access_token = None
        self.positions = {}
        self.orders = {}
        self.last_order_id = 0
        
        # For paper trading mode
        self.paper_positions = {}
        self.paper_balance = 100000  # Default paper trading balance
        self.paper_orders = {}
        
        # For request throttling (Zerodha rate limits)
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 200ms between requests (max 5 req/sec)
        
        # Token file for persistent sessions
        self.token_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                     'config', 'zerodha_token.txt')
    
    def _throttle_request(self):
        """Throttle API requests to avoid rate limits"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
            
        self.last_request_time = time.time()
    
    def authenticate(self):
        """
        Authenticate with Zerodha API
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        if self.mode == 'backtest':
            logger.info("Running in backtest mode, no authentication required")
            return True
            
        # Check if paper trading mode
        if self.mode == 'paper':
            logger.info("Running in paper trading mode, no real authentication required")
            self.authenticated = True
            return True
        
        # Try to load token from file first
        if self._load_token_from_file():
            return True
            
        # No saved token, start fresh authentication
        try:
            # Generate login URL
            login_url = self.kite.login_url()
            
            if self.totp_secret:
                # Automated login using TOTP
                logger.info("Attempting automated login with TOTP")
                # This is a placeholder - actual automated login would require browser automation
                # with tools like Selenium, which is complex for a code sample
                
                # For illustration:
                # 1. Generate TOTP
                totp = pyotp.TOTP(self.totp_secret)
                topt_value = totp.now()
                logger.info(f"Generated TOTP: {topt_value}")
                
                # 2. Here you would use browser automation to:
                #    - Open login_url
                #    - Enter user ID
                #    - Enter password
                #    - Enter TOTP
                #    - Extract the request token from the redirect URL
                
                # Placeholder for the request token you'd get from automation
                request_token = "PLACEHOLDER_TOKEN_FROM_AUTOMATION"
                
                # Generate session from request token
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.kite.set_access_token(data["access_token"])
                self.access_token = data["access_token"]
                self.authenticated = True
                
                # Save token to file
                self._save_token_to_file()
                
                logger.info("Authenticated successfully with Zerodha")
                return True
            else:
                # Manual login required
                logger.info(f"Please visit the following URL to authenticate: {login_url}")
                request_token = input("Enter the request token: ")
                
                # Generate session from request token
                data = self.kite.generate_session(request_token, api_secret=self.api_secret)
                self.kite.set_access_token(data["access_token"])
                self.access_token = data["access_token"]
                self.authenticated = True
                
                # Save token to file
                self._save_token_to_file()
                
                logger.info("Authenticated successfully with Zerodha")
                return True
                
        except (TokenException, NetworkException) as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
        except Exception as e:
            logger.exception(f"Unexpected error during authentication: {str(e)}")
            return False
    
    def _save_token_to_file(self):
        """Save access token to file for reuse"""
        try:
            with open(self.token_file, 'w') as f:
                f.write(f"{self.access_token}|{int(time.time())}")
            logger.debug("Zerodha access token saved to file")
            return True
        except Exception as e:
            logger.error(f"Failed to save token to file: {str(e)}")
            return False
    
    def _load_token_from_file(self):
        """Load access token from file if available and not expired"""
        try:
            if not os.path.exists(self.token_file):
                return False
                
            with open(self.token_file, 'r') as f:
                data = f.read().strip().split('|')
                
            if len(data) != 2:
                return False
                
            token, timestamp = data[0], int(data[1])
            current_time = int(time.time())
            
            # Check if token is less than 6 hours old (Zerodha tokens expire in 8 hours)
            if current_time - timestamp > (6 * 60 * 60):
                logger.info("Saved token has expired")
                return False
                
            # Try using the token
            self.access_token = token
            self.kite.set_access_token(token)
            
            # Verify token is valid by making a small request
            profile = self.kite.profile()
            if profile:
                logger.info(f"Authenticated using saved token (user: {profile.get('user_name', 'unknown')})")
                self.authenticated = True
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error loading token from file: {str(e)}")
            return False
    
    def logout(self):
        """Logout from Zerodha"""
        if self.mode == 'paper' or self.mode == 'backtest':
            logger.info(f"Logging out from {self.mode} mode")
            self.authenticated = False
            return True
            
        try:
            if self.authenticated:
                self.kite.invalidate_access_token()
                logger.info("Logged out from Zerodha successfully")
                
                # Remove the token file
                if os.path.exists(self.token_file):
                    os.remove(self.token_file)
                    
                self.authenticated = False
                return True
            return False
        except Exception as e:
            logger.error(f"Error during logout: {str(e)}")
            return False
    
    def get_profile(self):
        """Get user profile"""
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get profile.")
            return None
            
        if self.mode == 'paper' or self.mode == 'backtest':
            return {"user_name": "Paper Trading User", "email": "paper@example.com"}
            
        try:
            self._throttle_request()
            return self.kite.profile()
        except Exception as e:
            logger.error(f"Error getting profile: {str(e)}")
            return None
    
    def get_funds(self):
        """Get account funds"""
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get funds.")
            return None
            
        if self.mode == 'paper':
            return {"balance": self.paper_balance, "used_margin": 0}
            
        try:
            self._throttle_request()
            return self.kite.margins()
        except Exception as e:
            logger.error(f"Error getting funds: {str(e)}")
            return None
    
    def place_order(self, symbol, transaction_type, quantity, order_type='MARKET', 
                   price=None, trigger_price=None, tag=None):
        """
        Place an order
        
        Args:
            symbol (str): Trading symbol (e.g., 'INFY', 'RELIANCE')
            transaction_type (str): 'BUY' or 'SELL'
            quantity (int): Number of shares
            order_type (str): Order type - 'MARKET', 'LIMIT', 'SL', 'SL-M'
            price (float, optional): Order price (for LIMIT orders)
            trigger_price (float, optional): Trigger price (for SL, SL-M orders)
            tag (str, optional): Tag for the order
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot place order.")
            return None
        
        # Normalize inputs
        transaction_type = transaction_type.upper()
        order_type = order_type.upper()
        
        # Validate inputs
        if transaction_type not in ['BUY', 'SELL']:
            logger.error(f"Invalid transaction type: {transaction_type}")
            return None
            
        if order_type not in ['MARKET', 'LIMIT', 'SL', 'SL-M']:
            logger.error(f"Invalid order type: {order_type}")
            return None
            
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}")
            return None
            
        # Handle paper trading
        if self.mode == 'paper':
            # Simulate order placement
            self.last_order_id += 1
            order_id = f"PAPER_{self.last_order_id}"
            
            # Get current market price
            try:
                ltp = self.get_ltp(symbol)
                if ltp is None:
                    logger.error(f"Could not get LTP for {symbol}")
                    return None
                    
                executed_price = ltp
                
                # For limit orders, use the specified price
                if order_type == 'LIMIT' and price:
                    # Check if the order would be executed based on the limit price
                    if (transaction_type == 'BUY' and price < ltp) or \
                       (transaction_type == 'SELL' and price > ltp):
                        logger.info(f"Paper trading: Limit order would not execute immediately. Using limit price.")
                        executed_price = price
                
                # Store paper order
                order_time = datetime.now()
                self.paper_orders[order_id] = {
                    'symbol': symbol,
                    'transaction_type': transaction_type,
                    'quantity': quantity,
                    'order_type': order_type,
                    'price': executed_price,
                    'trigger_price': trigger_price,
                    'tag': tag,
                    'status': 'COMPLETE',
                    'order_time': order_time,
                    'fill_time': order_time
                }
                
                # Update paper positions
                if symbol not in self.paper_positions:
                    self.paper_positions[symbol] = {
                        'quantity': 0,
                        'average_price': 0,
                        'last_price': executed_price
                    }
                
                position = self.paper_positions[symbol]
                
                if transaction_type == 'BUY':
                    # Update average price and quantity for buys
                    total_cost = position['average_price'] * position['quantity']
                    new_quantity = position['quantity'] + quantity
                    total_cost += executed_price * quantity
                    position['average_price'] = total_cost / new_quantity if new_quantity > 0 else 0
                    position['quantity'] = new_quantity
                else:  # SELL
                    # Check if we have enough shares to sell
                    if position['quantity'] < quantity:
                        logger.warning(f"Paper trading: Not enough shares to sell {quantity} of {symbol}. Current position: {position['quantity']}")
                        return None
                        
                    # Update quantity for sells (average price remains the same)
                    position['quantity'] -= quantity
                
                # Update paper balance
                cost = executed_price * quantity
                if transaction_type == 'BUY':
                    self.paper_balance -= cost
                else:
                    self.paper_balance += cost
                
                logger.info(f"Paper trading: {transaction_type} {quantity} {symbol} at {executed_price} - Balance: {self.paper_balance}")
                logger.info(f"Paper trading: Order ID: {order_id}")
                
                return order_id
                
            except Exception as e:
                logger.error(f"Error in paper trading order: {str(e)}")
                return None
        
        # Real trading
        try:
            # Prepare order parameters
            params = {
                "tradingsymbol": symbol,
                "exchange": "NSE",  # Default to NSE, could be parameterized
                "transaction_type": transaction_type,
                "quantity": quantity,
                "product": "CNC",   # CNC for delivery, MIS for intraday
                "order_type": order_type,
                "validity": "DAY",
            }
            
            # Add conditional parameters
            if price and order_type in ['LIMIT', 'SL']:
                params["price"] = price
                
            if trigger_price and order_type in ['SL', 'SL-M']:
                params["trigger_price"] = trigger_price
                
            if tag:
                params["tag"] = tag
            
            # Place the order
            self._throttle_request()
            order_id = self.kite.place_order(**params)
            
            logger.info(f"Order placed: {transaction_type} {quantity} {symbol} - Order ID: {order_id}")
            return order_id
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def modify_order(self, order_id, price=None, quantity=None, 
                    order_type=None, trigger_price=None):
        """
        Modify an existing order
        
        Args:
            order_id (str): Order ID to modify
            price (float, optional): New price
            quantity (int, optional): New quantity
            order_type (str, optional): New order type
            trigger_price (float, optional): New trigger price
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot modify order.")
            return False
            
        # Handle paper trading
        if self.mode == 'paper':
            if order_id not in self.paper_orders:
                logger.error(f"Paper trading: Order {order_id} not found")
                return False
                
            order = self.paper_orders[order_id]
            
            # Check if order can be modified
            if order['status'] in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                logger.error(f"Paper trading: Cannot modify order with status {order['status']}")
                return False
                
            # Update order details
            if price is not None:
                order['price'] = price
            if quantity is not None:
                order['quantity'] = quantity
            if order_type is not None:
                order['order_type'] = order_type
            if trigger_price is not None:
                order['trigger_price'] = trigger_price
                
            logger.info(f"Paper trading: Modified order {order_id}")
            return True
        
        # Real trading
        try:
            params = {}
            
            # Add parameters that need to be modified
            if price is not None:
                params["price"] = price
            if quantity is not None:
                params["quantity"] = quantity
            if order_type is not None:
                params["order_type"] = order_type
            if trigger_price is not None:
                params["trigger_price"] = trigger_price
                
            # No parameters to modify
            if not params:
                logger.warning("No parameters provided for order modification")
                return False
                
            # Modify the order
            self._throttle_request()
            return self.kite.modify_order(order_id=order_id, **params)
            
        except Exception as e:
            logger.error(f"Error modifying order: {str(e)}")
            return False
    
    def cancel_order(self, order_id):
        """
        Cancel an order
        
        Args:
            order_id (str): Order ID to cancel
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot cancel order.")
            return False
            
        # Handle paper trading
        if self.mode == 'paper':
            if order_id not in self.paper_orders:
                logger.error(f"Paper trading: Order {order_id} not found")
                return False
                
            order = self.paper_orders[order_id]
            
            # Check if order can be cancelled
            if order['status'] in ['COMPLETE', 'CANCELLED', 'REJECTED']:
                logger.error(f"Paper trading: Cannot cancel order with status {order['status']}")
                return False
                
            # Update order status
            order['status'] = 'CANCELLED'
            logger.info(f"Paper trading: Cancelled order {order_id}")
            return True
        
        # Real trading
        try:
            self._throttle_request()
            return self.kite.cancel_order(order_id=order_id)
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_order(self, order_id):
        """
        Get order details
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order details if successful, None otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get order.")
            return None
            
        # Handle paper trading
        if self.mode == 'paper':
            if order_id not in self.paper_orders:
                logger.error(f"Paper trading: Order {order_id} not found")
                return None
                
            return self.paper_orders[order_id]
        
        # Real trading
        try:
            self._throttle_request()
            orders = self.kite.orders()
            for order in orders:
                if order['order_id'] == order_id:
                    return order
            return None
        except Exception as e:
            logger.error(f"Error getting order: {str(e)}")
            return None
    
    def get_orders(self):
        """
        Get all orders
        
        Returns:
            list: List of orders if successful, empty list otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get orders.")
            return []
            
        # Handle paper trading
        if self.mode == 'paper':
            return list(self.paper_orders.values())
        
        # Real trading
        try:
            self._throttle_request()
            return self.kite.orders()
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return []
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            dict: Dictionary of positions if successful, empty dict otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get positions.")
            return {}
            
        # Handle paper trading
        if self.mode == 'paper':
            # Update paper positions with current market prices
            for symbol in self.paper_positions:
                try:
                    ltp = self.get_ltp(symbol)
                    if ltp:
                        self.paper_positions[symbol]['last_price'] = ltp
                except Exception as e:
                    logger.warning(f"Could not update paper position price for {symbol}: {str(e)}")
                    
            return self.paper_positions
        
        # Real trading
        try:
            self._throttle_request()
            return self.kite.positions()
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    def get_position(self, symbol):
        """
        Get position for a specific symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            dict: Position details if found, None otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get position.")
            return None
            
        positions = self.get_positions()
        
        # Handle paper trading
        if self.mode == 'paper':
            return positions.get(symbol)
            
        # For real trading, positions are in a different format
        for position in positions.get('net', []):
            if position['tradingsymbol'] == symbol:
                return position
                
        return None
    
    def get_ltp(self, symbol):
        """
        Get last traded price for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            float: Last traded price if successful, None otherwise
        """
        if not self.authenticated and self.mode != 'paper':
            logger.error("Not authenticated. Cannot get LTP.")
            return None
            
        try:
            # Use quote API to get LTP
            self._throttle_request()
            quote = self.kite.quote(f"NSE:{symbol}")
            if quote and f"NSE:{symbol}" in quote:
                return quote[f"NSE:{symbol}"]["last_price"]
            return None
        except Exception as e:
            logger.error(f"Error getting LTP for {symbol}: {str(e)}")
            return None
    
    def get_historical_data(self, symbol, from_date, to_date, interval):
        """
        Get historical OHLC data for a symbol
        
        Args:
            symbol (str): Trading symbol
            from_date (datetime): Start date
            to_date (datetime): End date
            interval (str): Candle interval (minute, day, 5minute, etc.)
            
        Returns:
            list: List of OHLC data if successful, empty list otherwise
        """
        if not self.authenticated:
            logger.error("Not authenticated. Cannot get historical data.")
            return []
            
        try:
            self._throttle_request()
            return self.kite.historical_data(
                instrument_token=self._get_instrument_token(symbol),
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {str(e)}")
            return []
    
    def _get_instrument_token(self, symbol):
        """
        Get instrument token for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            int: Instrument token if found, None otherwise
        """
        try:
            self._throttle_request()
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument['instrument_token']
            return None
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {str(e)}")
            return None
    
    def place_stoploss_order(self, symbol, transaction_type, quantity, trigger_price, price=None):
        """
        Place a stop loss order
        
        Args:
            symbol (str): Trading symbol
            transaction_type (str): 'BUY' or 'SELL'
            quantity (int): Number of shares
            trigger_price (float): Trigger price
            price (float, optional): Limit price (for SL orders)
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        # If price is provided, use SL order type, otherwise use SL-M
        order_type = "SL" if price else "SL-M"
        
        return self.place_order(
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            price=price,
            trigger_price=trigger_price,
            tag="STOPLOSS"
        )
    
    def place_target_order(self, symbol, transaction_type, quantity, price):
        """
        Place a limit order for profit booking
        
        Args:
            symbol (str): Trading symbol
            transaction_type (str): 'BUY' or 'SELL'
            quantity (int): Number of shares
            price (float): Limit price
            
        Returns:
            str: Order ID if successful, None otherwise
        """
        return self.place_order(
            symbol=symbol,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type="LIMIT",
            price=price,
            tag="TARGET"
        )