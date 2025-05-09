"""
Order management for Indian stock markets.
Handles both live trading and paper trading.
"""

import os
import json
import logging
import requests
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta

from ..data.data_fetcher import DataFetcher
from .costs import TradingCosts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("OrderManager")


class OrderManager:
    """Class for managing orders and positions"""
    
    def __init__(self, api_key: str = '', api_secret: str = '', broker: str = 'zerodha', 
                 is_paper_trading: bool = True):
        """Initialize with API credentials and trading mode"""
        self.api_key = api_key or os.getenv('API_KEY', '')
        self.api_secret = api_secret or os.getenv('API_SECRET', '')
        self.broker = broker.lower()
        self.is_paper_trading = is_paper_trading
        self.access_token = None
        
        # Paper trading variables
        self.paper_positions = {}  # symbol -> {qty, avg_price, pnl}
        self.paper_orders = {}  # order_id -> {symbol, qty, price, type, status, timestamp}
        self.paper_balance = float(os.getenv('INITIAL_BALANCE', '100000'))
        self.paper_order_id = 1000
        
        # Connection status
        self.is_connected = False
        
        # Get the base URLs based on broker
        if self.broker == 'zerodha':
            self.api_url = "https://api.kite.trade"
            self.login_url = "https://kite.zerodha.com/api/login"
        elif self.broker == 'upstox':
            self.api_url = "https://api.upstox.com/v2"
            self.login_url = "https://api.upstox.com/v2/login"
        else:
            raise ValueError(f"Unsupported broker: {broker}")
        
        # Initialize data fetcher for market data
        self.data_fetcher = DataFetcher(api_key=self.api_key, api_secret=self.api_secret, broker=self.broker)
        
        # Authenticate if not paper trading
        if not self.is_paper_trading:
            self.authenticate()
            
        # Create directories for logging
        os.makedirs('data/trades', exist_ok=True)
        os.makedirs('logs/orders', exist_ok=True)
            
        logger.info(f"OrderManager initialized in {'paper' if is_paper_trading else 'live'} trading mode")
    
    def authenticate(self) -> bool:
        """Authenticate with the broker API"""
        # Use DataFetcher's authentication
        result = self.data_fetcher.authenticate()
        if result:
            self.access_token = self.data_fetcher.access_token
            self.is_connected = True
        return result
    
    def place_order(self, symbol: str, quantity: int, price: float = 0, 
                   order_type: str = 'MARKET', transaction_type: str = 'BUY') -> Dict:
        """Place an order with the broker or in paper trading mode"""
        logger.info(f"Placing {order_type} {transaction_type} order for {quantity} {symbol} @ {price}")
        
        try:
            # Validate inputs
            if quantity <= 0:
                return {'status': 'failed', 'message': 'Quantity must be positive'}
                
            if order_type not in ['MARKET', 'LIMIT', 'SL']:
                return {'status': 'failed', 'message': f'Unsupported order type: {order_type}'}
                
            if transaction_type not in ['BUY', 'SELL']:
                return {'status': 'failed', 'message': f'Unsupported transaction type: {transaction_type}'}
                
            if order_type != 'MARKET' and price <= 0:
                return {'status': 'failed', 'message': 'Price must be specified for non-market orders'}
            
            # Execute the order
            if self.is_paper_trading:
                return self._place_paper_order(symbol, quantity, price, order_type, transaction_type)
            else:
                if not self.is_connected and not self.authenticate():
                    logger.error("Not authenticated. Please authenticate first.")
                    return {'status': 'failed', 'message': 'Not authenticated'}
                    
                try:
                    if self.broker == 'zerodha':
                        return self._place_order_zerodha(symbol, quantity, price, order_type, transaction_type)
                    elif self.broker == 'upstox':
                        return self._place_order_upstox(symbol, quantity, price, order_type, transaction_type)
                    
                    return {'status': 'failed', 'message': 'Unsupported broker'}
                except Exception as e:
                    logger.error(f"Order placement error: {str(e)}")
                    return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in place_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _place_paper_order(self, symbol: str, quantity: int, price: float = 0, 
                          order_type: str = 'MARKET', transaction_type: str = 'BUY') -> Dict:
        """Place a paper trading order"""
        try:
            # Get the current market price
            if price == 0 or order_type == 'MARKET':
                market_data = self.data_fetcher.get_last_data(symbol)
                if not market_data:
                    logger.error("No market data available")
                    return {'status': 'failed', 'message': 'No market data available'}
                
                price = market_data.get('last_price', 0)
                if price == 0:
                    logger.error("Invalid price")
                    return {'status': 'failed', 'message': 'Invalid price'}
            
            # Check if we have enough balance for BUY
            if transaction_type.upper() == 'BUY':
                # Calculate costs
                costs = TradingCosts.calculate_costs(price, quantity, 'buy')
                total_cost = (price * quantity) + costs['total']
                
                if total_cost > self.paper_balance:
                    logger.warning(f"Insufficient balance: {self.paper_balance} < {total_cost}")
                    return {'status': 'failed', 'message': 'Insufficient balance'}
                    
                # Deduct from balance
                self.paper_balance -= total_cost
                
                # Update position
                if symbol in self.paper_positions:
                    # Average out the price
                    current_qty = self.paper_positions[symbol]['qty']
                    current_avg = self.paper_positions[symbol]['avg_price']
                    
                    new_qty = current_qty + quantity
                    new_avg = ((current_qty * current_avg) + (quantity * price)) / new_qty
                    
                    self.paper_positions[symbol] = {
                        'qty': new_qty,
                        'avg_price': new_avg,
                        'pnl': 0,  # Will be updated in monitor loop
                        'last_update': datetime.now()
                    }
                else:
                    self.paper_positions[symbol] = {
                        'qty': quantity,
                        'avg_price': price,
                        'pnl': 0,
                        'last_update': datetime.now()
                    }
            else:  # SELL
                # Check if we have the position
                if symbol not in self.paper_positions or self.paper_positions[symbol]['qty'] < quantity:
                    logger.warning(f"Insufficient position: {symbol} {self.paper_positions.get(symbol, {}).get('qty', 0)} < {quantity}")
                    return {'status': 'failed', 'message': 'Insufficient position'}
                    
                # Calculate costs
                costs = TradingCosts.calculate_costs(price, quantity, 'sell')
                
                # Calculate P&L
                avg_price = self.paper_positions[symbol]['avg_price']
                pnl = ((price - avg_price) * quantity) - costs['total']
                
                # Update balance and position
                self.paper_balance += (price * quantity) - costs['total']
                
                # Update position
                current_qty = self.paper_positions[symbol]['qty']
                new_qty = current_qty - quantity
                
                if new_qty > 0:
                    self.paper_positions[symbol]['qty'] = new_qty
                    self.paper_positions[symbol]['last_update'] = datetime.now()
                    # Keep the same average price
                else:
                    # Position closed
                    del self.paper_positions[symbol]
            
            # Create order record
            order_id = self.paper_order_id
            self.paper_order_id += 1
            
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'qty': quantity,
                'price': price,
                'type': order_type,
                'transaction_type': transaction_type,
                'status': 'COMPLETE',
                'timestamp': datetime.now(),
                'costs': costs['total'] if 'costs' in locals() else 0,
                'exchange': 'NSE',  # Assuming NSE
                'product': 'MIS',   # Intraday
            }
            
            self.paper_orders[order_id] = order
            
            # Log order to file
            self._log_paper_order(order)
            
            logger.info(f"Paper order placed: {order_id} {transaction_type} {quantity} {symbol} @ {price}")
            
            return {
                'status': 'success',
                'order_id': order_id,
                'message': 'Order placed successfully',
                'details': order
            }
        except Exception as e:
            logger.error(f"Error in _place_paper_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _log_paper_order(self, order: Dict) -> None:
        """Log paper order to file for record keeping"""
        try:
            # Create a log file for the day
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = f"logs/orders/paper_orders_{today}.csv"
            
            # Convert order to DataFrame row
            order_copy = order.copy()
            if isinstance(order_copy['timestamp'], datetime):
                order_copy['timestamp'] = order_copy['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            order_df = pd.DataFrame([order_copy])
            
            # Append to log file if exists, otherwise create new
            if os.path.exists(log_file):
                order_df.to_csv(log_file, mode='a', header=False, index=False)
            else:
                order_df.to_csv(log_file, index=False)
                
            # Also log position after order
            position_file = f"logs/orders/paper_positions_{today}.csv"
            position_data = []
            
            for sym, pos in self.paper_positions.items():
                position_data.append({
                    'symbol': sym,
                    'quantity': pos['qty'],
                    'avg_price': pos['avg_price'],
                    'current_value': pos['qty'] * pos['avg_price'],
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
                
            # Add cash balance
            position_data.append({
                'symbol': 'CASH',
                'quantity': 1,
                'avg_price': self.paper_balance,
                'current_value': self.paper_balance,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # Save position snapshot
            position_df = pd.DataFrame(position_data)
            
            if os.path.exists(position_file):
                position_df.to_csv(position_file, mode='a', header=False, index=False)
            else:
                position_df.to_csv(position_file, index=False)
            
        except Exception as e:
            logger.error(f"Error logging paper order: {str(e)}")
    
    def _place_order_zerodha(self, symbol: str, quantity: int, price: float = 0, 
                            order_type: str = 'MARKET', transaction_type: str = 'BUY') -> Dict:
        """Place an order with Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            order_url = f"{self.api_url}/orders/regular"
            
            # Convert order type to Zerodha format
            zerodha_order_type = order_type
            trigger_price = 0
            
            if order_type == 'MARKET':
                zerodha_order_type = 'MARKET'
                price = 0
                trigger_price = 0
            elif order_type == 'LIMIT':
                zerodha_order_type = 'LIMIT'
                trigger_price = 0
            elif order_type == 'SL':
                zerodha_order_type = 'SL'
                if price == 0:
                    return {'status': 'failed', 'message': 'Price is required for SL order'}
                trigger_price = price
            else:
                return {'status': 'failed', 'message': f'Unsupported order type: {order_type}'}
            
            # Prepare order data
            order_data = {
                'tradingsymbol': symbol,
                'exchange': 'NSE',  # Assuming NSE, change as needed
                'quantity': quantity,
                'transaction_type': transaction_type,
                'order_type': zerodha_order_type,
                'product': 'MIS',  # Day trading, change as needed
                'validity': 'DAY',
            }
            
            if price > 0:
                order_data['price'] = price
                
            if trigger_price > 0:
                order_data['trigger_price'] = trigger_price
            
            # Place the order
            response = requests.post(order_url, data=order_data, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            # Parse response
            response_data = response.json()
            order_id = response_data.get('data', {}).get('order_id', '')
            
            # Log the order
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'qty': quantity,
                'price': price,
                'type': order_type,
                'transaction_type': transaction_type,
                'status': 'PLACED',
                'timestamp': datetime.now(),
                'response': response_data
            }
            
            self._log_live_order(order)
            
            logger.info(f"Zerodha order placed: {order_id} {transaction_type} {quantity} {symbol} @ {price}")
            
            return {
                'status': 'success',
                'order_id': order_id,
                'message': 'Order placed successfully',
                'details': response_data.get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _place_order_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _place_order_upstox(self, symbol: str, quantity: int, price: float = 0, 
                           order_type: str = 'MARKET', transaction_type: str = 'BUY') -> Dict:
        """Place an order with Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            order_url = f"{self.api_url}/order/place"
            
            # Convert order type to Upstox format
            upstox_order_type = order_type
            trigger_price = 0
            
            if order_type == 'MARKET':
                upstox_order_type = 'MARKET'
                price = 0
                trigger_price = 0
            elif order_type == 'LIMIT':
                upstox_order_type = 'LIMIT'
                trigger_price = 0
            elif order_type == 'SL':
                upstox_order_type = 'SL'
                if price == 0:
                    return {'status': 'failed', 'message': 'Price is required for SL order'}
                trigger_price = price
            else:
                return {'status': 'failed', 'message': f'Unsupported order type: {order_type}'}
            
            # Get instrument key
            instruments_url = f"{self.api_url}/instruments"
            
            response = requests.get(instruments_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': f"Failed to fetch instruments: {response.text}"}
                
            instruments = response.json().get('data', [])
            instrument_key = None
            
            for instrument in instruments:
                if instrument.get('tradingsymbol') == symbol:
                    instrument_key = instrument.get('instrument_key')
                    break
            
            if not instrument_key:
                return {'status': 'failed', 'message': f"Instrument {symbol} not found"}
            
            # Prepare order data
            order_data = {
                'instrument_key': instrument_key,
                'quantity': quantity,
                'transaction_type': transaction_type,
                'order_type': upstox_order_type,
                'product': 'I',  # Intraday, change as needed
                'validity': 'DAY',
            }
            
            if price > 0:
                order_data['price'] = price
                
            if trigger_price > 0:
                order_data['trigger_price'] = trigger_price
            
            # Place the order
            response = requests.post(order_url, json=order_data, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            # Parse response
            response_data = response.json()
            order_id = response_data.get('data', {}).get('order_id', '')
            
            # Log the order
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'qty': quantity,
                'price': price,
                'type': order_type,
                'transaction_type': transaction_type,
                'status': 'PLACED',
                'timestamp': datetime.now(),
                'response': response_data
            }
            
            self._log_live_order(order)
            
            logger.info(f"Upstox order placed: {order_id} {transaction_type} {quantity} {symbol} @ {price}")
            
            return {
                'status': 'success',
                'order_id': order_id,
                'message': 'Order placed successfully',
                'details': response_data.get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _place_order_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _log_live_order(self, order: Dict) -> None:
        """Log live order to file for record keeping"""
        try:
            # Create a log file for the day
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = f"logs/orders/live_orders_{today}.json"
            
            # Prepare data
            order_data = {
                'order_id': order['order_id'],
                'symbol': order['symbol'],
                'qty': order['qty'],
                'price': order['price'],
                'type': order['type'],
                'transaction_type': order['transaction_type'],
                'status': order['status'],
                'timestamp': order['timestamp'].isoformat() if isinstance(order['timestamp'], datetime) else order['timestamp'],
            }
            
            # Append to log file if exists, otherwise create new
            if os.path.exists(log_file):
                with open(log_file, 'r+') as f:
                    try:
                        orders = json.load(f)
                    except json.JSONDecodeError:
                        orders = []
                    
                    orders.append(order_data)
                    
                    f.seek(0)
                    json.dump(orders, f, indent=2)
            else:
                with open(log_file, 'w') as f:
                    json.dump([order_data], f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error logging live order: {str(e)}")
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        try:
            if self.is_paper_trading:
                # Update positions with current market prices
                self.update_paper_positions()
                
                # Calculate total position value
                total_position_value = sum(
                    pos.get('qty', 0) * pos.get('current_price', pos.get('avg_price', 0))
                    for pos in self.paper_positions.values()
                )
                
                # Add up total unrealized P&L
                total_pnl = sum(pos.get('pnl', 0) for pos in self.paper_positions.values())
                
                return {
                    'positions': self.paper_positions,
                    'balance': self.paper_balance,
                    'total_value': self.paper_balance + total_position_value,
                    'unrealized_pnl': total_pnl
                }
                
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._get_positions_zerodha()
                elif self.broker == 'upstox':
                    return self._get_positions_upstox()
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error fetching positions: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in get_positions: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_positions_zerodha(self) -> Dict:
        """Get positions from Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            positions_url = f"{self.api_url}/portfolio/positions"
            
            response = requests.get(positions_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            # Get account balance
            margins_url = f"{self.api_url}/user/margins"
            margins_response = requests.get(margins_url, headers=headers)
            
            if margins_response.status_code == 200:
                margins_data = margins_response.json().get('data', {})
                equity_margins = margins_data.get('equity', {})
                available_balance = equity_margins.get('available', {}).get('cash', 0)
            else:
                available_balance = 0
                
            # Calculate total position value and P&L
            positions_data = response.json().get('data', {})
            day_positions = positions_data.get('day', [])
            net_positions = positions_data.get('net', [])
            
            # Combine day and net positions
            all_positions = {}
            
            for position in day_positions + net_positions:
                symbol = position.get('tradingsymbol', '')
                
                if symbol in all_positions:
                    continue  # Skip if already processed
                    
                all_positions[symbol] = {
                    'qty': position.get('quantity', 0),
                    'avg_price': position.get('average_price', 0),
                    'current_price': position.get('last_price', 0),
                    'pnl': position.get('pnl', 0),
                    'product': position.get('product', ''),
                    'exchange': position.get('exchange', ''),
                }
            
            # Calculate total values
            total_position_value = sum(
                pos.get('qty', 0) * pos.get('current_price', 0)
                for pos in all_positions.values()
            )
            
            total_pnl = sum(pos.get('pnl', 0) for pos in all_positions.values())
            
            return {
                'status': 'success',
                'positions': all_positions,
                'balance': available_balance,
                'total_value': available_balance + total_position_value,
                'unrealized_pnl': total_pnl
            }
        except Exception as e:
            logger.error(f"Error in _get_positions_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_positions_upstox(self) -> Dict:
        """Get positions from Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            positions_url = f"{self.api_url}/portfolio/positions"
            
            response = requests.get(positions_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            # Get account balance
            funds_url = f"{self.api_url}/user/funds-n-margins"
            funds_response = requests.get(funds_url, headers=headers)
            
            if funds_response.status_code == 200:
                funds_data = funds_response.json().get('data', {})
                available_balance = funds_data.get('equity', {}).get('available_balance', 0)
            else:
                available_balance = 0
                
            # Process positions
            positions_data = response.json().get('data', [])
            all_positions = {}
            
            for position in positions_data:
                symbol = position.get('tradingsymbol', '')
                
                all_positions[symbol] = {
                    'qty': position.get('quantity', 0),
                    'avg_price': position.get('average_price', 0),
                    'current_price': position.get('last_price', 0),
                    'pnl': position.get('pnl', 0),
                    'product': position.get('product', ''),
                    'exchange': position.get('exchange', ''),
                }
            
            # Calculate total values
            total_position_value = sum(
                pos.get('qty', 0) * pos.get('current_price', 0)
                for pos in all_positions.values()
            )
            
            total_pnl = sum(pos.get('pnl', 0) for pos in all_positions.values())
            
            return {
                'status': 'success',
                'positions': all_positions,
                'balance': available_balance,
                'total_value': available_balance + total_position_value,
                'unrealized_pnl': total_pnl
            }
        except Exception as e:
            logger.error(f"Error in _get_positions_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def get_order_history(self, order_id: str = None) -> Dict:
        """Get order history"""
        try:
            if self.is_paper_trading:
                if order_id:
                    if order_id in self.paper_orders:
                        return {'status': 'success', 'orders': [self.paper_orders[order_id]]}
                    else:
                        return {'status': 'failed', 'message': 'Order not found'}
                else:
                    return {'status': 'success', 'orders': list(self.paper_orders.values())}
                    
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._get_order_history_zerodha(order_id)
                elif self.broker == 'upstox':
                    return self._get_order_history_upstox(order_id)
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error fetching order history: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in get_order_history: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_order_history_zerodha(self, order_id: str = None) -> Dict:
        """Get order history from Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            
            if order_id:
                order_url = f"{self.api_url}/orders/{order_id}"
                response = requests.get(order_url, headers=headers)
            else:
                orders_url = f"{self.api_url}/orders"
                response = requests.get(orders_url, headers=headers)
                
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            orders_data = response.json().get('data', [])
            
            # Process orders
            processed_orders = []
            
            for order in orders_data:
                processed_orders.append({
                    'order_id': order.get('order_id', ''),
                    'symbol': order.get('tradingsymbol', ''),
                    'qty': order.get('quantity', 0),
                    'price': order.get('price', 0),
                    'avg_price': order.get('average_price', 0),
                    'type': order.get('order_type', ''),
                    'transaction_type': order.get('transaction_type', ''),
                    'status': order.get('status', ''),
                    'timestamp': order.get('order_timestamp', ''),
                    'exchange': order.get('exchange', ''),
                    'product': order.get('product', '')
                })
            
            return {'status': 'success', 'orders': processed_orders}
        except Exception as e:
            logger.error(f"Error in _get_order_history_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_order_history_upstox(self, order_id: str = None) -> Dict:
        """Get order history from Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            
            if order_id:
                order_url = f"{self.api_url}/order/{order_id}"
                response = requests.get(order_url, headers=headers)
            else:
                orders_url = f"{self.api_url}/order/history"
                response = requests.get(orders_url, headers=headers)
                
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            if order_id:
                # Single order response
                order_data = response.json().get('data', {})
                orders_data = [order_data] if order_data else []
            else:
                # List of orders
                orders_data = response.json().get('data', [])
            
            # Process orders
            processed_orders = []
            
            for order in orders_data:
                processed_order = {
                    'order_id': order.get('order_id', ''),
                    'symbol': order.get('tradingsymbol', ''),
                    'qty': order.get('quantity', 0),
                    'price': order.get('price', 0),
                    'avg_price': order.get('average_price', 0),
                    'type': order.get('order_type', ''),
                    'transaction_type': order.get('transaction_type', ''),
                    'status': order.get('status', ''),
                    'exchange': order.get('exchange', ''),
                    'product': order.get('product', '')
                }
                
                # Handle timestamp field which might have different names
                if 'order_timestamp' in order:
                    processed_order['timestamp'] = order['order_timestamp']
                elif 'order_creation_time' in order:
                    processed_order['timestamp'] = order['order_creation_time']
                else:
                    processed_order['timestamp'] = ''
                
                processed_orders.append(processed_order)
            
            return {'status': 'success', 'orders': processed_orders}
        except Exception as e:
            logger.error(f"Error in _get_order_history_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def update_paper_positions(self) -> None:
        """Update paper trading positions with latest market data"""
        try:
            for symbol, position in list(self.paper_positions.items()):
                market_data = self.data_fetcher.get_last_data(symbol)
                if market_data and 'last_price' in market_data:
                    current_price = market_data['last_price']
                    avg_price = position['avg_price']
                    qty = position['qty']
                    
                    # Calculate P&L
                    position['pnl'] = (current_price - avg_price) * qty
                    
                    # Add current price to position data
                    position['current_price'] = current_price
                    position['last_update'] = datetime.now()
                    
                    # Calculate additional metrics
                    position['value'] = current_price * qty
                    position['change_pct'] = ((current_price / avg_price) - 1) * 100
        except Exception as e:
            logger.error(f"Error updating paper positions: {str(e)}")
    
    def modify_order(self, order_id: str, price: float = None, quantity: int = None, 
                    order_type: str = None) -> Dict:
        """Modify an existing order"""
        try:
            if self.is_paper_trading:
                return self._modify_paper_order(order_id, price, quantity, order_type)
                
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._modify_order_zerodha(order_id, price, quantity, order_type)
                elif self.broker == 'upstox':
                    return self._modify_order_upstox(order_id, price, quantity, order_type)
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error modifying order: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in modify_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _modify_paper_order(self, order_id: str, price: float = None, quantity: int = None, 
                           order_type: str = None) -> Dict:
        """Modify a paper trading order"""
        try:
            # Check if order exists
            if not str(order_id) in self.paper_orders:
                return {'status': 'failed', 'message': 'Order not found'}
                
            # Get the order
            order = self.paper_orders[str(order_id)]
            
            # Check if order is still open
            if order['status'] != 'OPEN':
                return {'status': 'failed', 'message': f"Cannot modify order with status {order['status']}"}
                
            # Update order parameters
            if price is not None:
                order['price'] = price
                
            if quantity is not None:
                order['qty'] = quantity
                
            if order_type is not None:
                order['type'] = order_type
                
            # Update timestamp
            order['timestamp'] = datetime.now()
            
            logger.info(f"Paper order modified: {order_id}")
            return {'status': 'success', 'message': 'Order modified successfully', 'details': order}
        except Exception as e:
            logger.error(f"Error in _modify_paper_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _modify_order_zerodha(self, order_id: str, price: float = None, 
                             quantity: int = None, order_type: str = None) -> Dict:
        """Modify an order with Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            modify_url = f"{self.api_url}/orders/{order_id}"
            
            # Prepare modify data
            modify_data = {}
            
            if price is not None:
                modify_data['price'] = price
                
            if quantity is not None:
                modify_data['quantity'] = quantity
                
            if order_type is not None:
                modify_data['order_type'] = order_type
                
            # Modify the order
            response = requests.put(modify_url, data=modify_data, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            logger.info(f"Zerodha order modified: {order_id}")
            return {
                'status': 'success',
                'message': 'Order modified successfully',
                'details': response.json().get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _modify_order_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _modify_order_upstox(self, order_id: str, price: float = None, 
                            quantity: int = None, order_type: str = None) -> Dict:
        """Modify an order with Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            modify_url = f"{self.api_url}/order/modify"
            
            # Prepare modify data
            modify_data = {
                'order_id': order_id
            }
            
            if price is not None:
                modify_data['price'] = price
                
            if quantity is not None:
                modify_data['quantity'] = quantity
                
            if order_type is not None:
                modify_data['order_type'] = order_type
                
            # Modify the order
            response = requests.put(modify_url, json=modify_data, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            logger.info(f"Upstox order modified: {order_id}")
            return {
                'status': 'success',
                'message': 'Order modified successfully',
                'details': response.json().get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _modify_order_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an existing order"""
        try:
            if self.is_paper_trading:
                return self._cancel_paper_order(order_id)
                
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._cancel_order_zerodha(order_id)
                elif self.broker == 'upstox':
                    return self._cancel_order_upstox(order_id)
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error cancelling order: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in cancel_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _cancel_paper_order(self, order_id: str) -> Dict:
        """Cancel a paper trading order"""
        try:
            # Check if order exists
            if not str(order_id) in self.paper_orders:
                return {'status': 'failed', 'message': 'Order not found'}
                
            # Get the order
            order = self.paper_orders[str(order_id)]
            
            # Check if order is still open
            if order['status'] != 'OPEN':
                return {'status': 'failed', 'message': f"Cannot cancel order with status {order['status']}"}
                
            # Update order status
            order['status'] = 'CANCELLED'
            order['timestamp'] = datetime.now()
            
            logger.info(f"Paper order cancelled: {order_id}")
            return {'status': 'success', 'message': 'Order cancelled successfully', 'details': order}
        except Exception as e:
            logger.error(f"Error in _cancel_paper_order: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _cancel_order_zerodha(self, order_id: str) -> Dict:
        """Cancel an order with Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            cancel_url = f"{self.api_url}/orders/{order_id}"
            
            # Cancel the order
            response = requests.delete(cancel_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            logger.info(f"Zerodha order cancelled: {order_id}")
            return {
                'status': 'success',
                'message': 'Order cancelled successfully',
                'details': response.json().get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _cancel_order_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _cancel_order_upstox(self, order_id: str) -> Dict:
        """Cancel an order with Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            cancel_url = f"{self.api_url}/order/cancel/{order_id}"
            
            # Cancel the order
            response = requests.delete(cancel_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            logger.info(f"Upstox order cancelled: {order_id}")
            return {
                'status': 'success',
                'message': 'Order cancelled successfully',
                'details': response.json().get('data', {})
            }
        except Exception as e:
            logger.error(f"Error in _cancel_order_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def get_day_trades(self) -> Dict:
        """Get all trades executed today"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            if self.is_paper_trading:
                # Filter paper orders for today
                today_orders = [
                    order for order in self.paper_orders.values()
                    if isinstance(order['timestamp'], datetime) and 
                    order['timestamp'].strftime('%Y-%m-%d') == today
                ]
                
                return {'status': 'success', 'trades': today_orders}
                
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._get_day_trades_zerodha()
                elif self.broker == 'upstox':
                    return self._get_day_trades_upstox()
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error fetching day trades: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in get_day_trades: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_day_trades_zerodha(self) -> Dict:
        """Get day trades from Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            trades_url = f"{self.api_url}/trades"
            
            # Get all trades
            response = requests.get(trades_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            trades_data = response.json().get('data', [])
            
            # Process trades
            processed_trades = []
            
            for trade in trades_data:
                processed_trades.append({
                    'trade_id': trade.get('trade_id', ''),
                    'order_id': trade.get('order_id', ''),
                    'symbol': trade.get('tradingsymbol', ''),
                    'qty': trade.get('quantity', 0),
                    'price': trade.get('average_price', 0),
                    'transaction_type': trade.get('transaction_type', ''),
                    'timestamp': trade.get('fill_timestamp', ''),
                    'exchange': trade.get('exchange', ''),
                    'product': trade.get('product', '')
                })
            
            # Save trades to file
            self._save_day_trades_to_file(processed_trades)
            
            return {'status': 'success', 'trades': processed_trades}
        except Exception as e:
            logger.error(f"Error in _get_day_trades_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_day_trades_upstox(self) -> Dict:
        """Get day trades from Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            trades_url = f"{self.api_url}/trade/history"
            
            # Get all trades
            response = requests.get(trades_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            trades_data = response.json().get('data', [])
            
            # Process trades
            processed_trades = []
            
            for trade in trades_data:
                processed_trade = {
                    'trade_id': trade.get('trade_id', ''),
                    'order_id': trade.get('order_id', ''),
                    'symbol': trade.get('tradingsymbol', ''),
                    'qty': trade.get('quantity', 0),
                    'price': trade.get('price', 0),
                    'transaction_type': trade.get('transaction_type', ''),
                    'exchange': trade.get('exchange', ''),
                    'product': trade.get('product', '')
                }
                
                # Handle timestamp - Upstox may use a different field name
                if 'fill_timestamp' in trade:
                    processed_trade['timestamp'] = trade['fill_timestamp']
                elif 'execution_time' in trade:
                    processed_trade['timestamp'] = trade['execution_time']
                else:
                    processed_trade['timestamp'] = ''
                
                processed_trades.append(processed_trade)
            
            # Save trades to file
            self._save_day_trades_to_file(processed_trades)
            
            return {'status': 'success', 'trades': processed_trades}
        except Exception as e:
            logger.error(f"Error in _get_day_trades_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _save_day_trades_to_file(self, trades: List[Dict]) -> None:
        """Save day trades to a file"""
        try:
            if not trades:
                return
                
            # Create a trades directory if it doesn't exist
            os.makedirs('data/trades', exist_ok=True)
            
            # Save to CSV
            today = datetime.now().strftime('%Y-%m-%d')
            file_path = f"data/trades/trades_{today}.csv"
            
            # Prepare trades data for saving
            for trade in trades:
                if isinstance(trade.get('timestamp'), datetime):
                    trade['timestamp'] = trade['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            trades_df = pd.DataFrame(trades)
            
            if os.path.exists(file_path):
                trades_df.to_csv(file_path, mode='a', header=False, index=False)
            else:
                trades_df.to_csv(file_path, index=False)
            
            logger.info(f"Saved {len(trades)} trades to {file_path}")
        except Exception as e:
            logger.error(f"Error saving day trades to file: {str(e)}")
    
    def get_account_balance(self) -> Dict:
        """Get account balance"""
        try:
            if self.is_paper_trading:
                return {
                    'status': 'success',
                    'balance': self.paper_balance,
                    'equity': self.paper_balance + sum(
                        pos.get('qty', 0) * pos.get('current_price', pos.get('avg_price', 0))
                        for pos in self.paper_positions.values()
                    ),
                    'used_margin': sum(
                        pos.get('qty', 0) * pos.get('avg_price', 0) * 0.5  # Assuming 50% margin
                        for pos in self.paper_positions.values()
                    )
                }
                
            if not self.is_connected and not self.authenticate():
                logger.error("Not authenticated. Please authenticate first.")
                return {'status': 'failed', 'message': 'Not authenticated'}
                
            try:
                if self.broker == 'zerodha':
                    return self._get_account_balance_zerodha()
                elif self.broker == 'upstox':
                    return self._get_account_balance_upstox()
                    
                return {'status': 'failed', 'message': 'Unsupported broker'}
            except Exception as e:
                logger.error(f"Error fetching account balance: {str(e)}")
                return {'status': 'failed', 'message': str(e)}
        except Exception as e:
            logger.error(f"Error in get_account_balance: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_account_balance_zerodha(self) -> Dict:
        """Get account balance from Zerodha"""
        try:
            headers = {'Authorization': f"token {self.api_key}:{self.access_token}"}
            margins_url = f"{self.api_url}/user/margins"
            
            response = requests.get(margins_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            margins_data = response.json().get('data', {})
            equity_margins = margins_data.get('equity', {})
            
            return {
                'status': 'success',
                'balance': equity_margins.get('available', {}).get('cash', 0),
                'equity': equity_margins.get('net', 0),
                'used_margin': equity_margins.get('utilised', {}).get('exposure', 0) + equity_margins.get('utilised', {}).get('span', 0),
                'available_margin': equity_margins.get('available', {}).get('live_balance', 0)
            }
        except Exception as e:
            logger.error(f"Error in _get_account_balance_zerodha: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def _get_account_balance_upstox(self) -> Dict:
        """Get account balance from Upstox"""
        try:
            headers = {'Authorization': f"Bearer {self.access_token}"}
            funds_url = f"{self.api_url}/user/funds-n-margins"
            
            response = requests.get(funds_url, headers=headers)
            if response.status_code != 200:
                return {'status': 'failed', 'message': response.text}
                
            funds_data = response.json().get('data', {})
            equity_funds = funds_data.get('equity', {})
            
            return {
                'status': 'success',
                'balance': equity_funds.get('available_balance', 0),
                'equity': equity_funds.get('net_value', 0),
                'used_margin': equity_funds.get('used_margin', 0),
                'available_margin': equity_funds.get('available_margin', 0)
            }
        except Exception as e:
            logger.error(f"Error in _get_account_balance_upstox: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
    
    def save_portfolio_snapshot(self) -> None:
        """Save current portfolio snapshot to file"""
        try:
            # Get positions
            positions = self.get_positions()
            
            # Get account balance
            balance = self.get_account_balance()
            
            # Create timestamp
            timestamp = datetime.now()
            
            # Create portfolio snapshot
            snapshot = {
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'balance': balance.get('balance', 0),
                'equity': balance.get('equity', 0),
                'used_margin': balance.get('used_margin', 0),
                'positions': positions.get('positions', {})
            }
            
            # Save to file
            os.makedirs('data/snapshots', exist_ok=True)
            file_path = f"data/snapshots/portfolio_{timestamp.strftime('%Y-%m-%d')}.json"
            
            # Check if file exists
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = {'snapshots': []}
            else:
                data = {'snapshots': []}
                
            # Append snapshot
            data['snapshots'].append(snapshot)
            
            # Save file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Saved portfolio snapshot to {file_path}")
        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {str(e)}")
    
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        try:
            # Get positions
            positions_data = self.get_positions()
            
            if 'positions' not in positions_data:
                return {'status': 'success', 'message': 'No positions to close'}
                
            positions = positions_data.get('positions', {})
            
            # Track results
            results = []
            
            # Close each position
            for symbol, position in positions.items():
                # Skip positions with zero quantity
                if position.get('qty', 0) == 0:
                    continue
                    
                # Determine transaction type
                qty = position.get('qty', 0)
                
                if qty > 0:
                    # Long position, need to sell
                    result = self.place_order(
                        symbol,
                        abs(qty),
                        order_type='MARKET',
                        transaction_type='SELL'
                    )
                elif qty < 0:
                    # Short position, need to buy
                    result = self.place_order(
                        symbol,
                        abs(qty),
                        order_type='MARKET',
                        transaction_type='BUY'
                    )
                else:
                    # No position
                    continue
                    
                results.append({
                    'symbol': symbol,
                    'qty': qty,
                    'result': result
                })
                
            logger.info(f"Closed {len(results)} positions")
            return {'status': 'success', 'results': results}
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")
            return {'status': 'failed', 'message': str(e)}
            
    def reset_paper_trading(self) -> Dict:
        """
        Reset paper trading account to initial state.
        
        This function:
        - Clears all positions
        - Resets account balance to initial amount
        - Clears order history
        - Creates a backup of previous trading data
        - Logs the reset action
        
        Returns:
            Dict: Status of the reset operation
        """
        if not self.is_paper_trading:
            logger.warning("Cannot reset paper trading when in live trading mode.")
            return {'status': 'failed', 'message': 'Not in paper trading mode'}
            
        try:
            # Save current state for backup
            backup_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create backup directory
            backup_dir = f"data/paper_trading_backup_{backup_timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            
            # Backup positions
            if self.paper_positions:
                with open(f"{backup_dir}/positions.json", 'w') as f:
                    json.dump(self.paper_positions, f, indent=2, default=str)
            
            # Backup orders
            if self.paper_orders:
                with open(f"{backup_dir}/orders.json", 'w') as f:
                    json.dump(self.paper_orders, f, indent=2, default=str)
            
            # Backup balance
            with open(f"{backup_dir}/account_info.json", 'w') as f:
                json.dump({
                    'initial_balance': self.initial_balance if hasattr(self, 'initial_balance') else float(os.getenv('INITIAL_BALANCE', '100000')),
                    'current_balance': self.paper_balance,
                    'reset_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }, f, indent=2)
            
            # Clear current state
            previous_balance = self.paper_balance
            previous_positions_count = len(self.paper_positions)
            previous_orders_count = len(self.paper_orders)
            
            # Reset to initial state
            self.paper_positions = {}
            self.paper_orders = {}
            self.paper_balance = float(os.getenv('INITIAL_BALANCE', '100000'))
            self.paper_order_id = 1000
            
            # Log the reset
            logger.info(f"Reset paper trading account: Balance {previous_balance:.2f} -> {self.paper_balance:.2f}, "
                        f"cleared {previous_positions_count} positions and {previous_orders_count} orders. "
                        f"Backup created at {backup_dir}")
            
            return {
                'status': 'success',
                'message': 'Paper trading account reset successfully',
                'details': {
                    'previous_balance': previous_balance,
                    'new_balance': self.paper_balance,
                    'positions_cleared': previous_positions_count,
                    'orders_cleared': previous_orders_count,
                    'backup_location': backup_dir
                }
            }
        except Exception as e:
            logger.error(f"Error resetting paper trading account: {str(e)}")
            return {'status': 'failed', 'message': f'Error resetting paper trading: {str(e)}'}
    
    def get_trading_statistics(self) -> Dict:
        """
        Calculate and return trading statistics and performance metrics.
        
        Returns:
            Dict: Trading statistics including win/loss ratio, average profit/loss, etc.
        """
        try:
            # Get completed orders
            if self.is_paper_trading:
                completed_orders = [
                    order for order in self.paper_orders.values()
                    if order['status'] == 'COMPLETE'
                ]
            else:
                # For live trading, fetch from broker
                orders_response = self.get_order_history()
                if orders_response['status'] != 'success':
                    return {'status': 'failed', 'message': 'Could not fetch order history'}
                    
                completed_orders = [
                    order for order in orders_response['orders']
                    if order['status'] in ['COMPLETE', 'FILLED', 'EXECUTED']
                ]
            
            # Calculate statistics
            if not completed_orders:
                return {
                    'status': 'success',
                    'message': 'No completed trades yet',
                    'stats': {
                        'total_trades': 0,
                        'win_rate': 0,
                        'profit_factor': 0,
                        'average_profit': 0,
                        'average_loss': 0,
                        'largest_profit': 0,
                        'largest_loss': 0,
                        'total_profit': 0,
                        'total_loss': 0
                    }
                }
            
            # Organize trades into buy/sell pairs
            buys = {}
            sells = {}
            trades = []
            
            for order in completed_orders:
                symbol = order['symbol']
                
                if order['transaction_type'] == 'BUY':
                    if symbol not in buys:
                        buys[symbol] = []
                    buys[symbol].append(order)
                else:  # SELL
                    if symbol not in sells:
                        sells[symbol] = []
                    sells[symbol].append(order)
            
            # Match buys and sells to construct trades
            for symbol in buys.keys():
                if symbol in sells:
                    # Sort by timestamp
                    buys[symbol].sort(key=lambda x: x['timestamp'] if isinstance(x['timestamp'], str) else x['timestamp'].isoformat())
                    sells[symbol].sort(key=lambda x: x['timestamp'] if isinstance(x['timestamp'], str) else x['timestamp'].isoformat())
                    
                    # Create trade records
                    for buy in buys[symbol]:
                        for sell in list(sells[symbol]):
                            # Ensure sell comes after buy
                            if (isinstance(sell['timestamp'], str) and isinstance(buy['timestamp'], str) and 
                                sell['timestamp'] > buy['timestamp']) or (
                                not isinstance(sell['timestamp'], str) and not isinstance(buy['timestamp'], str) and
                                sell['timestamp'] > buy['timestamp']):
                                
                                # Calculate profit/loss
                                buy_price = buy['price']
                                sell_price = sell['price']
                                quantity = min(buy['qty'], sell['qty'])
                                
                                pnl = (sell_price - buy_price) * quantity
                                
                                # Account for costs if available
                                buy_cost = buy.get('costs', 0)
                                sell_cost = sell.get('costs', 0)
                                
                                total_pnl = pnl - buy_cost - sell_cost
                                
                                trades.append({
                                    'symbol': symbol,
                                    'buy_time': buy['timestamp'],
                                    'sell_time': sell['timestamp'],
                                    'buy_price': buy_price,
                                    'sell_price': sell_price,
                                    'quantity': quantity,
                                    'pnl': total_pnl,
                                    'costs': buy_cost + sell_cost
                                })
                                
                                # Update the remaining quantity
                                sell['qty'] -= quantity
                                if sell['qty'] <= 0:
                                    sells[symbol].remove(sell)
                                
                                break
            
            # Calculate statistics
            winning_trades = [trade for trade in trades if trade['pnl'] > 0]
            losing_trades = [trade for trade in trades if trade['pnl'] <= 0]
            
            total_profit = sum(trade['pnl'] for trade in winning_trades)
            total_loss = abs(sum(trade['pnl'] for trade in losing_trades))
            
            win_rate = len(winning_trades) / len(trades) if trades else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
            
            average_profit = total_profit / len(winning_trades) if winning_trades else 0
            average_loss = total_loss / len(losing_trades) if losing_trades else 0
            
            largest_profit = max([trade['pnl'] for trade in winning_trades]) if winning_trades else 0
            largest_loss = min([trade['pnl'] for trade in losing_trades]) if losing_trades else 0
            
            # Calculate additional metrics
            average_trade = (total_profit - total_loss) / len(trades) if trades else 0
            max_consecutive_wins = max_consecutive_losses = current_streak = 0
            prev_win = None
            
            for trade in trades:
                is_win = trade['pnl'] > 0
                
                if prev_win is None:
                    current_streak = 1
                elif is_win == prev_win:
                    current_streak += 1
                else:
                    current_streak = 1
                    
                if is_win:
                    max_consecutive_wins = max(max_consecutive_wins, current_streak)
                else:
                    max_consecutive_losses = max(max_consecutive_losses, current_streak)
                    
                prev_win = is_win
            
            return {
                'status': 'success',
                'stats': {
                    'total_trades': len(trades),
                    'winning_trades': len(winning_trades),
                    'losing_trades': len(losing_trades),
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'average_profit': average_profit,
                    'average_loss': average_loss,
                    'average_trade': average_trade,
                    'largest_profit': largest_profit,
                    'largest_loss': largest_loss,
                    'total_profit': total_profit,
                    'total_loss': total_loss,
                    'net_profit': total_profit - total_loss,
                    'max_consecutive_wins': max_consecutive_wins,
                    'max_consecutive_losses': max_consecutive_losses
                },
                'trades': trades
            }
        except Exception as e:
            logger.error(f"Error calculating trading statistics: {str(e)}")
            return {'status': 'failed', 'message': f'Error calculating statistics: {str(e)}'}

