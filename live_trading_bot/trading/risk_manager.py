"""
Risk manager for controlling trading risk.
"""
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk manager for controlling trading risk.
    """
    
    def __init__(self, broker, position_manager, config):
        """
        Initialize the risk manager
        
        Args:
            broker: Broker instance
            position_manager: Position manager instance
            config: Application configuration
        """
        self.broker = broker
        self.position_manager = position_manager
        self.config = config
        
        # Risk parameters
        self.max_position_size_pct = float(config.get('MAX_POSITION_SIZE_PCT', 0.4))
        self.max_open_positions = int(config.get('MAX_OPEN_POSITIONS', 8))
        self.max_trades_per_day = int(config.get('MAX_TRADES_PER_DAY', 10))
        self.daily_loss_limit_pct = float(config.get('DAILY_LOSS_LIMIT_PCT', 0.035))
        self.daily_profit_target_pct = float(config.get('DAILY_PROFIT_TARGET_PCT', 0.10))
        self.max_drawdown_allowed = float(config.get('MAX_DRAWDOWN_ALLOWED', 0.15))
        
        # Risk state
        self.daily_trades = {}  # Symbol -> number of trades
        self.daily_profit_loss = {}  # Symbol -> P/L
        self.initial_balance = self._get_initial_balance()
        self.max_balance = self.initial_balance
        self.trading_stopped = {}  # Symbol -> reason
        
        logger.info("Risk manager initialized")
    
    def _get_initial_balance(self):
        """
        Get initial balance from broker
        
        Returns:
            float: Initial balance
        """
        try:
            # Get account info
            funds = self.broker.get_funds()
            
            if not funds:
                return float(self.config.get('DEFAULT_INITIAL_BALANCE', 100000))
                
            return funds.get('balance', float(self.config.get('DEFAULT_INITIAL_BALANCE', 100000)))
            
        except Exception as e:
            logger.exception(f"Error getting initial balance: {str(e)}")
            return float(self.config.get('DEFAULT_INITIAL_BALANCE', 100000))
    
    def reset_daily_counters(self):
        """
        Reset daily counters
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Reset daily counters
            self.daily_trades = {}
            self.daily_profit_loss = {}
            self.trading_stopped = {}
            
            logger.info("Daily risk counters reset")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error resetting daily counters: {str(e)}")
            return False
    
    def check_max_positions(self):
        """
        Check if max positions limit is reached
        
        Returns:
            bool: True if limit reached, False otherwise
        """
        try:
            # Get open positions
            positions = self.position_manager.get_positions()
            
            # Check if limit reached
            return len(positions) >= self.max_open_positions
            
        except Exception as e:
            logger.exception(f"Error checking max positions: {str(e)}")
            return True  # Default to safe option
    
    def check_max_trades_per_day(self, symbol):
        """
        Check if max trades per day limit is reached for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if limit reached, False otherwise
        """
        try:
            # Check if symbol has trades today
            if symbol not in self.daily_trades:
                return False
                
            # Check if limit reached
            return self.daily_trades[symbol] >= self.max_trades_per_day
            
        except Exception as e:
            logger.exception(f"Error checking max trades per day: {str(e)}")
            return True  # Default to safe option
    
    def check_daily_loss_limit(self):
        """
        Check if daily loss limit is reached
        
        Returns:
            bool: True if limit reached, False otherwise
        """
        try:
            # Calculate total daily loss
            total_pnl = sum(self.daily_profit_loss.values())
            
            # Check if loss limit reached
            return total_pnl < -self.initial_balance * self.daily_loss_limit_pct
            
        except Exception as e:
            logger.exception(f"Error checking daily loss limit: {str(e)}")
            return True  # Default to safe option
    
    def check_daily_profit_target(self, symbol=None):
        """
        Check if daily profit target is reached
        
        Args:
            symbol (str, optional): Trading symbol
            
        Returns:
            bool: True if target reached, False otherwise
        """
        try:
            if symbol:
                # Check specific symbol
                if symbol not in self.daily_profit_loss:
                    return False
                    
                # Check if profit target reached for symbol
                symbol_pnl = self.daily_profit_loss[symbol]
                return symbol_pnl > self.initial_balance * self.daily_profit_target_pct / len(self.position_manager.get_positions())
            else:
                # Check all symbols
                total_pnl = sum(self.daily_profit_loss.values())
                return total_pnl > self.initial_balance * self.daily_profit_target_pct
            
        except Exception as e:
            logger.exception(f"Error checking daily profit target: {str(e)}")
            return False
    
    def check_max_drawdown(self):
        """
        Check if max drawdown limit is reached
        
        Returns:
            bool: True if limit reached, False otherwise
        """
        try:
            # Get current balance
            funds = self.broker.get_funds()
            
            if not funds:
                return False
                
            current_balance = funds.get('balance', 0)
            
            # Update max balance
            if current_balance > self.max_balance:
                self.max_balance = current_balance
                
            # Calculate drawdown
            drawdown = (self.max_balance - current_balance) / self.max_balance if self.max_balance > 0 else 0
            
            # Check if drawdown limit reached
            return drawdown > self.max_drawdown_allowed
            
        except Exception as e:
            logger.exception(f"Error checking max drawdown: {str(e)}")
            return False
    
    def calculate_position_size(self, symbol, price):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol (str): Trading symbol
            price (float): Current price
            
        Returns:
            int: Number of shares to buy
        """
        try:
            # Get available funds
            funds = self.broker.get_funds()
            
            if not funds:
                return 0
                
            available_balance = funds.get('balance', 0)
            
            # Apply position size limit
            max_position_value = available_balance * self.max_position_size_pct
            
            # Adjust based on market regime
            if symbol in self.trading_stopped:
                return 0
                
            # Adjust for consecutive losses
            # This logic would need to be added based on your trading strategy
            
            # Calculate shares to buy
            max_shares = int(max_position_value / price)
            
            return max_shares
            
        except Exception as e:
            logger.exception(f"Error calculating position size: {str(e)}")
            return 0
    
    def record_trade(self, symbol, quantity, price, trade_type):
        """
        Record a trade for risk tracking
        
        Args:
            symbol (str): Trading symbol
            quantity (int): Number of shares
            price (float): Trade price
            trade_type (str): Trade type (BUY, SELL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Initialize counters for symbol if not exists
            if symbol not in self.daily_trades:
                self.daily_trades[symbol] = 0
                
            if symbol not in self.daily_profit_loss:
                self.daily_profit_loss[symbol] = 0
                
            # Update trade count
            self.daily_trades[symbol] += 1
            
            # Update profit/loss for SELL trades
            if trade_type.upper() == 'SELL':
                # Get position
                position = self.position_manager.get_position(symbol)
                
                if position:
                    # Calculate P/L
                    avg_price = position.get('average_price', 0)
                    trade_pnl = (price - avg_price) * quantity
                    
                    # Update P/L
                    self.daily_profit_loss[symbol] += trade_pnl
            
            logger.info(f"Recorded trade: {symbol}, {trade_type}, {quantity} @ {price}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error recording trade: {str(e)}")
            return False
    
    def stop_trading(self, symbol, reason):
        """
        Stop trading for a symbol
        
        Args:
            symbol (str): Trading symbol
            reason (str): Reason for stopping
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Stop trading for symbol
            self.trading_stopped[symbol] = reason
            
            logger.warning(f"Trading stopped for {symbol}: {reason}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error stopping trading: {str(e)}")
            return False
    
    def resume_trading(self, symbol):
        """
        Resume trading for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if trading is stopped for symbol
            if symbol not in self.trading_stopped:
                return True
                
            # Resume trading
            del self.trading_stopped[symbol]
            
            logger.info(f"Trading resumed for {symbol}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error resuming trading: {str(e)}")
            return False
    
    def is_trading_allowed(self, symbol):
        """
        Check if trading is allowed for a symbol
        
        Args:
            symbol (str): Trading symbol
            
        Returns:
            tuple: (allowed, reason)
        """
        try:
            # Check if trading is stopped for symbol
            if symbol in self.trading_stopped:
                return False, self.trading_stopped[symbol]
                
            # Check max positions
            if self.check_max_positions():
                return False, "max_positions_reached"
                
            # Check max trades per day
            if self.check_max_trades_per_day(symbol):
                return False, "max_trades_per_day_reached"
                
            # Check daily loss limit
            if self.check_daily_loss_limit():
                return False, "daily_loss_limit_reached"
                
            # Check daily profit target
            if self.check_daily_profit_target(symbol):
                return False, "daily_profit_target_reached"
                
            # Check max drawdown
            if self.check_max_drawdown():
                return False, "max_drawdown_reached"
                
            return True, None
            
        except Exception as e:
            logger.exception(f"Error checking if trading is allowed: {str(e)}")
            return False, "error"
    
    def get_risk_report(self):
        """
        Get risk report
        
        Returns:
            dict: Risk report
        """
        try:
            # Get account info
            funds = self.broker.get_funds()
            
            if not funds:
                return {}
                
            current_balance = funds.get('balance', 0)
            
            # Calculate drawdown
            drawdown = (self.max_balance - current_balance) / self.max_balance if self.max_balance > 0 else 0
            
            # Calculate daily P/L
            daily_pnl = sum(self.daily_profit_loss.values())
            daily_pnl_pct = daily_pnl / self.initial_balance if self.initial_balance > 0 else 0
            
            # Calculate position allocation
            positions = self.position_manager.get_positions()
            total_position_value = sum(p.get('quantity', 0) * p.get('last_price', 0) for p in positions.values())
            allocation_pct = total_position_value / current_balance if current_balance > 0 else 0
            
            # Prepare report
            report = {
                'current_balance': current_balance,
                'initial_balance': self.initial_balance,
                'max_balance': self.max_balance,
                'drawdown': drawdown,
                'max_drawdown_allowed': self.max_drawdown_allowed,
                'daily_pnl': daily_pnl,
                'daily_pnl_pct': daily_pnl_pct,
                'daily_loss_limit_pct': self.daily_loss_limit_pct,
                'daily_profit_target_pct': self.daily_profit_target_pct,
                'total_position_value': total_position_value,
                'allocation_pct': allocation_pct,
                'max_position_size_pct': self.max_position_size_pct,
                'open_positions': len(positions),
                'max_open_positions': self.max_open_positions,
                'trading_stopped': self.trading_stopped
            }
            
            return report
            
        except Exception as e:
            logger.exception(f"Error getting risk report: {str(e)}")
            return {}