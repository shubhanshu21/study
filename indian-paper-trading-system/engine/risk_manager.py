import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
import pytz
from config.trading_config import (
    MAX_POSITION_SIZE_PCT, MAX_OPEN_POSITIONS, MAX_TRADES_PER_DAY,
    DAILY_LOSS_LIMIT_PCT, DAILY_PROFIT_TARGET_PCT, MAX_DRAWDOWN_ALLOWED
)
from config.market_config import MarketConfig

logger = logging.getLogger(__name__)

class RiskManager:
    """Risk management system for the trading environment"""
    
    def __init__(self, initial_balance=100000):
        """Initialize RiskManager with account parameters"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        
        # Daily tracking variables
        self.current_date = datetime.now(self.timezone).date()
        self.daily_trades = 0
        self.daily_high_capital = initial_balance
        self.daily_low_capital = initial_balance
        self.daily_profit = 0
        self.daily_loss = 0
        
        # Position tracking
        self.open_positions = {}  # Symbol -> position details
        self.open_positions_count = 0
        self.position_size_pct = MAX_POSITION_SIZE_PCT
        
        # Risk metrics
        self.max_net_worth = initial_balance
        self.max_drawdown = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.drawdown_reduction_applied = False
        
        # Trading status
        self.trading_stopped_for_day = False
        self.trading_stopped_reason = None
        
        # Market regime detection
        self.current_market_regime = "unknown"
        
        logger.info(f"RiskManager initialized with balance: ₹{initial_balance:.2f}")
    
    def check_new_day(self, current_date=None):
        """Check if it's a new trading day and reset daily counters"""
        if current_date is None:
            current_date = datetime.now(self.timezone).date()
        
        if current_date != self.current_date:
            # It's a new day, reset daily counters
            logger.info(f"New trading day: {current_date}, resetting daily counters")
            
            # Store previous date
            previous_date = self.current_date
            
            # Update current date
            self.current_date = current_date
            
            # Reset daily counters
            self.daily_trades = 0
            self.daily_high_capital = self.current_balance
            self.daily_low_capital = self.current_balance
            self.trading_stopped_for_day = False
            self.trading_stopped_reason = None
            self.daily_profit = 0
            self.daily_loss = 0
            
            # Increment days in trade for all positions
            for symbol in self.open_positions:
                self.open_positions[symbol]['days_in_trade'] += 1
            
            return True
        
        return False
    
    def update_balance(self, new_balance):
        """Update current balance and related metrics"""
        previous_balance = self.current_balance
        self.current_balance = new_balance
        
        # Update daily capital extremes
        self.daily_high_capital = max(self.daily_high_capital, new_balance)
        self.daily_low_capital = min(self.daily_low_capital, new_balance)
        
        # Update maximum account value
        self.max_net_worth = max(self.max_net_worth, new_balance)
        
        # Calculate current drawdown
        if self.max_net_worth > 0:
            current_drawdown = (self.max_net_worth - new_balance) / self.max_net_worth
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Update daily P&L
        daily_change = new_balance - previous_balance
        if daily_change > 0:
            self.daily_profit += daily_change
        else:
            self.daily_loss += abs(daily_change)
        
        logger.debug(f"Balance updated: ₹{new_balance:.2f}, Change: ₹{daily_change:.2f}, Daily P&L: +₹{self.daily_profit:.2f}/-₹{self.daily_loss:.2f}")
        
        # Check risk limits after update
        self.check_risk_limits()
    
    def check_risk_limits(self):
        """Check if any risk limits are breached"""
        # Calculate daily P&L percentage
        daily_pnl_pct = (self.current_balance - self.initial_balance) / self.initial_balance
        
        # Check daily loss limit
        if daily_pnl_pct < -DAILY_LOSS_LIMIT_PCT and not self.trading_stopped_for_day:
            self.trading_stopped_for_day = True
            self.trading_stopped_reason = "daily_loss_limit"
            logger.warning(f"Trading stopped for day due to daily loss limit: {daily_pnl_pct:.2%}")
            return False
        
        # Don't stop trading on daily profit target in strong uptrends
        if self.current_market_regime == "trending_up" and daily_pnl_pct > DAILY_PROFIT_TARGET_PCT:
            # Keep trading in strong uptrends even after hitting profit target
            self.trading_stopped_for_day = False
            self.trading_stopped_reason = None
        # Stop trading if daily profit target hit in non-trending markets
        elif daily_pnl_pct > DAILY_PROFIT_TARGET_PCT and not self.trading_stopped_for_day and self.current_market_regime != "trending_up":
            self.trading_stopped_for_day = True
            self.trading_stopped_reason = "daily_profit_target"
            logger.info(f"Trading stopped for day due to reaching profit target: {daily_pnl_pct:.2%}")
            return False
        
        # Dynamic position sizing based on drawdown
        if self.max_drawdown > MAX_DRAWDOWN_ALLOWED and not self.drawdown_reduction_applied:
            # Reduce position size after large drawdown
            self.position_size_pct = MAX_POSITION_SIZE_PCT * 0.5
            self.drawdown_reduction_applied = True
            logger.warning(f"Position size reduced to {self.position_size_pct:.2%} due to drawdown of {self.max_drawdown:.2%}")
        
        # Adaptive position sizing based on winning streak
        if self.consecutive_wins >= 3 and not self.drawdown_reduction_applied:
            # Increase position size after 3 consecutive wins
            self.position_size_pct = min(self.position_size_pct * 1.2, MAX_POSITION_SIZE_PCT * 1.5)
            logger.info(f"Position size increased to {self.position_size_pct:.2%} after {self.consecutive_wins} consecutive wins")
        
        # Reset position size after recovery
        if self.drawdown_reduction_applied and self.max_drawdown < MAX_DRAWDOWN_ALLOWED/3:
            self.position_size_pct = MAX_POSITION_SIZE_PCT
            self.drawdown_reduction_applied = False
            logger.info(f"Position size restored to {self.position_size_pct:.2%} after drawdown recovery")
        
        # Scale back position sizing after consecutive losses
        if self.consecutive_losses >= 2:
            # Reduce position size after consecutive losses
            self.position_size_pct = max(self.position_size_pct * 0.8, MAX_POSITION_SIZE_PCT * 0.4)
            logger.info(f"Position size reduced to {self.position_size_pct:.2%} after {self.consecutive_losses} consecutive losses")
        
        return True
    
    def can_place_trade(self, symbol, price, quantity):
        """Check if a new trade can be placed based on risk parameters"""
        # Check if market is open
        if not self.market_config.is_market_open():
            logger.warning("Cannot place trade - market is closed")
            return False, "market_closed"
        
        # Check if trading is stopped for the day
        if self.trading_stopped_for_day:
            logger.warning(f"Cannot place trade - trading stopped for day due to {self.trading_stopped_reason}")
            return False, self.trading_stopped_reason
        
        # Check if max trades per day reached
        if self.daily_trades >= MAX_TRADES_PER_DAY:
            logger.warning(f"Cannot place trade - max trades per day ({MAX_TRADES_PER_DAY}) reached")
            return False, "max_trades_reached"
        
        # Check if max open positions reached
        if self.open_positions_count >= MAX_OPEN_POSITIONS and symbol not in self.open_positions:
            logger.warning(f"Cannot place trade - max open positions ({MAX_OPEN_POSITIONS}) reached")
            return False, "max_positions_reached"
        
        # Check if we have enough capital
        trade_value = price * quantity
        if trade_value > self.current_balance:
            logger.warning(f"Cannot place trade - insufficient capital (₹{self.current_balance:.2f}) for trade value (₹{trade_value:.2f})")
            return False, "insufficient_capital"
        
        # Check position sizing limits (if new position)
        if symbol not in self.open_positions:
            max_position_value = self.current_balance * self.position_size_pct
            if trade_value > max_position_value:
                logger.warning(f"Cannot place trade - exceeds position size limit (₹{max_position_value:.2f})")
                return False, "position_size_exceeded"
        
        # Check high volatility period
        if self.market_config.should_avoid_high_volatility():
            # Only warn about high volatility, but still allow trading
            logger.info("Warning: Trading during high volatility period (market open/close)")
        
        return True, None
    
    def get_position_size(self, symbol, price, sizing_pct=None):
        """Calculate appropriate position size based on risk parameters"""
        if sizing_pct is None:
            sizing_pct = self.position_size_pct
        
        # Cap at max position size percentage
        sizing_pct = min(sizing_pct, MAX_POSITION_SIZE_PCT)
        
        # Calculate position value
        position_value = self.current_balance * sizing_pct
        
        # Calculate number of shares
        num_shares = int(position_value // price)
        
        # Make sure we're not exceeding available balance
        while num_shares * price > self.current_balance and num_shares > 0:
            num_shares -= 1
        
        return num_shares
    
    def register_trade(self, trade_type, symbol, price, quantity, pnl=0):
        """Register a new trade and update counters"""
        trade_value = price * quantity
        
        if trade_type.lower() == 'buy':
            # New position or add to existing
            if symbol in self.open_positions:
                # Update existing position
                position = self.open_positions[symbol]
                new_quantity = position['quantity'] + quantity
                new_avg_price = (position['avg_price'] * position['quantity'] + price * quantity) / new_quantity
                
                position['quantity'] = new_quantity
                position['avg_price'] = new_avg_price
                position['last_update'] = datetime.now(self.timezone)
                
                logger.info(f"Added to position: {symbol}, Total: {new_quantity}, Avg Price: ₹{new_avg_price:.2f}")
            else:
                # Create new position
                self.open_positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_date': datetime.now(self.timezone).date(),
                    'entry_time': datetime.now(self.timezone).time(),
                    'days_in_trade': 0,
                    'last_update': datetime.now(self.timezone)
                }
                self.open_positions_count += 1
                
                logger.info(f"New position: {symbol}, Quantity: {quantity}, Price: ₹{price:.2f}")
        
        elif trade_type.lower() == 'sell':
            # Close or reduce position
            if symbol in self.open_positions:
                position = self.open_positions[symbol]
                
                if quantity >= position['quantity']:
                    # Close position completely
                    final_quantity = position['quantity']
                    entry_price = position['avg_price']
                    days_held = (datetime.now(self.timezone).date() - position['entry_date']).days
                    
                    # Calculate P&L
                    trade_pnl = (price - entry_price) * final_quantity
                    trade_pnl_pct = (price - entry_price) / entry_price
                    
                    # Update win/loss counters
                    if trade_pnl > 0:
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                        logger.info(f"Winning trade: {symbol}, Profit: ₹{trade_pnl:.2f} ({trade_pnl_pct:.2%})")
                    else:
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0
                        logger.info(f"Losing trade: {symbol}, Loss: ₹{trade_pnl:.2f} ({trade_pnl_pct:.2%})")
                    
                    # Remove position
                    del self.open_positions[symbol]
                    self.open_positions_count -= 1
                    
                    logger.info(f"Closed position: {symbol}, Days held: {days_held}, P&L: ₹{trade_pnl:.2f} ({trade_pnl_pct:.2%})")
                else:
                    # Reduce position
                    position['quantity'] -= quantity
                    position['last_update'] = datetime.now(self.timezone)
                    
                    logger.info(f"Reduced position: {symbol}, Remaining: {position['quantity']}")
            else:
                logger.warning(f"Attempted to sell non-existent position: {symbol}")
                return False
        
        # Increment trade counter
        self.daily_trades += 1
        
        # Update balance based on P&L
        if pnl != 0:
            self.update_balance(self.current_balance + pnl)
        
        return True
    
    def get_position_risk_metrics(self, symbol, current_price):
        """Calculate risk metrics for a given position"""
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        entry_price = position['avg_price']
        quantity = position['quantity']
        days_in_trade = position['days_in_trade']
        
        # Calculate P&L
        unrealized_pnl = (current_price - entry_price) * quantity
        unrealized_pnl_pct = (current_price - entry_price) / entry_price
        
        # Calculate position value
        position_value = current_price * quantity
        position_pct_of_portfolio = position_value / self.current_balance
        
        return {
            'symbol': symbol,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': current_price,
            'days_in_trade': days_in_trade,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'position_value': position_value,
            'position_pct_of_portfolio': position_pct_of_portfolio
        }
    
    def get_portfolio_risk_metrics(self, current_prices):
        """Calculate risk metrics for the entire portfolio
        
        Args:
            current_prices: Dict mapping symbols to current prices
        """
        # Calculate total position value
        total_position_value = 0
        unrealized_pnl = 0
        
        for symbol, position in self.open_positions.items():
            if symbol in current_prices:
                price = current_prices[symbol]
                quantity = position['quantity']
                entry_price = position['avg_price']
                
                position_value = price * quantity
                position_pnl = (price - entry_price) * quantity
                
                total_position_value += position_value
                unrealized_pnl += position_pnl
        
        # Calculate metrics
        cash_ratio = self.current_balance / (self.current_balance + total_position_value) if total_position_value > 0 else 1.0
        equity_ratio = 1.0 - cash_ratio
        
        # Calculate exposure
        net_exposure = equity_ratio  # As % of total capital
        
        # Calculate beta-weighted exposure (if we had beta values)
        beta_weighted_exposure = net_exposure  # Simplified for now
        
        return {
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_position_value': total_position_value,
            'total_account_value': self.current_balance + total_position_value,
            'cash_ratio': cash_ratio,
            'equity_ratio': equity_ratio,
            'net_exposure': net_exposure,
            'beta_weighted_exposure': beta_weighted_exposure,
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0,
            'daily_profit': self.daily_profit,
            'daily_loss': self.daily_loss,
            'daily_net': self.daily_profit - self.daily_loss,
            'max_drawdown': self.max_drawdown,
            'position_count': self.open_positions_count,
            'daily_trades': self.daily_trades,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'trading_stopped_for_day': self.trading_stopped_for_day,
            'trading_stopped_reason': self.trading_stopped_reason,
            'current_market_regime': self.current_market_regime
        }
    
    def get_dynamic_stop_loss(self, symbol, entry_price, current_price, atr_value, position_type='long'):
        """Calculate dynamic stop loss based on ATR and position type"""
        # Dynamic ATR multiplier based on market regime
        atr_multiplier = 2.0  # Default
        
        # Adjust multiplier based on market regime
        if self.current_market_regime == "trending_up":
            atr_multiplier = 2.5 if position_type == 'long' else 1.8
        elif self.current_market_regime == "trending_down":
            atr_multiplier = 1.8 if position_type == 'long' else 2.5
        elif self.current_market_regime == "ranging":
            atr_multiplier = 1.5
        elif self.current_market_regime == "volatile":
            atr_multiplier = 3.0
        
        # Adjust for consecutive losses (tighten stops)
        if self.consecutive_losses >= 3:
            atr_multiplier *= 0.8  # Reduce by 20% after 3 consecutive losses
        
        # Calculate stop loss based on position type
        if position_type == 'long':
            stop_loss = entry_price - (atr_value * atr_multiplier)
            
            # Adjust for current price if trailing stop is enabled
            if current_price > entry_price:
                trailing_stop = current_price - (atr_value * atr_multiplier)
                stop_loss = max(stop_loss, trailing_stop)
        else:  # short position
            stop_loss = entry_price + (atr_value * atr_multiplier)
            
            # Adjust for current price if trailing stop is enabled
            if current_price < entry_price:
                trailing_stop = current_price + (atr_value * atr_multiplier)
                stop_loss = min(stop_loss, trailing_stop)
        
        return stop_loss
    
    def get_dynamic_target(self, entry_price, atr_value, position_type='long', min_rr_ratio=1.5):
        """Calculate dynamic target based on ATR and risk-reward ratio"""
        # Dynamic ATR multiplier based on market regime
        target_multiplier = 3.0  # Default
        
        # Adjust multiplier based on market regime
        if self.current_market_regime == "trending_up":
            target_multiplier = 6.0 if position_type == 'long' else 4.5
        elif self.current_market_regime == "trending_down":
            target_multiplier = 4.5 if position_type == 'long' else 6.0
        elif self.current_market_regime == "ranging":
            target_multiplier = 3.0
        elif self.current_market_regime == "volatile":
            target_multiplier = 7.0
        
        # Calculate target
        if position_type == 'long':
            target = entry_price + (atr_value * target_multiplier)
        else:  # short position
            target = entry_price - (atr_value * target_multiplier)
        
        return target
    
    def update_market_regime(self, regime):
        """Update current market regime"""
        self.current_market_regime = regime
        logger.info(f"Market regime updated to: {regime}")
        
    def should_exit_position(self, symbol, current_price, trailing_stop, target_price, position_type='long'):
        """Determine if a position should be exited based on stops and targets"""
        if symbol not in self.open_positions:
            return False, None
        
        position = self.open_positions[symbol]
        days_in_trade = position['days_in_trade']
        entry_price = position['avg_price']
        
        # Calculate current profit percentage
        if position_type == 'long':
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short position
            profit_pct = (entry_price - current_price) / entry_price
        
        # Check stop loss
        if position_type == 'long' and current_price <= trailing_stop:
            return True, "stop_loss"
        elif position_type == 'short' and current_price >= trailing_stop:
            return True, "stop_loss"
        
        # Check target
        if position_type == 'long' and current_price >= target_price:
            return True, "target"
        elif position_type == 'short' and current_price <= target_price:
            return True, "target"
        
        # Dynamic exits based on market regime
        if self.current_market_regime == "volatile" and profit_pct > 0.04:
            # Take higher profits in volatile markets
            return True, "regime_volatile_take_profit"
        
        elif self.current_market_regime == "ranging" and profit_pct > 0.05:
            # Take higher profits in ranging markets
            return True, "regime_ranging_take_profit"
        
        # Time-based exit for swing trading
        if days_in_trade > 8 and profit_pct < 0.02:
            # Exit trades not performing after 8 days
            return True, "time_based_exit"
        
        return False, None