import pandas as pd
import numpy as np
import logging
import os
import json
from datetime import datetime, time, timedelta
import pytz
from config.market_config import MarketConfig
from config.trading_config import (
    DEFAULT_INITIAL_BALANCE, MAX_POSITION_SIZE_PCT, 
    BROKERAGE_INTRADAY, BROKERAGE_DELIVERY, STT_INTRADAY, STT_DELIVERY,
    EXCHANGE_TXN_CHARGE, SEBI_CHARGES, STAMP_DUTY, GST
)
from engine.risk_manager import RiskManager
from data.processors.indicators import IndicatorProcessor

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """Detects the current market regime (trending, ranging, volatile)"""
    
    def __init__(self, lookback=20):
        self.lookback = lookback
        
    def detect_regime(self, df, current_idx):
        """Detect the current market regime"""
        if current_idx < self.lookback:
            return "unknown"
            
        window = df.iloc[max(0, current_idx-self.lookback):current_idx+1]
        
        # Calculate key metrics for regime detection
        if 'ADX' in window.columns:
            adx = window['ADX'].iloc[-1]
        else:
            adx = 25  # Default to neutral
            
        if 'ATR' in window.columns and 'Close' in window.columns:
            atr_pct = window['ATR'].iloc[-1] / window['Close'].iloc[-1]
        else:
            atr_pct = 0.015  # Default to 1.5% volatility
            
        if 'RSI' in window.columns:
            rsi = window['RSI'].iloc[-1]
        else:
            rsi = 50  # Default to neutral
            
        # Detect trend direction using multiple indicators
        if 'EMA20' in window.columns and 'EMA50' in window.columns:
            trend_direction = 1 if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1] else -1
        else:
            price_change = window['Close'].iloc[-1] / window['Close'].iloc[0] - 1
            trend_direction = 1 if price_change > 0 else -1
        
        # Regime classification logic
        if adx > 25 and abs(rsi - 50) > 15:
            regime = "trending"
            
            if trend_direction > 0:
                return "trending_up"
            else:
                return "trending_down"
                
        elif atr_pct > 0.025:  # High volatility threshold (2.5%)
            return "volatile"
            
        else:
            return "ranging"
            
    def get_regime_parameters(self, regime):
        """Get optimal parameters for the current market regime"""
        
        params = {
            "trending_up": {
                "trailing_stop_atr_multiplier": 2.0,
                "target_atr_multiplier": 6.0,
                "position_size_pct": 0.35,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            },
            "trending_down": {
                "trailing_stop_atr_multiplier": 1.8,
                "target_atr_multiplier": 4.5,
                "position_size_pct": 0.20,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.8
            },
            "ranging": {
                "trailing_stop_atr_multiplier": 1.0,
                "target_atr_multiplier": 3.0,
                "position_size_pct": 0.25,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            },
            "volatile": {
                "trailing_stop_atr_multiplier": 2.5,
                "target_atr_multiplier": 7.0,
                "position_size_pct": 0.18,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.8
            },
            "unknown": {
                "trailing_stop_atr_multiplier": 1.8,
                "target_atr_multiplier": 4.0,
                "position_size_pct": 0.20,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            }
        }
        
        return params.get(regime, params["unknown"])

class TradingEnvironment:
    """Paper trading environment for Indian stock market"""
    
    def __init__(self, initial_balance=DEFAULT_INITIAL_BALANCE, data_path="data/historical"):
        """Initialize trading environment"""
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.data_path = data_path
        
        # Set up market config
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        
        # Set up risk manager
        self.risk_manager = RiskManager(initial_balance)
        
        # Set up market regime detector
        self.regime_detector = MarketRegimeDetector(lookback=20)
        
        # Set up indicator processor
        self.indicator_processor = IndicatorProcessor()
        
        # Positions and orders tracking
        self.positions = {}  # Symbol -> position details
        self.orders = []  # List of all orders
        self.pending_orders = []  # List of pending orders
        
        # Current market data
        self.current_data = {}  # Symbol -> current data
        self.historical_data = {}  # Symbol -> historical data
        
        # Performance tracking
        self.net_worth_history = [initial_balance]
        self.trades = []
        self.daily_stats = []
        
        # Initialize trading day
        self.current_date = datetime.now(self.timezone).date()
        self.current_time = datetime.now(self.timezone).time()
        
        logger.info(f"Trading environment initialized with balance: ₹{initial_balance:.2f}")
    
    def calculate_transaction_costs(self, trade_value, is_intraday=False):
        """Calculate all transaction costs for a trade"""
        # Check for reasonable values and convert percentages
        brokerage_rate = min(BROKERAGE_INTRADAY if is_intraday else BROKERAGE_DELIVERY, 0.05)  # Cap at 5%
        stt_rate = min(STT_INTRADAY if is_intraday else STT_DELIVERY, 0.05)  # Cap at 5%
        exchange_rate = min(EXCHANGE_TXN_CHARGE, 0.01)  # Cap at 1% 
        sebi_rate = min(SEBI_CHARGES, 0.01)  # Cap at 1%
        stamp_rate = min(STAMP_DUTY, 0.01)  # Cap at 1%
        gst_rate = min(GST, 0.30) if GST < 1 else 0.18  # Default to 18% if GST > 1
        
        # Calculate individual components
        brokerage = trade_value * brokerage_rate
        stt = trade_value * stt_rate
        exchange_charge = trade_value * exchange_rate
        sebi_charges = trade_value * sebi_rate
        stamp_duty = trade_value * stamp_rate
        
        # GST applies on brokerage and exchange charges
        gst = (brokerage + exchange_charge) * gst_rate
        
        # Total transaction cost
        total_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst
        
        # Safety check - cap transaction costs at a reasonable percentage
        if total_cost > trade_value * 0.1:  # Cap at 10% of trade value
            logger.warning(f"Transaction costs capped from {total_cost} to {trade_value * 0.1}")
            total_cost = trade_value * 0.1
            
        return {
            'total': total_cost,
            'brokerage': brokerage,
            'stt': stt,
            'exchange_charge': exchange_charge,
            'sebi_charges': sebi_charges,
            'stamp_duty': stamp_duty,
            'gst': gst
        }
    
    def load_historical_data(self, symbol):
        """Load historical data for a symbol"""
        try:
            file_path = os.path.join(self.data_path, f"{symbol}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"Historical data file not found for {symbol}")
                return None
            
            df = pd.read_csv(file_path)
            
            # Convert date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            
            # Calculate technical indicators
            df = self.indicator_processor.calculate_all_indicators(df)
            
            logger.info(f"Loaded and processed historical data for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            return None
    
    def update_market_data(self, symbol, current_data):
        """Update current market data for a symbol"""
        self.current_data[symbol] = current_data
        
        # Update historical data if needed
        if symbol in self.historical_data:
            # Check if we need to append to historical
            hist_last_date = self.historical_data[symbol]['Date'].max()
            current_date = pd.to_datetime(current_data['Date'])
            
            if current_date > hist_last_date:
                # Append to historical
                self.historical_data[symbol] = pd.concat([
                    self.historical_data[symbol],
                    pd.DataFrame([current_data])
                ]).reset_index(drop=True)
                
                # Recalculate indicators for the updated data
                self.historical_data[symbol] = self.indicator_processor.calculate_all_indicators(
                    self.historical_data[symbol]
                )
                
                logger.debug(f"Historical data updated for {symbol}")
        else:
            # Load historical data first time
            self.historical_data[symbol] = self.load_historical_data(symbol)
    
    def update_clock(self, current_date=None, current_time=None):
        """Update the environment's current date and time"""
        if current_date is None:
            current_date = datetime.now(self.timezone).date()
        
        if current_time is None:
            current_time = datetime.now(self.timezone).time()
        
        # Check if day changed
        if current_date != self.current_date:
            self._on_new_day(current_date)
        
        self.current_date = current_date
        self.current_time = current_time
        
        # Check if market is open
        is_market_open = self.market_config.is_market_open(
            datetime.combine(current_date, current_time).replace(tzinfo=self.timezone)
        )
        
        return is_market_open
    
    def _on_new_day(self, new_date):
        """Handle start of new trading day"""
        logger.info(f"New trading day: {new_date}")
        
        # Reset risk manager for new day
        self.risk_manager.check_new_day(new_date)
        
        # Save daily stats
        self._save_daily_stats()
        
        # Process any overnight events
        self._process_overnight_events()
    
    def _save_daily_stats(self):
        """Save statistics for the current trading day"""
        # Calculate daily P&L
        prev_balance = self.net_worth_history[-2] if len(self.net_worth_history) > 1 else self.initial_balance
        current_balance = self.net_worth_history[-1]
        daily_pnl = current_balance - prev_balance
        daily_pnl_pct = daily_pnl / prev_balance if prev_balance > 0 else 0
        
        # Count today's trades
        today_trades = [trade for trade in self.trades if trade['date'] == self.current_date]
        
        # Calculate daily statistics
        daily_stats = {
            'date': self.current_date,
            'open_balance': prev_balance,
            'close_balance': current_balance,
            'daily_pnl': daily_pnl,
            'daily_pnl_pct': daily_pnl_pct,
            'trade_count': len(today_trades),
            'winning_trades': sum(1 for trade in today_trades if trade.get('pnl', 0) > 0),
            'losing_trades': sum(1 for trade in today_trades if trade.get('pnl', 0) < 0),
            'largest_win': max((trade.get('pnl', 0) for trade in today_trades), default=0),
            'largest_loss': min((trade.get('pnl', 0) for trade in today_trades), default=0),
            'positions_held': len(self.positions)
        }
        
        self.daily_stats.append(daily_stats)
        logger.info(f"Daily stats saved for {self.current_date}: P&L: ₹{daily_pnl:.2f} ({daily_pnl_pct:.2%})")
    
    def _process_overnight_events(self):
        """Process any events that happen overnight between trading days"""
        # Update days in trade for all positions
        for symbol in self.positions:
            self.positions[symbol]['days_in_trade'] += 1
    
    def place_order(self, symbol, order_type, quantity, price=None, stop_loss=None, target=None):
        """Place a new order in the paper trading system
        
        Args:
            symbol: Stock symbol
            order_type: 'BUY' or 'SELL'
            quantity: Number of shares
            price: Limit price (None for market order)
            stop_loss: Stop loss price
            target: Target price
        """
        # Check if market is open
        is_market_open = self.market_config.is_market_open(
            datetime.combine(self.current_date, self.current_time).replace(tzinfo=self.timezone)
        )
        
        if not is_market_open:
            logger.warning(f"Order rejected - market is closed: {symbol} {order_type} {quantity}")
            return False, "market_closed"
        
        # Get current price if not provided (for market orders)
        if price is None and symbol in self.current_data:
            price = self.current_data[symbol]['Close']
        elif price is None:
            logger.error(f"Cannot place market order - no current price for {symbol}")
            return False, "no_price_data"
        
        # Check with risk manager if trade is allowed
        can_trade, reason = self.risk_manager.can_place_trade(symbol, price, quantity)
        
        if not can_trade:
            logger.warning(f"Order rejected by risk manager: {symbol} {order_type} {quantity} - {reason}")
            return False, reason
        
        # Create order
        order = {
            'symbol': symbol,
            'order_type': order_type.upper(),
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'target': target,
            'status': 'PENDING',
            'create_time': datetime.now(self.timezone),
            'update_time': datetime.now(self.timezone),
            'transaction_costs': None,
            'filled_price': None,
            'filled_time': None,
            'order_id': f"ORD{len(self.orders) + 1:06d}"
        }
        
        # Add to orders list
        self.orders.append(order)
        self.pending_orders.append(order)
        
        logger.info(f"Order placed: {order['order_id']} - {symbol} {order_type} {quantity} @ ₹{price:.2f}")
        
        # Try to execute immediately if market order
        if price is None:
            self._execute_pending_orders()
            
        return True, order['order_id']
    
    def cancel_order(self, order_id):
        """Cancel a pending order"""
        for i, order in enumerate(self.pending_orders):
            if order['order_id'] == order_id:
                order['status'] = 'CANCELLED'
                order['update_time'] = datetime.now(self.timezone)
                
                # Remove from pending orders
                self.pending_orders.pop(i)
                
                logger.info(f"Order cancelled: {order_id}")
                return True
        
        logger.warning(f"Order not found for cancellation: {order_id}")
        return False
    
    def modify_order(self, order_id, new_price=None, new_quantity=None, new_stop_loss=None, new_target=None):
        """Modify a pending order"""
        for order in self.pending_orders:
            if order['order_id'] == order_id:
                if new_price is not None:
                    order['price'] = new_price
                
                if new_quantity is not None:
                    order['quantity'] = new_quantity
                
                if new_stop_loss is not None:
                    order['stop_loss'] = new_stop_loss
                
                if new_target is not None:
                    order['target'] = new_target
                
                order['update_time'] = datetime.now(self.timezone)
                
                logger.info(f"Order modified: {order_id}")
                return True
        
        logger.warning(f"Order not found for modification: {order_id}")
        return False
    
    def _execute_pending_orders(self):
        """Try to execute any pending orders"""
        # Check if market is open
        is_market_open = self.market_config.is_market_open(
            datetime.combine(self.current_date, self.current_time).replace(tzinfo=self.timezone)
        )
        
        if not is_market_open:
            return
        
        # Process each pending order
        executed_orders = []
        
        for order in self.pending_orders:
            symbol = order['symbol']
            
            # Skip if no current data available
            if symbol not in self.current_data:
                continue
            
            current_price = self.current_data[symbol]['Close']
            
            # Check if order can be executed
            if order['order_type'] == 'BUY':
                if order['price'] is None or current_price <= order['price']:
                    # Execute buy order
                    self._execute_buy_order(order, current_price)
                    executed_orders.append(order)
            
            elif order['order_type'] == 'SELL':
                if order['price'] is None or current_price >= order['price']:
                    # Execute sell order
                    self._execute_sell_order(order, current_price)
                    executed_orders.append(order)
        
        # Remove executed orders from pending list
        for order in executed_orders:
            self.pending_orders.remove(order)
    
    def _execute_buy_order(self, order, execution_price):
        """Execute a buy order"""
        symbol = order['symbol']
        quantity = order['quantity']
        
        # Calculate transaction costs
        trade_value = execution_price * quantity
        txn_costs = self.calculate_transaction_costs(trade_value, is_intraday=False)
        
        # Update order
        order['status'] = 'EXECUTED'
        order['filled_price'] = execution_price
        order['filled_time'] = datetime.now(self.timezone)
        order['update_time'] = datetime.now(self.timezone)
        order['transaction_costs'] = txn_costs
        
        # Update position
        if symbol in self.positions:
            # Update existing position
            position = self.positions[symbol]
            new_quantity = position['quantity'] + quantity
            new_avg_price = (position['avg_price'] * position['quantity'] + execution_price * quantity) / new_quantity
            
            position['quantity'] = new_quantity
            position['avg_price'] = new_avg_price
            position['value'] = new_avg_price * new_quantity
            position['last_update'] = datetime.now(self.timezone)
            
            # Adjust stops if provided
            if order['stop_loss'] is not None:
                position['stop_loss'] = order['stop_loss']
            if order['target'] is not None:
                position['target'] = order['target']
            
            logger.info(f"Added to position: {symbol}, Total: {new_quantity}, Avg Price: ₹{new_avg_price:.2f}")
        else:
            # Create new position
            self.positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'avg_price': execution_price,
                'value': execution_price * quantity,
                'entry_date': self.current_date,
                'entry_time': self.current_time,
                'days_in_trade': 0,
                'last_update': datetime.now(self.timezone),
                'stop_loss': order['stop_loss'],
                'target': order['target'],
                'position_type': 'long'
            }
            
            logger.info(f"New position: {symbol}, Quantity: {quantity}, Price: ₹{execution_price:.2f}")
        
        # Update balance
        total_cost = trade_value + txn_costs['total']
        self.current_balance -= total_cost
        
        # Update risk manager
        self.risk_manager.register_trade('buy', symbol, execution_price, quantity)
        self.risk_manager.update_balance(self.current_balance)
        
        # Record trade in history
        trade = {
            'order_id': order['order_id'],
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': execution_price,
            'value': trade_value,
            'transaction_costs': txn_costs,
            'date': self.current_date,
            'time': self.current_time,
            'balance_after': self.current_balance
        }
        
        self.trades.append(trade)
        
        # Update net worth history
        self._update_net_worth()
        
        logger.info(f"Buy order executed: {symbol}, {quantity} @ ₹{execution_price:.2f}, Total cost: ₹{total_cost:.2f}")
    
    def _execute_sell_order(self, order, execution_price):
        """Execute a sell order"""
        symbol = order['symbol']
        quantity = order['quantity']
        
        # Check if position exists
        if symbol not in self.positions:
            logger.warning(f"Cannot execute sell order - no position for {symbol}")
            order['status'] = 'REJECTED'
            order['update_time'] = datetime.now(self.timezone)
            return
        
        position = self.positions[symbol]
        
        # Check if we have enough quantity
        if position['quantity'] < quantity:
            logger.warning(f"Cannot execute sell order - insufficient quantity for {symbol}")
            order['status'] = 'REJECTED'
            order['update_time'] = datetime.now(self.timezone)
            return
        
        # Calculate transaction costs
        trade_value = execution_price * quantity
        txn_costs = self.calculate_transaction_costs(trade_value, is_intraday=False)
        
        # Calculate P&L
        entry_price = position['avg_price']
        pnl = (execution_price - entry_price) * quantity - txn_costs['total']
        pnl_pct = pnl / (entry_price * quantity)
        
        # Update order
        order['status'] = 'EXECUTED'
        order['filled_price'] = execution_price
        order['filled_time'] = datetime.now(self.timezone)
        order['update_time'] = datetime.now(self.timezone)
        order['transaction_costs'] = txn_costs
        
        # Update position
        if position['quantity'] == quantity:
            # Close position completely
            days_held = (self.current_date - position['entry_date']).days
            
            # Remove position
            del self.positions[symbol]
            
            logger.info(f"Closed position: {symbol}, Days held: {days_held}, P&L: ₹{pnl:.2f} ({pnl_pct:.2%})")
        else:
            # Reduce position
            position['quantity'] -= quantity
            position['value'] = position['avg_price'] * position['quantity']
            position['last_update'] = datetime.now(self.timezone)
            
            logger.info(f"Reduced position: {symbol}, Remaining: {position['quantity']}")
        
        # Update balance
        net_proceeds = trade_value - txn_costs['total']
        self.current_balance += net_proceeds
        
        # Update risk manager
        self.risk_manager.register_trade('sell', symbol, execution_price, quantity, pnl)
        self.risk_manager.update_balance(self.current_balance)
        
        # Record trade in history
        trade = {
            'order_id': order['order_id'],
            'symbol': symbol,
            'action': 'SELL',
            'quantity': quantity,
            'price': execution_price,
            'value': trade_value,
            'transaction_costs': txn_costs,
            'date': self.current_date,
            'time': self.current_time,
            'entry_price': entry_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance_after': self.current_balance
        }
        
        self.trades.append(trade)
        
        # Update net worth history
        self._update_net_worth()
        
        logger.info(f"Sell order executed: {symbol}, {quantity} @ ₹{execution_price:.2f}, P&L: ₹{pnl:.2f} ({pnl_pct:.2%})")
    
    def _update_net_worth(self):
        """Calculate and update current net worth"""
        # Sum up value of all positions
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in self.current_data:
                current_price = self.current_data[symbol]['Close']
                positions_value += current_price * position['quantity']
            else:
                positions_value += position['value']  # Use last known value
        
        # Calculate net worth
        net_worth = self.current_balance + positions_value
        
        # Add to history
        self.net_worth_history.append(net_worth)
        
        # Update max drawdown if needed
        max_net_worth = max(self.net_worth_history)
        if max_net_worth > 0:
            current_drawdown = (max_net_worth - net_worth) / max_net_worth
            self.risk_manager.max_drawdown = max(self.risk_manager.max_drawdown, current_drawdown)
        
        return net_worth
    
    def check_stops_and_targets(self):
        """Check if any positions hit their stop loss or target"""
        # Process each position
        for symbol, position in list(self.positions.items()):
            # Skip if no current data
            if symbol not in self.current_data:
                continue
            
            current_price = self.current_data[symbol]['Close']
            position_type = position.get('position_type', 'long')
            stop_loss = position.get('stop_loss')
            target = position.get('target')
            
            # Check with risk manager if position should be exited
            should_exit, exit_reason = self.risk_manager.should_exit_position(
                symbol, current_price, stop_loss, target, position_type
            )
            
            if should_exit:
                logger.info(f"Exit signal for {symbol}: {exit_reason}, Price: ₹{current_price:.2f}")
                
                # Place sell order
                self.place_order(
                    symbol=symbol,
                    order_type='SELL',
                    quantity=position['quantity'],
                    price=None  # Market order
                )
    
    def update_stops_and_targets(self):
        """Update stop losses and targets based on current market conditions"""
        for symbol, position in self.positions.items():
            # Skip if no current data or historical data
            if symbol not in self.current_data or symbol not in self.historical_data:
                continue
            
            current_price = self.current_data[symbol]['Close']
            position_type = position.get('position_type', 'long')
            entry_price = position['avg_price']
            
            # Get ATR value
            if 'ATR' in self.current_data[symbol]:
                atr_value = self.current_data[symbol]['ATR']
            else:
                # Use a default percentage of price if ATR not available
                atr_value = current_price * 0.02  # 2% volatility
            
            # Determine current market regime
            if 'MARKET_REGIME_TEXT' in self.current_data[symbol]:
                regime = self.current_data[symbol]['MARKET_REGIME_TEXT']
            else:
                # Use detector to determine regime
                historical_df = self.historical_data[symbol]
                current_idx = len(historical_df) - 1
                regime = self.regime_detector.detect_regime(historical_df, current_idx)
            
            # Update risk manager with current regime
            self.risk_manager.update_market_regime(regime)
            
            # Get dynamic stop loss
            new_stop_loss = self.risk_manager.get_dynamic_stop_loss(
                symbol, entry_price, current_price, atr_value, position_type
            )
            
            # Only move stop loss up for long positions, down for short positions
            if position_type == 'long':
                if position.get('stop_loss') is None or new_stop_loss > position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
                    logger.debug(f"Updated stop loss for {symbol}: ₹{new_stop_loss:.2f}")
            else:  # short position
                if position.get('stop_loss') is None or new_stop_loss < position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
                    logger.debug(f"Updated stop loss for {symbol}: ₹{new_stop_loss:.2f}")
            
            # Update target price
            if position.get('target') is None:
                new_target = self.risk_manager.get_dynamic_target(
                    entry_price, atr_value, position_type
                )
                position['target'] = new_target
                logger.debug(f"Set target for {symbol}: ₹{new_target:.2f}")
    
    def update(self):
        """Update the trading environment
        
        This method should be called periodically to update the environment
        with new market data and process any pending orders or position adjustments.
        """
        # Update clock
        is_market_open = self.update_clock()
        
        if not is_market_open:
            logger.debug(f"Market is closed at {self.current_time}")
            return
        
        # Process pending orders
        self._execute_pending_orders()
        
        # Check stops and targets
        self.check_stops_and_targets()
        
        # Update stops and targets
        self.update_stops_and_targets()
        
        # Update net worth
        self._update_net_worth()
    
    def get_portfolio_summary(self):
        """Get summary of current portfolio"""
        # Get current prices for positions
        position_values = {}
        unrealized_pnl = {}
        unrealized_pnl_pct = {}
        
        for symbol, position in self.positions.items():
            if symbol in self.current_data:
                current_price = self.current_data[symbol]['Close']
                position_values[symbol] = current_price * position['quantity']
                unrealized_pnl[symbol] = (current_price - position['avg_price']) * position['quantity']
                unrealized_pnl_pct[symbol] = (current_price - position['avg_price']) / position['avg_price']
            else:
                position_values[symbol] = position['value']
                unrealized_pnl[symbol] = 0
                unrealized_pnl_pct[symbol] = 0
        
        # Calculate total values
        total_position_value = sum(position_values.values())
        total_unrealized_pnl = sum(unrealized_pnl.values())
        
        # Get current portfolio value
        portfolio_value = self.current_balance + total_position_value
        
        # Calculate returns
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        
        # Calculate cash ratio
        cash_ratio = self.current_balance / portfolio_value if portfolio_value > 0 else 1.0
        
        # Get positions as list
        positions_list = []
        for symbol, position in self.positions.items():
            current_price = self.current_data[symbol]['Close'] if symbol in self.current_data else 0
            days_held = (self.current_date - position['entry_date']).days
            
            positions_list.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'value': position['quantity'] * current_price,
                'unrealized_pnl': (current_price - position['avg_price']) * position['quantity'],
                'unrealized_pnl_pct': (current_price - position['avg_price']) / position['avg_price'] if position['avg_price'] > 0 else 0,
                'days_held': days_held,
                'stop_loss': position.get('stop_loss'),
                'target': position.get('target')
            })
        
        # Sort by value (largest first)
        positions_list.sort(key=lambda x: x['value'], reverse=True)
        
        return {
            'date': self.current_date,
            'time': self.current_time,
            'portfolio_value': portfolio_value,
            'current_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'total_position_value': total_position_value,
            'total_unrealized_pnl': total_unrealized_pnl,
            'total_return': total_return,
            'cash_ratio': cash_ratio,
            'position_count': len(self.positions),
            'positions': positions_list,
            'daily_trades': self.risk_manager.daily_trades,
            'max_drawdown': self.risk_manager.max_drawdown
        }
    
    def get_performance_metrics(self):
        """Calculate detailed performance metrics"""
        # Calculate daily returns
        returns = []
        for i in range(1, len(self.net_worth_history)):
            returns.append((self.net_worth_history[i] - self.net_worth_history[i-1]) / self.net_worth_history[i-1])
        
        # Calculate metrics
        total_return = (self.net_worth_history[-1] - self.initial_balance) / self.initial_balance
        
        # Annualized return (assuming 252 trading days in a year)
        days = len(self.daily_stats)
        annual_return = (1 + total_return) ** (252 / max(days, 1)) - 1 if days > 0 else 0
        
        # Calculate daily returns std and Sharpe ratio
        std_daily_return = np.std(returns) if returns else 0
        risk_free_rate = 0.05  # 5% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        sharpe_ratio = ((np.mean(returns) - daily_risk_free) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate max drawdown
        max_dd = self.risk_manager.max_drawdown
        
        # Calculate win rate and stats
        win_count = sum(1 for trade in self.trades if trade.get('action') == 'SELL' and trade.get('pnl', 0) > 0)
        loss_count = sum(1 for trade in self.trades if trade.get('action') == 'SELL' and trade.get('pnl', 0) <= 0)
        
        total_completed_trades = win_count + loss_count
        win_rate = win_count / total_completed_trades if total_completed_trades > 0 else 0
        
        # Calculate total profit and loss
        total_profit = sum(trade.get('pnl', 0) for trade in self.trades if trade.get('action') == 'SELL' and trade.get('pnl', 0) > 0)
        total_loss = sum(-trade.get('pnl', 0) for trade in self.trades if trade.get('action') == 'SELL' and trade.get('pnl', 0) < 0)
        
        # Average profit/loss
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
        
        # Risk-adjusted return - Calmar Ratio (Annual Return / Max Drawdown)
        calmar_ratio = annual_return / max_dd if max_dd > 0 else float('inf') if annual_return > 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'total_trades': total_completed_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': total_return / max_dd if max_dd > 0 else float('inf') if total_return > 0 else 0,
            'avg_win_loss_ratio': avg_profit / avg_loss if avg_loss > 0 else float('inf') if avg_profit > 0 else 0,
            'portfolio_value': self.net_worth_history[-1],
            'current_balance': self.current_balance,
        }
    
    def save_state(self, file_path):
        """Save current state to a file"""
        state = {
            'current_balance': self.current_balance,
            'current_date': self.current_date.isoformat(),
            'current_time': self.current_time.isoformat(),
            'positions': self.positions,
            'net_worth_history': self.net_worth_history,
            'trades': self.trades,
            'daily_stats': self.daily_stats,
            'risk_manager': {
                'max_drawdown': self.risk_manager.max_drawdown,
                'consecutive_wins': self.risk_manager.consecutive_wins,
                'consecutive_losses': self.risk_manager.consecutive_losses,
                'daily_trades': self.risk_manager.daily_trades,
                'position_size_pct': self.risk_manager.position_size_pct,
                'drawdown_reduction_applied': self.risk_manager.drawdown_reduction_applied
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=4, default=str)
        
        logger.info(f"State saved to {file_path}")
    
    def load_state(self, file_path):
        """Load state from a file"""
        with open(file_path, 'r') as f:
            state = json.load(f)
        
        self.current_balance = state['current_balance']
        self.current_date = datetime.fromisoformat(state['current_date'])
        self.current_time = datetime.fromisoformat(state['current_time']).time()
        self.positions = state['positions']
        self.net_worth_history = state['net_worth_history']
        self.trades = state['trades']
        self.daily_stats = state['daily_stats']
        
        # Restore risk manager state
        risk_state = state['risk_manager']
        self.risk_manager.max_drawdown = risk_state['max_drawdown']
        self.risk_manager.consecutive_wins = risk_state['consecutive_wins']
        self.risk_manager.consecutive_losses = risk_state['consecutive_losses']
        self.risk_manager.daily_trades = risk_state['daily_trades']
        self.risk_manager.position_size_pct = risk_state['position_size_pct']
        self.risk_manager.drawdown_reduction_applied = risk_state['drawdown_reduction_applied']
        
        logger.info(f"State loaded from {file_path}")
        
        # Update risk manager with current date
        self.risk_manager.check_new_day(self.current_date)