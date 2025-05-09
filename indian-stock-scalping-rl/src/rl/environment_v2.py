"""
Improved Gymnasium environment for RL-based stock trading.
Enhanced with better reward function, risk management, and feature engineering.
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import logging

from ..constants import IndianMarketConstants
from ..data.data_fetcher import DataFetcher
from ..data.indicators import TechnicalIndicators
from ..trading.costs import TradingCosts
from ..trading.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScalpingEnv")


class StockScalpingEnv(gym.Env):
    """Enhanced Gymnasium Environment for Indian Stock Market Scalping"""
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, symbol: str, data: pd.DataFrame = None, initial_balance: float = 100000.0,
                 window_size: int = 10, commission_rate: float = 0.0003,
                 is_training: bool = True, render_mode: str = None,
                 data_fetcher: DataFetcher = None, order_manager: OrderManager = None):
        """
        Initialize the environment.
        
        Args:
            symbol: Stock symbol to trade
            data: Historical price data (required for training)
            initial_balance: Starting account balance
            window_size: Size of the observation window
            commission_rate: Commission rate as a decimal
            is_training: Whether in training mode or live trading mode
            render_mode: Rendering mode
            data_fetcher: Data fetcher for live data
            order_manager: Order manager for live trading
        """
        super(StockScalpingEnv, self).__init__()
        
        self.symbol = symbol
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_balance_today = initial_balance
        self.is_training = is_training
        self.render_mode = render_mode
        self.commission_rate = commission_rate
        
        # Live trading components
        self.data_fetcher = data_fetcher
        self.order_manager = order_manager
        
        # Enhanced risk management parameters
        self.max_position_pct = 0.05  # Maximum 5% of capital per position
        self.daily_loss_limit_pct = 0.02  # Maximum 2% daily loss limit
        self.max_drawdown_limit = 0.15  # Maximum 15% drawdown limit
        self.trade_cooldown_steps = 3  # Minimum steps between trades
        self.last_trade_step = -999  # Last step when a trade was made
        
        # Portfolio state
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.max_drawdown = 0
        self.max_balance = initial_balance
        
        # Daily tracking
        self.daily_pnl = {}  # date -> pnl
        self.current_date = None
        
        # Episode state
        self.current_step = 0
        self.done = False
        self.steps_beyond_done = None
        
        # Price data
        if data is not None:
            self.data = data.copy()
            
            # Add technical indicators
            self._add_technical_indicators()
            
            self.prices = self.data['close'].values
            self.dates = self.data.index
            self.episode_length = len(self.data) - window_size
        elif not is_training and data_fetcher:
            # Live mode - will fetch data in real-time
            self.data = pd.DataFrame()
            self.prices = np.array([])
            self.dates = pd.DatetimeIndex([])
            self.episode_length = IndianMarketConstants.MAX_STEPS
        else:
            raise ValueError("Either historical data or data fetcher must be provided")
            
        # Feature extraction (will be filled in reset)
        self.features = None
        
        # Calculate feature dimension based on added indicators
        if data is not None:
            self.feature_columns = ['open', 'high', 'low', 'close', 'volume',
                                   'rsi', 'ma_fast', 'ma_slow', 'bb_upper', 'bb_lower']
            num_features = len(self.feature_columns) * window_size + 6  # Features + position info + time features
        else:
            # Default if no data available yet
            num_features = 5 * window_size + 6
            
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Features (normalized) + position info + time features
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        # Logging
        self.trades = []
        self.rewards_history = []
        self.balance_history = []
        self.equity_history = []
        self.action_history = []
        
        # Stats for early stopping
        self.consecutive_losses = 0
        
        # Initialize the environment
        self.reset()
        
        logger.info(f"Initialized environment for {symbol} in {'training' if is_training else 'trading'} mode")
        
    def _add_technical_indicators(self):
        """Add enhanced technical indicators to the data"""
        # Existing indicators
        self.data['ma_fast'] = self.data['close'].rolling(window=5).mean()
        self.data['ma_slow'] = self.data['close'].rolling(window=20).mean()
        
        # RSI
        close_diff = self.data['close'].diff()
        gain = close_diff.where(close_diff > 0, 0)
        loss = -close_diff.where(close_diff < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        std = self.data['close'].rolling(window=20).std()
        self.data['bb_upper'] = self.data['ma_slow'] + 2 * std
        self.data['bb_lower'] = self.data['ma_slow'] - 2 * std
        
        # --- ADD NEW INDICATORS ---
        
        # Add ATR for volatility-based position sizing and stops
        high_low = self.data['high'] - self.data['low']
        high_close = abs(self.data['high'] - self.data['close'].shift())
        low_close = abs(self.data['low'] - self.data['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.data['atr'] = true_range.rolling(14).mean()
        
        # Volume filters
        self.data['volume_sma'] = self.data['volume'].rolling(10).mean()
        self.data['volume_ratio'] = self.data['volume'] / self.data['volume_sma']
        
        # Trend strength indicator (ADX)
        plus_dm = self.data['high'].diff()
        minus_dm = self.data['low'].diff(-1).abs()
        plus_dm = plus_dm.where(((plus_dm > 0) & (plus_dm > minus_dm)), 0)
        minus_dm = minus_dm.where(((minus_dm > 0) & (minus_dm > plus_dm)), 0)
        
        tr = true_range
        plus_di = 100 * (plus_dm.rolling(14).mean() / tr.rolling(14).mean())
        minus_di = 100 * (minus_dm.rolling(14).mean() / tr.rolling(14).mean())
        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).abs())
        self.data['adx'] = dx.rolling(14).mean()
        
        # MACD
        self.data['ema12'] = self.data['close'].ewm(span=12, adjust=False).mean()
        self.data['ema26'] = self.data['close'].ewm(span=26, adjust=False).mean()
        self.data['macd'] = self.data['ema12'] - self.data['ema26']
        self.data['macd_signal'] = self.data['macd'].ewm(span=9, adjust=False).mean()
        self.data['macd_hist'] = self.data['macd'] - self.data['macd_signal']
        
        # Price action patterns
        self.data['body_size'] = abs(self.data['open'] - self.data['close'])
        self.data['upper_shadow'] = self.data['high'] - self.data[['open', 'close']].max(axis=1)
        self.data['lower_shadow'] = self.data[['open', 'close']].min(axis=1) - self.data['low']
        
        # Market session time features
        if isinstance(self.data.index, pd.DatetimeIndex):
            self.data['hour'] = self.data.index.hour
            self.data['minute'] = self.data.index.minute
            # Convert to total minutes from market open
            market_open_hour = IndianMarketConstants.MARKET_OPEN_TIME.hour
            market_open_minute = IndianMarketConstants.MARKET_OPEN_TIME.minute
            self.data['session_minute'] = (self.data['hour'] - market_open_hour) * 60 + (self.data['minute'] - market_open_minute)
        
        # Fill NaN values
        self.data = self.data.fillna(method='bfill')

    def _check_entry_conditions(self, action, current_step):
        """Check if entry conditions are met"""
        if current_step + self.window_size >= len(self.data):
            return False
            
        # Get current values
        curr_idx = current_step + self.window_size - 1
        
        # Basic data
        price = self.data['close'].iloc[curr_idx]
        rsi = self.data['rsi'].iloc[curr_idx]
        volume_ratio = self.data['volume_ratio'].iloc[curr_idx]
        adx = self.data['adx'].iloc[curr_idx]
        macd = self.data['macd'].iloc[curr_idx]
        macd_signal = self.data['macd_signal'].iloc[curr_idx]
        macd_hist = self.data['macd_hist'].iloc[curr_idx]
        session_minute = self.data['session_minute'].iloc[curr_idx] if 'session_minute' in self.data.columns else None
        
        # Common filters for both long and short entries
        # 1. Sufficient volume
        if volume_ratio < 0.8:  # Require at least 80% of average volume
            return False
            
        # 2. Strong enough trend
        if adx < 20:  # ADX below 20 indicates weak trend
            return False
            
        # 3. Avoid trading in first 15 minutes and last 15 minutes
        if session_minute is not None:
            market_duration = (IndianMarketConstants.MARKET_CLOSE_TIME.hour - IndianMarketConstants.MARKET_OPEN_TIME.hour) * 60 + \
                              (IndianMarketConstants.MARKET_CLOSE_TIME.minute - IndianMarketConstants.MARKET_OPEN_TIME.minute)
            if session_minute < 15 or session_minute > (market_duration - 15):
                return False
        
        # Action-specific checks
        if action == 1:  # Buy/Long entry
            # 1. RSI conditions - not overbought
            if rsi > 70:
                return False
                
            # 2. MACD bullish crossover or positive momentum
            if macd_hist <= 0 or (macd < macd_signal):
                return False
                
            # 3. Price above short-term MA
            if price < self.data['ma_fast'].iloc[curr_idx]:
                return False
                
        elif action == 2:  # Sell/Short entry
            # 1. RSI conditions - not oversold
            if rsi < 30:
                return False
                
            # 2. MACD bearish crossover or negative momentum
            if macd_hist >= 0 or (macd > macd_signal):
                return False
                
            # 3. Price below short-term MA
            if price > self.data['ma_fast'].iloc[curr_idx]:
                return False
        
        # All conditions passed
        return True
    
    def _calculate_position_size(self, price, action):
        """Calculate optimal position size based on current volatility and risk"""
        # Default position sizing (as before)
        max_shares = int(self.balance * 0.95 / price)
        position_size_limit = int(self.balance * self.max_position_pct / price)
        
        # If ATR-based position sizing is enabled
        if hasattr(self, 'position_sizing_atr') and self.position_sizing_atr:
            curr_idx = self.current_step + self.window_size - 1
            if curr_idx < len(self.data) and 'atr' in self.data.columns:
                current_atr = self.data['atr'].iloc[curr_idx]
                
                # Risk 0.5% of account per trade
                risk_amount = self.balance * 0.005
                
                # Use 2 x ATR as stop distance
                stop_distance = current_atr * 2
                
                # Calculate position size based on stop
                if stop_distance > 0:
                    risk_based_size = int(risk_amount / stop_distance)
                    # Use the most conservative position size
                    return max(1, min(5, max_shares, position_size_limit, risk_based_size))
        
        # Default if ATR not available
        return max(1, min(5, max_shares, position_size_limit))
    
    def _should_exit_position(self, current_price):
        """Determine if we should exit the current position based on various factors"""
        if self.position == 0:
            return False
            
        curr_idx = self.current_step + self.window_size - 1
        if curr_idx >= len(self.data):
            return False
            
        # Calculate unrealized P&L
        if self.position > 0:  # Long position
            # Calculate P&L percentage
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            # Take profit at 1.5% gain
            if pnl_pct >= 0.015:
                return True
                
            # Stop loss at 0.7% loss
            if pnl_pct <= -0.007:
                return True
                
            # Time-based exit - exit after 10 periods (50 minutes for 5-min data)
            if (self.current_step - self.last_trade_step) > 10:
                return True
                
            # Technical exit for long - RSI overbought or MACD turning down
            rsi = self.data['rsi'].iloc[curr_idx]
            macd_hist = self.data['macd_hist'].iloc[curr_idx]
            prev_macd_hist = self.data['macd_hist'].iloc[curr_idx-1] if curr_idx > 0 else 0
            
            if rsi > 75 or (macd_hist < 0 and prev_macd_hist > 0):
                return True
                
        elif self.position < 0:  # Short position
            # Calculate P&L percentage
            pnl_pct = (self.entry_price - current_price) / self.entry_price
            
            # Take profit at 1.5% gain
            if pnl_pct >= 0.015:
                return True
                
            # Stop loss at 0.7% loss
            if pnl_pct <= -0.007:
                return True
                
            # Time-based exit - exit after 10 periods (50 minutes for 5-min data)
            if (self.current_step - self.last_trade_step) > 10:
                return True
                
            # Technical exit for short - RSI oversold or MACD turning up
            rsi = self.data['rsi'].iloc[curr_idx]
            macd_hist = self.data['macd_hist'].iloc[curr_idx]
            prev_macd_hist = self.data['macd_hist'].iloc[curr_idx-1] if curr_idx > 0 else 0
            
            if rsi < 25 or (macd_hist > 0 and prev_macd_hist < 0):
                return True
        
        return False
        
    def _next_observation(self) -> np.ndarray:
        """Get the next observation with enhanced features"""
        if self.is_training:
            # Use historical data
            frame = np.array([])
            
            if self.current_step + self.window_size < len(self.prices):
                # Extract window of data with indicators
                data_window = self.data.iloc[self.current_step:self.current_step + self.window_size]
                
                # Extract and normalize all features
                feature_list = []
                for col in self.feature_columns:
                    # Normalize to percentage change from first value
                    values = data_window[col].values
                    normalized = values / values[0] - 1
                    feature_list.append(normalized)
                
                # Concatenate all features
                frame = np.concatenate(feature_list)
                
                # Add time features
                current_time = self.get_current_datetime().time()
                
                # Normalize time to [0, 1] within trading hours
                market_open_seconds = IndianMarketConstants.MARKET_OPEN_TIME.hour * 3600 + IndianMarketConstants.MARKET_OPEN_TIME.minute * 60
                market_close_seconds = IndianMarketConstants.MARKET_CLOSE_TIME.hour * 3600 + IndianMarketConstants.MARKET_CLOSE_TIME.minute * 60
                current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
                
                # Time features
                time_of_day = (current_seconds - market_open_seconds) / (market_close_seconds - market_open_seconds)
                near_open = np.exp(-((time_of_day - 0) ** 2) / 0.01)  # Gaussian near open
                near_close = np.exp(-((time_of_day - 1) ** 2) / 0.01)  # Gaussian near close
                midday = np.exp(-((time_of_day - 0.5) ** 2) / 0.01)  # Gaussian at midday
                
            else:
                # Handle end of data
                frame = np.zeros(len(self.feature_columns) * self.window_size)
                time_of_day = 0
                near_open = 0
                near_close = 0
                midday = 0
                
        else:
            # Live trading - use recent candlesticks
            if self.data_fetcher:
                # Get last N candles
                recent_data = self.data_fetcher.fetch_historical_data(
                    self.symbol, interval='5minute', 
                    from_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                    to_date=datetime.now().strftime('%Y-%m-%d')
                )
                
                if len(recent_data) >= self.window_size:
                    # Add indicators to recent data
                    recent_data['ma_fast'] = recent_data['close'].rolling(window=5).mean()
                    recent_data['ma_slow'] = recent_data['close'].rolling(window=20).mean()
                    
                    # RSI
                    close_diff = recent_data['close'].diff()
                    gain = close_diff.where(close_diff > 0, 0)
                    loss = -close_diff.where(close_diff < 0, 0)
                    avg_gain = gain.rolling(window=14).mean()
                    avg_loss = loss.rolling(window=14).mean()
                    rs = avg_gain / avg_loss
                    recent_data['rsi'] = 100 - (100 / (1 + rs))
                    
                    # Bollinger Bands
                    std = recent_data['close'].rolling(window=20).std()
                    recent_data['bb_upper'] = recent_data['ma_slow'] + 2 * std
                    recent_data['bb_lower'] = recent_data['ma_slow'] - 2 * std
                    
                    # Fill NaN values
                    recent_data = recent_data.fillna(method='bfill')
                    
                    # Use the most recent window
                    data_window = recent_data.iloc[-self.window_size:]
                    
                    # Extract and normalize features
                    feature_list = []
                    for col in ['open', 'high', 'low', 'close', 'volume', 
                               'rsi', 'ma_fast', 'ma_slow', 'bb_upper', 'bb_lower']:
                        if col in data_window.columns:
                            values = data_window[col].values
                            normalized = values / values[0] - 1
                            feature_list.append(normalized)
                        else:
                            # Handle missing columns
                            normalized = np.zeros(self.window_size)
                            feature_list.append(normalized)
                    
                    # Concatenate all features
                    frame = np.concatenate(feature_list)
                    
                    # Add time features
                    current_time = datetime.now().time()
                    
                    # Normalize time to [0, 1] within trading hours
                    market_open_seconds = IndianMarketConstants.MARKET_OPEN_TIME.hour * 3600 + IndianMarketConstants.MARKET_OPEN_TIME.minute * 60
                    market_close_seconds = IndianMarketConstants.MARKET_CLOSE_TIME.hour * 3600 + IndianMarketConstants.MARKET_CLOSE_TIME.minute * 60
                    current_seconds = current_time.hour * 3600 + current_time.minute * 60 + current_time.second
                    
                    # Time features
                    time_of_day = (current_seconds - market_open_seconds) / (market_close_seconds - market_open_seconds)
                    near_open = np.exp(-((time_of_day - 0) ** 2) / 0.01)
                    near_close = np.exp(-((time_of_day - 1) ** 2) / 0.01)
                    midday = np.exp(-((time_of_day - 0.5) ** 2) / 0.01)
                    
                else:
                    # Not enough data
                    frame = np.zeros(len(self.feature_columns) * self.window_size)
                    time_of_day = 0
                    near_open = 0
                    near_close = 0
                    midday = 0
            else:
                # No data fetcher
                frame = np.zeros(len(self.feature_columns) * self.window_size)
                time_of_day = 0
                near_open = 0
                near_close = 0
                midday = 0
        
        # Add position indicators
        position_indicators = np.array([
            1 if self.position > 0 else 0,  # Long indicator
            1 if self.position < 0 else 0,  # Short indicator
            self.entry_price / self.current_price() - 1 if self.position != 0 else 0,  # Entry price ratio
            self.position / 10 if self.position != 0 else 0,  # Position size (normalized)
            time_of_day,  # Time of day [0, 1]
            near_open + near_close  # Near market open/close indicator
        ])
        
        # Concatenate everything
        obs = np.concatenate([frame, position_indicators])
        return obs.astype(np.float32)
        
    def current_price(self) -> float:
        """Get current price"""
        if self.is_training:
            if self.current_step + self.window_size - 1 < len(self.prices):
                return self.prices[self.current_step + self.window_size - 1]
            return 0
        else:
            if self.data_fetcher:
                market_data = self.data_fetcher.get_last_data(self.symbol)
                if market_data and 'last_price' in market_data:
                    return market_data['last_price']
            
            # Fallback
            return 0
    
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment with improved reward function and risk management.
        
        Args:
            action: 0 = Hold, 1 = Buy, 2 = Sell
            
        Returns:
            observation, reward, done, info
        """
        if self.done:
            if self.steps_beyond_done is None:
                self.steps_beyond_done = 0
            else:
                self.steps_beyond_done += 1
                
            return self._next_observation(), 0.0, True, {"warning": "Episode already done"}
        
        # Get current price
        current_price = self.current_price()
        if current_price <= 0:
            return self._next_observation(), -1.0, True, {"error": "Invalid price"}
        
        # Get current date for daily tracking
        current_datetime = self.get_current_datetime()
        current_date = current_datetime.date() if isinstance(current_datetime, datetime) else None
        
        # Update daily tracking
        if current_date != self.current_date:
            # New day - reset daily metrics
            self.current_date = current_date
            if current_date not in self.daily_pnl:
                self.daily_pnl[current_date] = 0
            self.max_balance_today = self.balance + (self.position * current_price)
        
        # Check daily loss limit
        if current_date and current_date in self.daily_pnl:
            daily_pnl = self.daily_pnl[current_date]
            if daily_pnl < -(self.initial_balance * self.daily_loss_limit_pct):
                logger.info(f"Daily loss limit reached: {daily_pnl:.2f}")
                return self._next_observation(), -2.0, True, {"error": "Daily loss limit reached"}
            
        # Calculate previous portfolio value
        prev_position_value = self.position * current_price
        prev_portfolio_value = self.balance + prev_position_value
        
        # Initialize reward
        reward = 0.0
        transaction_costs = 0.0
        info = {}
        
        # Execute action with trade cooldown and improved risk management
        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        
        # Track if action was actually executed
        action_executed = False
        
        # Check cooldown period
        cooldown_active = (self.current_step - self.last_trade_step) < self.trade_cooldown_steps
        
        # Check if we need to force exit based on stop-loss or take-profit
        force_exit = False
        if self.position != 0:
            force_exit = self._should_exit_position(current_price)
            if force_exit:
                # Force exit by setting the appropriate action
                if self.position > 0:
                    action = 2  # Sell to close long
                elif self.position < 0:
                    action = 1  # Buy to close short
                
                # Log forced exit
                logger.debug(f"Forced exit at step {self.current_step}, price: {current_price}")
        
        if cooldown_active and not force_exit:
            # Skip trading during cooldown unless it's a forced exit
            info['cooldown'] = True
            # Small penalty for trying to trade during cooldown
            reward -= 0.05  # Reduced penalty
        elif action == 1:  # Buy
            if self.position <= 0:  # Only buy if not already long
                # Close any short position first
                if self.position < 0:
                    # Calculate P&L and costs for closing short
                    short_pnl = self.entry_price - current_price
                    
                    # Get trading costs
                    if self.is_training:
                        # Simulate costs
                        costs = TradingCosts.calculate_costs(current_price, abs(self.position), 'buy')
                        transaction_costs = costs['total']
                    else:
                        # Real costs from live trading
                        costs = TradingCosts.calculate_costs(current_price, abs(self.position), 'buy')
                        transaction_costs = costs['total']
                    
                    # Update account
                    trade_profit = (short_pnl * abs(self.position)) - transaction_costs
                    self.balance += trade_profit
                    self.total_pnl += trade_profit
                    
                    # Update daily PnL
                    if current_date in self.daily_pnl:
                        self.daily_pnl[current_date] += trade_profit
                    
                    # Log trade
                    self.trades.append({
                        'timestamp': self.get_current_datetime(),
                        'action': 'close_short',
                        'price': current_price,
                        'quantity': abs(self.position),
                        'pnl': trade_profit,
                        'costs': transaction_costs,
                        'balance': self.balance
                    })
                    
                    self.total_trades += 1
                    if trade_profit > 0:
                        self.successful_trades += 1
                        self.consecutive_losses = 0
                        
                        # Reward for successful short trade
                        profit_pct = (short_pnl / self.entry_price) * 100
                        reward += min(profit_pct * 2, 10)  # Cap rewards
                    else:
                        self.consecutive_losses += 1
                        
                        # Smaller penalty for losses when we're closing due to exit rules
                        if force_exit:
                            loss_pct = (abs(short_pnl) / self.entry_price) * 100
                            reward -= loss_pct * 1.5  # Reduced penalty multiplier for managed exits
                    
                    # Reset position
                    self.position = 0
                    self.entry_price = 0
                    
                    # Set action executed
                    action_executed = True
                    self.last_trade_step = self.current_step
                
                # Now check if we should open a long position
                # Only open position if entry conditions are satisfied (unless it's a forced exit)
                if not force_exit and not self._check_entry_conditions(action, self.current_step):
                    info['warning'] = "Entry conditions not met"
                    return self._next_observation(), reward, False, info
                
                # Calculate buy quantity with improved position sizing
                buy_qty = self._calculate_position_size(current_price, action)
                
                if buy_qty > 0 and self.balance > current_price:
                    # Calculate costs
                    if self.is_training:
                        costs = TradingCosts.calculate_costs(current_price, buy_qty, 'buy')
                        transaction_costs = costs['total']
                    else:
                        costs = TradingCosts.calculate_costs(current_price, buy_qty, 'buy')
                        transaction_costs = costs['total']
                    
                    # Update position
                    total_cost = (current_price * buy_qty) + transaction_costs
                    
                    if total_cost <= self.balance:
                        # Execute the buy
                        if not self.is_training and self.order_manager:
                            # Place real order
                            order_result = self.order_manager.place_order(
                                self.symbol, buy_qty, current_price, 'MARKET', 'BUY'
                            )
                            
                            if order_result.get('status') != 'success':
                                info['order_error'] = order_result.get('message', 'Order failed')
                                return self._next_observation(), -1.0, False, info
                        
                        # Update account
                        self.balance -= total_cost
                        self.position = buy_qty
                        self.entry_price = current_price
                        
                        # Log trade
                        self.trades.append({
                            'timestamp': self.get_current_datetime(),
                            'action': 'buy',
                            'price': current_price,
                            'quantity': buy_qty,
                            'costs': transaction_costs,
                            'balance': self.balance
                        })
                        
                        # Small reward for opening a position
                        reward += 0.2  # Increased reward for opening good positions
                        
                        # Set action executed
                        action_executed = True
                        self.last_trade_step = self.current_step
                    else:
                        # Not enough balance
                        info['error'] = "Insufficient balance"
                else:
                    # Not enough balance or zero quantity
                    info['error'] = "Invalid buy quantity"
            else:
                # Already long, no action
                info['warning'] = "Already long position"
                
        elif action == 2:  # Sell
            if self.position >= 0:  # Only sell if not already short
                # Close any long position first
                if self.position > 0:
                    # Calculate P&L and costs for closing long
                    long_pnl = current_price - self.entry_price
                    
                    # Get trading costs
                    if self.is_training:
                        costs = TradingCosts.calculate_costs(current_price, self.position, 'sell')
                        transaction_costs = costs['total']
                    else:
                        costs = TradingCosts.calculate_costs(current_price, self.position, 'sell')
                        transaction_costs = costs['total']
                    
                    # Update account
                    trade_profit = (long_pnl * self.position) - transaction_costs
                    self.balance += (current_price * self.position) - transaction_costs
                    self.total_pnl += trade_profit
                    
                    # Update daily PnL
                    if current_date in self.daily_pnl:
                        self.daily_pnl[current_date] += trade_profit
                    
                    # Log trade
                    self.trades.append({
                        'timestamp': self.get_current_datetime(),
                        'action': 'close_long',
                        'price': current_price,
                        'quantity': self.position,
                        'pnl': trade_profit,
                        'costs': transaction_costs,
                        'balance': self.balance
                    })
                    
                    self.total_trades += 1
                    if trade_profit > 0:
                        self.successful_trades += 1
                        self.consecutive_losses = 0
                        
                        # Reward based on profit with risk adjustment
                        profit_pct = (long_pnl / self.entry_price) * 100
                        reward += min(profit_pct * 2, 10)  # Cap rewards for huge wins
                    else:
                        self.consecutive_losses += 1
                        
                        # Smaller penalty for losses when we're closing due to exit rules
                        if force_exit:
                            loss_pct = (abs(long_pnl) / self.entry_price) * 100
                            reward -= loss_pct * 1.5  # Reduced penalty multiplier for managed exits
                        else:
                            loss_pct = (abs(long_pnl) / self.entry_price) * 100
                            reward -= loss_pct * 2.5  # Higher penalty multiplier for unmanaged losses
                    
                    # Reset position
                    self.position = 0
                    self.entry_price = 0
                    
                    # Set action executed
                    action_executed = True
                    self.last_trade_step = self.current_step
                
                # Open a short position (only in training mode)
                if self.is_training and not force_exit:
                    # Only open position if entry conditions are satisfied
                    if not self._check_entry_conditions(action, self.current_step):
                        info['warning'] = "Entry conditions not met"
                        return self._next_observation(), reward, False, info
                    
                    # In training, we can simulate shorting
                    # Calculate short quantity with improved position sizing
                    short_qty = self._calculate_position_size(current_price, action)
                    
                    if short_qty > 0:
                        # Calculate costs (initial margin for shorting)
                        costs = TradingCosts.calculate_costs(current_price, short_qty, 'sell')
                        transaction_costs = costs['total']
                        
                        # Update position
                        margin_required = (current_price * short_qty) * 0.5  # 50% margin
                        
                        if margin_required <= self.balance:
                            # Execute the short
                            self.balance -= transaction_costs  # Only deduct transaction costs initially
                            self.position = -short_qty
                            self.entry_price = current_price
                            
                            # Log trade
                            self.trades.append({
                                'timestamp': self.get_current_datetime(),
                                'action': 'short',
                                'price': current_price,
                                'quantity': short_qty,
                                'costs': transaction_costs,
                                'balance': self.balance
                            })
                            
                            # Small reward for opening a position
                            reward += 0.2  # Increased reward for opening good positions
                            
                            # Set action executed
                            action_executed = True
                            self.last_trade_step = self.current_step
                        else:
                            # Not enough margin
                            info['error'] = "Insufficient margin for short"
                    else:
                        # Invalid quantity or RSI conditions not met
                        info['error'] = "Invalid short quantity or conditions not met"
                else:
                    # In live trading, check if shorting is allowed
                    # Most retail traders in India cannot do intraday shorting on all stocks
                    # So we'll not implement shorting for live trading
                    pass
            else:
                # Already short, no action
                info['warning'] = "Already short position"
                
        else:  # Hold (action == 0)
            # If we have a position, check for exit conditions
            if self.position != 0 and force_exit:
                info['warning'] = "Should exit but hold action was selected"
                reward -= 0.5  # Penalty for not exiting when conditions indicate exit
            
            # Calculate unrealized P&L
            if self.position > 0:
                # Long position
                unrealized_pnl = (current_price - self.entry_price) * self.position
                # Small reward/penalty based on price movement
                price_change_pct = (current_price / self.entry_price) - 1
                
                # Time decay penalty for holding too long
                holding_penalty = 0.005 * (self.current_step - self.last_trade_step) / 10  # Reduced time decay
                
                reward += price_change_pct * 0.1 - holding_penalty  # Smaller reward for unrealized gains
            elif self.position < 0:
                # Short position
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
                # Small reward/penalty based on price movement
                price_change_pct = 1 - (current_price / self.entry_price)
                
                # Time decay penalty for holding too long
                holding_penalty = 0.005 * (self.current_step - self.last_trade_step) / 10  # Reduced time decay
                
                reward += price_change_pct * 0.1 - holding_penalty  # Smaller reward for unrealized gains
            else:
                # No position - tiny penalty for staying idle but reduced from original
                reward -= 0.002  # Smaller penalty to encourage patience
        
        # Update current step
        self.current_step += 1
        
        # Record action
        self.action_history.append(action)
        
        # Calculate current portfolio value
        position_value = self.position * current_price
        portfolio_value = self.balance + position_value
        
        # Update max balance
        if portfolio_value > self.max_balance:
            self.max_balance = portfolio_value
            
        # Update max balance today
        if portfolio_value > self.max_balance_today:
            self.max_balance_today = portfolio_value
        
        # Calculate drawdown
        drawdown = (self.max_balance - portfolio_value) / self.max_balance if self.max_balance > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        # Calculate daily drawdown
        daily_drawdown = (self.max_balance_today - portfolio_value) / self.max_balance_today if self.max_balance_today > 0 else 0
            
        # Penalize for high drawdown (more severely)
        if drawdown > 0.01:  # More than 1% drawdown (lowered threshold)
            reward -= drawdown * 150  # Much higher penalty for drawdowns
        
        # Additional penalty for consecutive losses
        if self.consecutive_losses >= 2:
            reward -= self.consecutive_losses * 0.7  # Increasing penalty for streak of losses
        
        # Check if episode is done
        self.done = False
        
        # Training mode uses fixed episode length
        if self.is_training:
            if self.current_step >= self.episode_length - 1:
                self.done = True
        else:
            # Live trading checks for market close, max steps, or stop conditions
            if self.current_step >= self.episode_length:
                self.done = True
            elif not self.data_fetcher.is_market_open():
                self.done = True
            elif drawdown > self.max_drawdown_limit:  # Stop if drawdown exceeds limit
                self.done = True
                reward -= 20  # Extra penalty for excessive drawdown
        
        # Record history
        self.rewards_history.append(reward)
        self.balance_history.append(self.balance)
        self.equity_history.append(portfolio_value)
        
        # Prepare info dict
        info.update({
            'step': self.current_step,
            'price': current_price,
            'position': self.position,
            'balance': self.balance,
            'portfolio_value': portfolio_value,
            'transaction_costs': transaction_costs,
            'drawdown': drawdown,
            'max_drawdown': self.max_drawdown,
            'trade_count': self.total_trades,
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            'action_executed': action_executed,
            'consecutive_losses': self.consecutive_losses,
            'force_exit': force_exit
        })
        
        return self._next_observation(), reward, self.done, info

    def reset(self, **kwargs) -> np.ndarray:
        """Reset the environment for a new episode"""
        # Reset portfolio state
        self.balance = self.initial_balance
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.max_drawdown = 0
        self.max_balance = self.initial_balance
        self.max_balance_today = self.initial_balance
        self.total_trades = 0
        self.successful_trades = 0
        self.consecutive_losses = 0
        self.last_trade_step = -999
        
        # Reset daily tracking
        self.daily_pnl = {}
        self.current_date = None
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.steps_beyond_done = None
        
        # Reset histories
        self.trades = []
        self.rewards_history = []
        self.balance_history = []
        self.equity_history = []
        self.action_history = []
        
        # If live trading, ensure data connection
        if not self.is_training and self.data_fetcher:
            if not self.data_fetcher.is_connected:
                self.data_fetcher.authenticate()
                
            # Subscribe to symbol
            self.data_fetcher.subscribe([self.symbol])
            
        # Get the observation
        obs = self._next_observation()
        
        # Handle Gymnasium API which expects a tuple of (obs, info)
        if kwargs.get('return_info', False):
            return obs, {}
        return obs
        
    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            return
            
        # Print current state
        current_datetime = self.get_current_datetime()
        date_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, datetime) else str(current_datetime)
        
        print(f"Step: {self.current_step}")
        print(f"Date: {date_str}")
        print(f"Price: {self.current_price()}")
        print(f"Balance: {self.balance}")
        print(f"Position: {self.position}")
        print(f"P&L: {self.total_pnl}")
        print(f"Drawdown: {self.max_drawdown * 100:.2f}%")
        print(f"Trades: {self.total_trades}")
        print(f"Success Rate: {self.successful_trades / self.total_trades * 100:.2f}%" if self.total_trades > 0 else "Success Rate: N/A")
        
        # If we have a position, show entry price and unrealized P&L
        if self.position != 0:
            unrealized_pnl = (self.current_price() - self.entry_price) * self.position
            print(f"Entry Price: {self.entry_price}")
            print(f"Unrealized P&L: {unrealized_pnl}")
            print(f"Holding for: {self.current_step - self.last_trade_step} steps")
        
        print("-" * 50)
        
    def close(self):
        """Clean up resources"""
        if not self.is_training and self.data_fetcher:
            self.data_fetcher.stop_websocket()
            
    def get_current_datetime(self) -> datetime:
        """Get current datetime"""
        if self.is_training and self.current_step < len(self.dates) - self.window_size:
            return self.dates[self.current_step + self.window_size - 1]
        else:
            return datetime.now()
            
    def get_performance_metrics(self) -> Dict:
        """Calculate and return performance metrics"""
        metrics = {
            'total_trades': self.total_trades,
            'successful_trades': self.successful_trades,
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0,
            'final_balance': self.balance,
            'total_pnl': self.total_pnl,
            'pnl_pct': (self.total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            'max_drawdown': self.max_drawdown,
        }
        
        # Calculate additional metrics if we have enough data
        if len(self.equity_history) > 1:
            # Daily returns
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            metrics['mean_return'] = np.mean(returns) if len(returns) > 0 else 0
            metrics['std_return'] = np.std(returns) if len(returns) > 0 else 0
            
            # Sharpe ratio (annualized)
            risk_free_rate = 0.05 / 252  # Daily risk-free rate (5% annual)
            if metrics['std_return'] > 0:
                metrics['sharpe_ratio'] = (metrics['mean_return'] - risk_free_rate) / metrics['std_return'] * np.sqrt(252)
            else:
                metrics['sharpe_ratio'] = 0
                
            # Winning/losing streaks
            win_streak = lose_streak = current_streak = 0
            prev_profit = None
            
            for trade in self.trades:
                if 'pnl' in trade:
                    profit = trade['pnl'] > 0
                    
                    if prev_profit is None:
                        current_streak = 1
                    elif profit == prev_profit:
                        current_streak += 1
                    else:
                        current_streak = 1
                        
                    if profit:
                        win_streak = max(win_streak, current_streak)
                    else:
                        lose_streak = max(lose_streak, current_streak)
                        
                    prev_profit = profit
            
            metrics['max_win_streak'] = win_streak
            metrics['max_lose_streak'] = lose_streak
            
            # Average profit/loss
            profits = [trade['pnl'] for trade in self.trades if 'pnl' in trade and trade['pnl'] > 0]
            losses = [trade['pnl'] for trade in self.trades if 'pnl' in trade and trade['pnl'] <= 0]
            
            metrics['avg_profit'] = np.mean(profits) if profits else 0
            metrics['avg_loss'] = np.mean(losses) if losses else 0
            metrics['profit_factor'] = abs(sum(profits) / sum(losses)) if sum(losses) != 0 else float('inf')
            
            # Trade frequency
            if len(self.trades) > 1:
                first_trade = self.trades[0]['timestamp'] if 'timestamp' in self.trades[0] else None
                last_trade = self.trades[-1]['timestamp'] if 'timestamp' in self.trades[-1] else None
                
                if first_trade and last_trade and isinstance(first_trade, datetime) and isinstance(last_trade, datetime):
                    days_diff = (last_trade - first_trade).days + 1
                    if days_diff > 0:
                        metrics['trades_per_day'] = len(self.trades) / days_diff
                    else:
                        metrics['trades_per_day'] = len(self.trades)
                else:
                    metrics['trades_per_day'] = len(self.trades) / max(1, self.current_step / 78)  # Assuming 78 5-min bars per day
            else:
                metrics['trades_per_day'] = 0
        
        return metrics
