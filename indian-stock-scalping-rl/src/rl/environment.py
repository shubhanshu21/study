"""
Gymnasium environment for RL-based stock trading.
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
from ..trading.costs import TradingCosts
from ..trading.order_manager import OrderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ScalpingEnv")


class StockScalpingEnv(gym.Env):
    """Custom Gymnasium Environment for Indian Stock Market Scalping"""
    
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
        self.is_training = is_training
        self.render_mode = render_mode
        self.commission_rate = commission_rate
        
        # Live trading components
        self.data_fetcher = data_fetcher
        self.order_manager = order_manager
        
        # Portfolio state
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.total_pnl = 0
        self.total_trades = 0
        self.successful_trades = 0
        self.max_drawdown = 0
        self.max_balance = initial_balance
        
        # Episode state
        self.current_step = 0
        self.done = False
        self.steps_beyond_done = None
        
        # Price data
        if data is not None:
            self.data = data.copy()
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
        
        # Define action and observation space
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Features (normalized) + position info
        num_features = 5 * window_size + 3  # OHLCV x window_size + position indicators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32)
        
        # Logging
        self.trades = []
        self.rewards_history = []
        self.balance_history = []
        self.equity_history = []
        
        # Stats for early stopping
        self.consecutive_losses = 0
        
        # Initialize the environment
        self.reset()
        
        logger.info(f"Initialized environment for {symbol} in {'training' if is_training else 'trading'} mode")
        
    def _next_observation(self) -> np.ndarray:
        """Get the next observation"""
        if self.is_training:
            # Use historical data
            frame = np.array([])
            
            if self.current_step + self.window_size < len(self.prices):
                # Extract window of OHLCV data
                ohlcv_window = self.data.iloc[self.current_step:self.current_step + self.window_size]
                
                # Normalize within the window
                open_prices = ohlcv_window['open'].values / ohlcv_window['open'].values[0] - 1
                high_prices = ohlcv_window['high'].values / ohlcv_window['open'].values[0] - 1
                low_prices = ohlcv_window['low'].values / ohlcv_window['open'].values[0] - 1
                close_prices = ohlcv_window['close'].values / ohlcv_window['open'].values[0] - 1
                # volumes = ohlcv_window['volume'].values / ohlcv_window['volume'].values[0] - 1
                volumes = (ohlcv_window['volume'].values / (ohlcv_window['volume'].values[0] + 1e-6)) - 1
                
                # Concatenate into a single array
                frame = np.concatenate([
                    open_prices, high_prices, low_prices, close_prices, volumes
                ])
            else:
                # Handle end of data
                frame = np.zeros(5 * self.window_size)
                
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
                    # Use the most recent window
                    ohlcv_window = recent_data.iloc[-self.window_size:]
                    
                    # Normalize within the window
                    open_prices = ohlcv_window['open'].values / ohlcv_window['open'].values[0] - 1
                    high_prices = ohlcv_window['high'].values / ohlcv_window['open'].values[0] - 1
                    low_prices = ohlcv_window['low'].values / ohlcv_window['open'].values[0] - 1
                    close_prices = ohlcv_window['close'].values / ohlcv_window['open'].values[0] - 1
                    # volumes = ohlcv_window['volume'].values / ohlcv_window['volume'].values[0] - 1
                    volumes = (ohlcv_window['volume'].values / (ohlcv_window['volume'].values[0] + 1e-6)) - 1
                    
                    # Concatenate into a single array
                    frame = np.concatenate([
                        open_prices, high_prices, low_prices, close_prices, volumes
                    ])
                else:
                    # Not enough data
                    frame = np.zeros(5 * self.window_size)
            else:
                # No data fetcher
                frame = np.zeros(5 * self.window_size)
        
        # Add position indicators
        position_indicators = np.array([
            1 if self.position > 0 else 0,  # Long indicator
            1 if self.position < 0 else 0,  # Short indicator
            self.entry_price / self.current_price() - 1 if self.position != 0 else 0  # Entry price ratio
        ])
        
        # Concatenate everything
        obs = np.concatenate([frame, position_indicators])
        return obs.astype(np.float32)
        
    def current_price(self) -> float:
        """Get current price"""
        if self.is_training:
            return self.prices[self.current_step + self.window_size - 1]
        else:
            if self.data_fetcher:
                market_data = self.data_fetcher.get_last_data(self.symbol)
                if market_data and 'last_price' in market_data:
                    return market_data['last_price']
            
            # Fallback
            return 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take an action in the environment.
        
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
            
        # Calculate previous portfolio value
        prev_position_value = self.position * current_price
        prev_portfolio_value = self.balance + prev_position_value
        
        # Initialize reward
        reward = 0.0
        transaction_costs = 0.0
        info = {}
        
        # Execute action
        # Action: 0 = Hold, 1 = Buy, 2 = Sell
        if action == 1:  # Buy
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
                    else:
                        self.consecutive_losses += 1
                
                # Calculate buy quantity
                max_shares = int(self.balance * 0.95 / current_price)  # Use 95% of balance
                buy_qty = max(1, min(10, max_shares))  # Min 1, max 10 shares or max affordable
                
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
                        reward += 0.1
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
                    else:
                        self.consecutive_losses += 1
                    
                    # Reward based on profit
                    if long_pnl > 0:
                        # Positive reward for profit
                        profit_pct = long_pnl / self.entry_price
                        reward += profit_pct * 10  # Scale reward by profit percentage
                    else:
                        # Negative reward for loss
                        loss_pct = -long_pnl / self.entry_price
                        reward -= loss_pct * 10  # Scale penalty by loss percentage
                    
                    # Reset position
                    self.position = 0
                    self.entry_price = 0
                
                # Open a short position
                if self.is_training:
                    # In training, we can simulate shorting
                    # Calculate short quantity
                    max_shares = int(self.balance * 0.95 / current_price)  # Use 95% of balance
                    short_qty = max(1, min(10, max_shares))  # Min 1, max 10 shares or max affordable
                    
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
                            reward += 0.1
                        else:
                            # Not enough margin
                            info['error'] = "Insufficient margin for short"
                    else:
                        # Invalid quantity
                        info['error'] = "Invalid short quantity"
                else:
                    # In live trading, check if shorting is allowed
                    # Most retail traders in India cannot do intraday shorting on all stocks
                    # So we'll not implement shorting for live trading
                    pass
            else:
                # Already short, no action
                info['warning'] = "Already short position"
                
        else:  # Hold (action == 0)
            # Calculate unrealized P&L
            if self.position > 0:
                # Long position
                unrealized_pnl = (current_price - self.entry_price) * self.position
                # Small reward/penalty based on price movement
                price_change_pct = (current_price / self.entry_price) - 1
                reward += price_change_pct * 0.1  # Smaller reward for unrealized gains
            elif self.position < 0:
                # Short position
                unrealized_pnl = (self.entry_price - current_price) * abs(self.position)
                # Small reward/penalty based on price movement
                price_change_pct = 1 - (current_price / self.entry_price)
                reward += price_change_pct * 0.1  # Smaller reward for unrealized gains
            else:
                # No position
                unrealized_pnl = 0
                # Small penalty for staying idle (encourage trading)
                reward -= 0.01
        
        # Update current step
        self.current_step += 1
        
        # Calculate current portfolio value
        position_value = self.position * current_price
        portfolio_value = self.balance + position_value
        
        # Update max balance
        if portfolio_value > self.max_balance:
            self.max_balance = portfolio_value
        
        # Calculate drawdown
        drawdown = (self.max_balance - portfolio_value) / self.max_balance if self.max_balance > 0 else 0
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            
        # Penalize for high drawdown
        if drawdown > 0.05:  # More than 5% drawdown
            reward -= drawdown * 5  # Penalize based on drawdown size
        
        # Additional penalty for consecutive losses
        if self.consecutive_losses >= 3:
            reward -= self.consecutive_losses * 0.2  # Increasing penalty for streak of losses
        
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
            elif drawdown > 0.15:  # Stop if drawdown exceeds 15%
                self.done = True
                reward -= 5  # Extra penalty
        
        truncated = False
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
            'success_rate': self.successful_trades / self.total_trades if self.total_trades > 0 else 0
        })
        
        return self._next_observation(), reward, self.done, truncated, info
        
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
        self.total_trades = 0
        self.successful_trades = 0
        self.consecutive_losses = 0
        
        # Reset episode state
        self.current_step = 0
        self.done = False
        self.steps_beyond_done = None
        
        # Reset histories
        self.trades = []
        self.rewards_history = []
        self.balance_history = []
        self.equity_history = []
        
        # If live trading, ensure data connection
        if not self.is_training and self.data_fetcher:
            if not self.data_fetcher.is_connected:
                self.data_fetcher.authenticate()
                
            # Subscribe to symbol
            self.data_fetcher.subscribe([self.symbol])
        
        return self._next_observation(), {}
        
    def render(self, mode='human'):
        """Render the environment"""
        if mode != 'human':
            return
            
        # Print current state
        print(f"Step: {self.current_step}")
        print(f"Date: {self.get_current_datetime()}")
        print(f"Price: {self.current_price()}")
        print(f"Balance: {self.balance}")
        print(f"Position: {self.position}")
        print(f"P&L: {self.total_pnl}")
        print(f"Drawdown: {self.max_drawdown * 100:.2f}%")
        print(f"Trades: {self.total_trades}")
        print(f"Success Rate: {self.successful_trades / self.total_trades * 100:.2f}%" if self.total_trades > 0 else "Success Rate: N/A")
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