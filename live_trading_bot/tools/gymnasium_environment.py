import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging

logger = logging.getLogger(__name__)

class GymTradingEnvironment(gym.Env):
    """
    Trading environment for reinforcement learning using Gymnasium interface.
    Simulates the trading process for model training and evaluation.
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, symbol, data, config, initial_balance=None, window_size=None):
        """
        Initialize the trading environment
        
        Args:
            symbol (str): Trading symbol
            data (pandas.DataFrame): DataFrame with OHLCV and feature data
            config: Application configuration
            initial_balance (float, optional): Initial balance
            window_size (int, optional): Window size for observations
        """
        super(GymTradingEnvironment, self).__init__()
        
        self.symbol = symbol
        self.df = data
        self.config = config
        self.initial_balance = initial_balance or getattr(config, 'DEFAULT_INITIAL_BALANCE', 100000)
        self.window_size = window_size or getattr(config, 'WINDOW_SIZE', 20)
        
        # Trading parameters
        self.max_position_size_pct = float(getattr(config, 'MAX_POSITION_SIZE_PCT', 0.4))
        self.trailing_stop_atr_multiplier = float(getattr(config, 'TRAILING_STOP_ATR_MULTIPLIER', 2.0))
        self.target_atr_multiplier = float(getattr(config, 'TARGET_ATR_MULTIPLIER', 3.0))
        self.min_rr_ratio = float(getattr(config, 'MIN_RR_RATIO', 1.5))
        
        # Define action and observation spaces
        # Actions: 0 = Hold, 1 = Buy with 25% capital, 2 = Buy with 50% capital, 
        # 3 = Buy with 75% capital, 4 = Buy with 100% capital, 5 = Sell all
        self.action_space = spaces.Discrete(6)
        
        # Observation space: balance, shares, cost_basis, current_price, technical indicators, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32
        )
        
        # Reset account variables
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = self.initial_balance
        
        # Position management
        self.stop_loss = 0
        self.trailing_stop = 0
        self.initial_stop_loss = 0
        self.target_price = 0
        self.highest_price_since_buy = 0
        self.position_type = None
        self.last_txn_cost = 0
        self.entry_date = None
        self.entry_time = None
        
        # Enhanced risk management features
        self.daily_trades = 0
        self.daily_high_capital = initial_balance or getattr(config, 'DEFAULT_INITIAL_BALANCE', 100000)
        self.daily_low_capital = initial_balance or getattr(config, 'DEFAULT_INITIAL_BALANCE', 100000)
        self.trading_stopped_for_day = False
        self.trading_stopped_reason = None
        self.drawdown_reduction_applied = False
        self.position_size_pct = self.max_position_size_pct
        self.days_in_trade = 0
        self.current_date = None
        self.previous_date = None
        self.daily_pnl = 0
        self.current_market_regime = "unknown"
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.daily_profit = 0
        self.daily_loss = 0
        
        # Multiple position tracking
        self.open_positions_count = 0

        # Trading signals
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Episode variables
        self.current_step = self.window_size
        self.rewards = []
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        
        logger.info(f"Trading environment initialized for {symbol} with {len(data)} data points")
    
    def reset(self, *, seed=None, options=None):
        """
        Reset the environment
        
        Args:
            seed (int, optional): Random seed
            options (dict, optional): Additional options
            
        Returns:
            tuple: (observation, info)
        """
        super().reset(seed=seed)
        
        # Reset account variables
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = self.initial_balance
        
        # Reset position management
        self.stop_loss = 0
        self.trailing_stop = 0
        self.target_price = 0
        self.highest_price_since_buy = 0
        self.position_type = None
        self.initial_stop_loss = 0
        self.entry_date = None
        self.entry_time = None
        self.days_in_trade = 0
        
        # Reset trading indicators
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Reset episode variables
        self.current_step = self.window_size
        self.rewards = []
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        
        # Reset risk management
        self.daily_trades = 0
        self.daily_profit = 0
        self.daily_loss = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.current_market_regime = "unknown"
        self.open_positions_count = 0
        
        # Get initial observation
        obs = self._get_observation()
        
        return obs, {}
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action (int): Action to take
                0: Hold
                1: Buy with 25% of allowed position size
                2: Buy with 50% of allowed position size
                3: Buy with 75% of allowed position size
                4: Buy with 100% of allowed position size
                5: Sell
                
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Get current price
        current_price = self.df.iloc[self.current_step]["close"]
        prev_net_worth = self.net_worth
        
        # Check if stop loss or target hit before handling action
        if self.shares_held > 0:
            self._update_trailing_stop(current_price)
            stop_or_target_hit, exit_type = self._check_stop_and_target(current_price)
            
            if stop_or_target_hit:
                # Force sell if stop loss or target hit
                action = 5  # Sell action
        
        # Execute trading action
        if action == 0:  # Hold
            pass
        
        elif action >= 1 and action <= 4 and self.shares_held == 0:  # Buy
            # Calculate position size based on action
            position_size_multiplier = action * 0.25  # 25%, 50%, 75%, or 100% of allowed size
            
            # Apply dynamic position sizing
            position_size = self.balance * self.position_size_pct * position_size_multiplier
            position_size = min(position_size, self.balance * 0.95)  # Cap at 95% of balance
            
            # Calculate shares to buy
            shares_bought = int(position_size // current_price)

            if shares_bought > 0:
                # Calculate transaction costs
                trade_value = shares_bought * current_price
                txn_costs = self._calculate_transaction_costs(trade_value)  
                self.last_txn_cost = txn_costs
                cost = trade_value + txn_costs
                
                # Update account
                self.balance -= cost
                self.shares_held += shares_bought
                self.total_shares_bought += shares_bought
                self.cost_basis = current_price
                self.total_trades += 1
                self.daily_trades += 1
                self.open_positions_count += 1
                
                # Set position type
                self.position_type = 'long'
                
                # Set stop loss and target
                self._setup_risk_management(current_price)
                
                # Reset days in trade counter
                self.days_in_trade = 0
                
                # Store entry date if available
                if 'date' in self.df.columns:
                    self.entry_date = self.df.iloc[self.current_step]['date']
                if 'time' in self.df.columns:
                    self.entry_time = self.df.iloc[self.current_step]['time']

                # Log trade
                self.trades.append({
                    'step': self.current_step,
                    'type': 'buy',
                    'price': current_price,
                    'shares': shares_bought,
                    'cost': cost,
                    'stop_loss': self.stop_loss,
                    'trailing_stop': self.trailing_stop,
                    'target': self.target_price
                })
        
        elif action == 5 and self.shares_held > 0:  # Sell
            # Sell all shares
            shares_sold = self.shares_held
            revenue = shares_sold * current_price

            # Calculate transaction costs
            txn_costs = self._calculate_transaction_costs(revenue)
            self.last_txn_cost = txn_costs
            revenue -= txn_costs

            # Calculate profit/loss
            position_value_before = shares_sold * self.cost_basis
            position_value_after = shares_sold * current_price
            profit = position_value_after - position_value_before
            profit_pct = profit / position_value_before * 100 if position_value_before > 0 else 0
            
            # Update account
            self.balance += revenue
            self.shares_held = 0
            self.total_shares_sold += shares_sold
            self.total_trades += 1
            self.daily_trades += 1
            self.open_positions_count -= 1
            
            # Reset position tracking
            position_type = self.position_type
            self.position_type = None
            self.stop_loss = 0
            self.trailing_stop = 0
            self.target_price = 0
            self.highest_price_since_buy = 0
            
            # Log trade
            self.trades.append({
                'step': self.current_step,
                'type': 'sell',
                'price': current_price,
                'shares': shares_sold,
                'revenue': revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'days_held': self.days_in_trade,
                'exit_type': exit_type if exit_type else 'manual'
            })
            
            # Update win/loss counters
            if profit > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.daily_profit += profit
            else:
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.daily_loss += abs(profit)
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.rewards.append(reward)
        
        # Update net worth history
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        # Update maximum drawdown
        if self.net_worth < self.max_net_worth:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        else:
            self.max_net_worth = self.net_worth
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info for analysis
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'step': self.current_step,
            'current_price': current_price,
            'position_type': self.position_type,
            'trailing_stop': self.trailing_stop,
            'target_price': self.target_price,
            'market_regime': self.current_market_regime,
            'daily_trades': self.daily_trades,
            'open_positions': self.open_positions_count,
            'max_drawdown': self.max_drawdown,
            'buy_signal_strength': self.buy_signal_strength,
            'sell_signal_strength': self.sell_signal_strength
        }

        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode (str): Render mode
            
        Returns:
            None
        """
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            print(f"Step: {self.current_step}")
            print(f"Symbol: {self.symbol}")
            print(f"Balance: ₹{self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Net worth: ₹{self.net_worth:.2f}")
            print(f"Profit: ₹{profit:.2f} ({(profit/self.initial_balance)*100:.2f}%)")
            if self.position_type:
                print(f"Position: {self.position_type}")
                print(f"Trailing Stop: ₹{self.trailing_stop:.2f}")
                print(f"Target: ₹{self.target_price:.2f}")
    
    def _calculate_transaction_costs(self, trade_value, is_intraday=False):
        """
        Calculate transaction costs
        
        Args:
            trade_value (float): Value of the trade
            is_intraday (bool): Whether the trade is intraday
            
        Returns:
            float: Transaction costs
        """
        # Default values if config attributes not found
        brokerage_rate = float(getattr(self.config, 'BROKERAGE_INTRADAY' if is_intraday else 'BROKERAGE_DELIVERY', 0.0003))
        stt_rate = float(getattr(self.config, 'STT_INTRADAY' if is_intraday else 'STT_DELIVERY', 0.00025))
        exchange_rate = float(getattr(self.config, 'EXCHANGE_TXN_CHARGE', 0.0000345))
        sebi_rate = float(getattr(self.config, 'SEBI_CHARGES', 0.000001))
        stamp_rate = float(getattr(self.config, 'STAMP_DUTY', 0.00015))
        gst_rate = float(getattr(self.config, 'GST', 0.18))
        
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
        max_cost = trade_value * 0.05  # Cap at 5% of trade value
        total_cost = min(total_cost, max_cost)
        
        return total_cost
    
    def _setup_risk_management(self, current_price):
        """
        Set up stop loss and target price
        
        Args:
            current_price (float): Current price
        """
        # Calculate ATR for dynamic stop loss and targets
        if 'ATR' in self.df.columns:
            current_atr = self.df.iloc[self.current_step]['ATR']
        else:
            current_atr = current_price * 0.02  # Default to 2% of price
        
        # Set stop loss
        self.stop_loss = current_price - (current_atr * self.trailing_stop_atr_multiplier)
        self.initial_stop_loss = self.stop_loss
        self.trailing_stop = self.stop_loss
        
        # Calculate target price
        stop_distance = current_price - self.stop_loss
        target_distance = stop_distance * self.min_rr_ratio
        
        # Make sure target is at least the target_atr_multiplier * ATR away
        target_distance = max(target_distance, current_atr * self.target_atr_multiplier)
        
        self.target_price = current_price + target_distance
        
        # Initialize highest price
        self.highest_price_since_buy = current_price
    
    def _update_trailing_stop(self, current_price):
        """
        Update trailing stop if price moves in favorable direction
        
        Args:
            current_price (float): Current price
        """
        if not self.position_type:
            return
            
        # For long positions (short positions not implemented)
        if self.position_type == 'long':
            if current_price > self.highest_price_since_buy:
                # Update highest price
                self.highest_price_since_buy = current_price
                
                # Calculate profit percentage
                profit_pct = (current_price - self.cost_basis) / self.cost_basis if self.cost_basis > 0 else 0
                
                # Get current ATR
                if 'ATR' in self.df.columns:
                    current_atr = self.df.iloc[self.current_step]['ATR']
                else:
                    current_atr = current_price * 0.02  # Default to 2% of price
                
                # Adjust trailing stop multiplier based on profit
                multiplier = self.trailing_stop_atr_multiplier
                if profit_pct > 0.1:  # Over 10% profit
                    multiplier *= 0.7  # Tighter trail to protect profits
                elif profit_pct > 0.05:  # Over 5% profit
                    multiplier *= 0.8
                
                # Calculate new stop
                new_stop = current_price - (current_atr * multiplier)
                
                # Only update if new stop is higher
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    
                    # Move to breakeven after sufficient profit
                    if profit_pct > 0.03 and self.trailing_stop < self.cost_basis:
                        self.trailing_stop = max(self.trailing_stop, self.cost_basis * 1.001)  # 0.1% above entry
    
    def _check_stop_and_target(self, current_price):
        """
        Check if stop loss or target price is hit
        
        Args:
            current_price (float): Current price
            
        Returns:
            tuple: (hit, exit_type)
        """
        if not self.position_type:
            return False, None
            
        hit_stop = False
        hit_target = False
        
        # For long positions (short positions not implemented)
        if self.position_type == 'long':
            # Check stop loss
            if current_price <= self.trailing_stop:
                hit_stop = True
                exit_type = 'stop_loss'
            
            # Check target
            elif current_price >= self.target_price:
                hit_target = True
                exit_type = 'target'
        
        return hit_stop or hit_target, exit_type if (hit_stop or hit_target) else None
    
    def _calculate_reward(self, action):
        """
        Calculate reward for the current step
        
        Args:
            action (int): Action taken
            
        Returns:
            float: Reward value
        """
        # Get current price
        current_price = self.df.iloc[self.current_step]["close"]
        
        # Calculate change in net worth
        prev_net_worth = self.net_worth_history[-1]
        current_net_worth = self.balance + self.shares_held * current_price
        net_worth_change = current_net_worth - prev_net_worth
        
        # Base reward is the normalized change in net worth
        reward = net_worth_change / self.initial_balance * 100  # Scale to percentage
        
        # Reward for holding profitable position
        if action == 0 and self.shares_held > 0 and net_worth_change > 0:
            reward += 0.1  # Small incentive
        
        # Penalty for excessive drawdown
        if self.shares_held > 0:
            current_drawdown = (self.max_net_worth - current_net_worth) / self.max_net_worth
            if current_drawdown > 0.1:  # Only penalize for large drawdowns (10%+)
                reward -= current_drawdown * 3  # Moderate penalty
        
        # Reward for following signals
        if action >= 1 and action <= 4 and self.buy_signal_strength > self.sell_signal_strength:
            reward += 0.5  # Reward for buying on strong buy signal
        elif action == 5 and self.sell_signal_strength > self.buy_signal_strength:
            reward += 0.5  # Reward for selling on strong sell signal
        
        # Penalty for going against signals
        if action >= 1 and action <= 4 and self.sell_signal_strength > self.buy_signal_strength + 3:
            reward -= 0.5  # Penalty for buying against a stronger sell signal
        elif action == 5 and self.buy_signal_strength > self.sell_signal_strength + 3:
            reward -= 0.5  # Penalty for selling against a stronger buy signal
        
        return reward
    
    def _get_observation(self):
        """
        Get the current observation
        
        Returns:
            numpy.ndarray: Observation vector
        """
        try:
            # Check if we have enough data
            if self.current_step < self.window_size:
                return np.zeros(60, dtype=np.float32)  # Return zeros if not enough data
                
            # Get current price data
            current_price = self.df.iloc[self.current_step]["close"]
            
            # Basic account state features
            obs = np.array([
                self.balance / self.initial_balance,
                self.shares_held / 100,  # Normalize shares
                self.cost_basis / current_price if self.cost_basis > 0 else 0,
                current_price / self.df["close"].iloc[max(0, self.current_step-20):self.current_step+1].mean(),  # Normalized price
                1 if self.position_type == 'long' else 0,  # Is in long position
                0,  # Is in short position (not supported in this implementation)
                self.trailing_stop / current_price if self.trailing_stop > 0 else 0,  # Normalized trailing stop
                self.target_price / current_price if self.target_price > 0 else 0,  # Normalized target
                self.days_in_trade / 10 if self.days_in_trade > 0 else 0,  # Days in trade
                self.daily_trades / 10,  # Daily trades
                self.max_drawdown * 10,  # Amplified drawdown signal
                self.consecutive_losses / 5 if self.consecutive_losses > 0 else 0,  # Consecutive losses
                self.consecutive_wins / 5 if self.consecutive_wins > 0 else 0,  # Consecutive wins
                self.buy_signal_strength / 10 if hasattr(self, 'buy_signal_strength') else 0,  # Buy signal strength
                self.sell_signal_strength / 10 if hasattr(self, 'sell_signal_strength') else 0,  # Sell signal strength
                # Market regime (one-hot encoding)
                1 if self.current_market_regime == "trending_up" else 0,
                1 if self.current_market_regime == "trending_down" else 0,
                1 if self.current_market_regime == "ranging" else 0,
                1 if self.current_market_regime == "volatile" else 0
            ])
            
            # Add technical indicators if available
            # We'll use placeholders for indicators if they don't exist in the DataFrame
            
            # Add price data
            price_data = np.array([
                self.df.iloc[self.current_step].get('open', current_price*0.99) / current_price,
                self.df.iloc[self.current_step].get('high', current_price*1.01) / current_price,
                self.df.iloc[self.current_step].get('low', current_price*0.99) / current_price,
                current_price / current_price,  # Normalized to 1
                self.df.iloc[self.current_step].get('volume', 10000) / 100000  # Normalized volume
            ])
            
            # Combine observations
            obs = np.concatenate([obs, price_data])
            
            # Add technical indicators (placeholders)
            technical_indicators = np.zeros(36)  # Remaining indicators to reach 60 elements
            
            # Combine all features
            obs = np.concatenate([obs, technical_indicators])
            
            # Ensure observation has exactly 60 elements
            if len(obs) > 60:
                obs = obs[:60]
            elif len(obs) < 60:
                obs = np.concatenate([obs, np.zeros(60 - len(obs))])
                
            return obs.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error creating observation: {str(e)}")
            return np.zeros(60, dtype=np.float32)