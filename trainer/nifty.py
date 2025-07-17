"""
Nifty and Bank Nifty Options Trading RL Agent
Using Gymnasium, Stable-Baselines3, and Python 3.12

This implementation includes:
1. Custom Gymnasium environment for options trading
2. Data fetching from NSE (National Stock Exchange) 
3. Feature engineering for options trading
4. RL agent implementation using PPO algorithm
5. Backtesting and evaluation framework
"""

import os
import datetime as dt
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
import yfinance as yf  # For fetching market data
import nsepy  # For NSE-specific data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

# Suppress warnings
warnings.filterwarnings('ignore')

class OptionsDataHandler:
    """
    Handles data fetching and preprocessing for Nifty and Bank Nifty options
    """
    def __init__(self, index_symbol: str = "^NSEI", start_date: str = "2022-01-01",
                 end_date: Optional[str] = None, expiry_gap: int = 30):
        """
        Initialize the data handler
        
        Args:
            index_symbol: The index symbol (^NSEI for Nifty, ^NSEBANK for Bank Nifty)
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to current date)
            expiry_gap: Days until expiry for options to fetch
        """
        self.index_symbol = index_symbol
        self.start_date = start_date
        self.end_date = end_date if end_date else dt.datetime.now().strftime("%Y-%m-%d")
        self.expiry_gap = expiry_gap
        self.index_data = None
        self.options_data = {}
        
    def fetch_index_data(self) -> pd.DataFrame:
        """Fetch historical index data"""
        print(f"Fetching historical data for {self.index_symbol}...")
        
        if self.index_symbol == "^NSEI":
            # Fetch Nifty data
            index_data = nsepy.get_history(symbol="NIFTY", 
                                         start=dt.datetime.strptime(self.start_date, "%Y-%m-%d"),
                                         end=dt.datetime.strptime(self.end_date, "%Y-%m-%d"),
                                         index=True)
        elif self.index_symbol == "^NSEBANK":
            # Fetch Bank Nifty data
            index_data = nsepy.get_history(symbol="BANKNIFTY", 
                                         start=dt.datetime.strptime(self.start_date, "%Y-%m-%d"),
                                         end=dt.datetime.strptime(self.end_date, "%Y-%m-%d"),
                                         index=True)
        else:
            # Fallback to yfinance for other symbols
            index_data = yf.download(self.index_symbol, start=self.start_date, end=self.end_date)
        
        # Clean and preprocess data
        index_data = index_data.dropna()
        self.index_data = index_data
        return index_data
    
    def fetch_options_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch options chain data for multiple dates
        This is a simplified implementation - in production, you'd use NSE's API
        or a data vendor for more comprehensive options data
        """
        if self.index_data is None:
            self.fetch_index_data()
        
        print("Fetching options data...")
        options_data = {}
        
        # Get a sample of dates from our index data
        sample_dates = list(self.index_data.index)[::20]  # Sample every 20th day
        
        for date in sample_dates:
            try:
                # Convert to datetime if it's not already
                if not isinstance(date, dt.datetime):
                    date = pd.to_datetime(date)
                
                # Find next expiry date
                expiry_date = date + dt.timedelta(days=self.expiry_gap)
                
                # Get current index price
                if isinstance(self.index_data.index, pd.DatetimeIndex):
                    closest_date = self.index_data.index[self.index_data.index <= date][-1]
                    current_price = self.index_data.loc[closest_date, 'Close']
                else:
                    current_price = self.index_data.loc[date, 'Close']
                
                # Calculate strikes (ATM + 5 strikes above/below)
                atm_strike = round(current_price / 100) * 100  # Round to nearest 100
                strikes = [atm_strike + i * 100 for i in range(-5, 6)]
                
                # Create synthetic options data
                # In production, you'd fetch real options chain data from NSE
                options_df = pd.DataFrame()
                options_df['Strike'] = strikes
                options_df['ExpiryDate'] = expiry_date
                options_df['SpotPrice'] = current_price
                
                # Calculate synthetic greeks and option prices using Black-Scholes
                # (simplified for this example)
                for strike in strikes:
                    moneyness = current_price / strike
                    
                    # Call option
                    days_to_expiry = (expiry_date - date).days
                    iv = 0.15 + 0.05 * abs(1 - moneyness)  # Higher IV for OTM options
                    time_value = current_price * iv * np.sqrt(days_to_expiry/365)
                    intrinsic_value_call = max(0, current_price - strike)
                    call_price = intrinsic_value_call + time_value
                    
                    # Put option
                    intrinsic_value_put = max(0, strike - current_price)
                    put_price = intrinsic_value_put + time_value
                    
                    # Filter to this strike
                    strike_df = options_df[options_df['Strike'] == strike].copy()
                    strike_idx = strike_df.index[0]
                    
                    # Add option prices and greeks
                    options_df.loc[strike_idx, 'CallPrice'] = call_price
                    options_df.loc[strike_idx, 'PutPrice'] = put_price
                    options_df.loc[strike_idx, 'CallIV'] = iv
                    options_df.loc[strike_idx, 'PutIV'] = iv
                    
                    # Calculate simplified greeks
                    options_df.loc[strike_idx, 'CallDelta'] = 0.5 + 0.5 * (moneyness - 1) / iv
                    options_df.loc[strike_idx, 'PutDelta'] = -0.5 + 0.5 * (1 - moneyness) / iv
                    options_df.loc[strike_idx, 'Gamma'] = 0.1 * (1 - abs(moneyness - 1))
                    options_df.loc[strike_idx, 'Vega'] = 0.1 * current_price * np.sqrt(days_to_expiry/365)
                    options_df.loc[strike_idx, 'Theta'] = -0.05 * current_price * iv / np.sqrt(days_to_expiry/365)
                
                options_data[date.strftime("%Y-%m-%d")] = options_df
            except Exception as e:
                print(f"Error fetching options data for {date}: {e}")
                continue
        
        self.options_data = options_data
        return options_data
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer features for the RL model
        
        Returns:
            DataFrame with technical indicators and options features
        """
        if self.index_data is None:
            self.fetch_index_data()
            
        if not self.options_data:
            self.fetch_options_data()
        
        # Create copy of index data for feature engineering
        df = self.index_data.copy()
        
        # Calculate technical indicators
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_std'] = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * df['BB_std']
        df['BB_lower'] = df['BB_middle'] - 2 * df['BB_std']
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Momentum
        df['Momentum'] = df['Close'].pct_change(periods=10)
        
        # MACD
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Volatility
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Add options-specific features
        # For each day, find the closest options data we have
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, dt.datetime) else idx
            
            # Find closest available options date
            available_dates = list(self.options_data.keys())
            if available_dates:
                closest_date = min(available_dates, key=lambda x: abs(pd.to_datetime(x) - pd.to_datetime(date_str)))
                options_df = self.options_data[closest_date]
                
                # Find ATM option
                spot_price = row['Close']
                atm_strike = options_df.iloc[(options_df['Strike'] - spot_price).abs().argsort()[0]]['Strike']
                atm_row = options_df[options_df['Strike'] == atm_strike].iloc[0]
                
                # Add ATM option features
                df.loc[idx, 'ATM_Call_Price'] = atm_row['CallPrice']
                df.loc[idx, 'ATM_Put_Price'] = atm_row['PutPrice']
                df.loc[idx, 'ATM_Call_IV'] = atm_row['CallIV']
                df.loc[idx, 'ATM_Put_IV'] = atm_row['PutIV']
                df.loc[idx, 'ATM_Call_Delta'] = atm_row['CallDelta']
                df.loc[idx, 'ATM_Put_Delta'] = atm_row['PutDelta']
                df.loc[idx, 'ATM_Gamma'] = atm_row['Gamma']
                df.loc[idx, 'ATM_Vega'] = atm_row['Vega']
                df.loc[idx, 'ATM_Theta'] = atm_row['Theta']
                
                # Calculate Put-Call ratio (PCR)
                df.loc[idx, 'PCR'] = atm_row['PutPrice'] / atm_row['CallPrice']
        
        # Forward returns (used for reward calculation)
        df['Next_Return'] = df['Close'].pct_change(periods=1).shift(-1)
        
        # Drop NaN values after creating indicators
        df = df.dropna()
        
        return df


class OptionsEnvironment(gym.Env):
    """
    Custom Gymnasium environment for options trading
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 100000.0, 
                 transaction_cost: float = 0.0005, lot_size: int = 50, 
                 max_positions: int = 5):
        """
        Initialize the trading environment
        
        Args:
            data: DataFrame with price data and features
            initial_balance: Starting capital
            transaction_cost: Cost per trade as a percentage
            lot_size: Number of options per lot
            max_positions: Maximum number of concurrent positions
        """
        super(OptionsEnvironment, self).__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.lot_size = lot_size
        self.max_positions = max_positions
        
        # Environment attributes
        self.current_step = 0
        self.current_balance = initial_balance
        self.current_nav = initial_balance  # Net Asset Value
        self.done = False
        
        # Track positions
        self.positions = []  # List of active positions
        
        # Portfolio history
        self.portfolio_history = []
        
        # Define action space
        # 0: Do nothing
        # 1: Buy Call
        # 2: Buy Put
        # 3: Sell to close Call
        # 4: Sell to close Put
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        # We include price data, technical indicators, and options-specific features
        num_features = 22  # Adjust based on the number of features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float32
        )
        
    def _get_observation(self):
        """
        Get the current observation (state)
        
        Returns:
            Numpy array with current market state features
        """
        obs = np.array([
            self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['Open'],
            self.data.iloc[self.current_step]['High'],
            self.data.iloc[self.current_step]['Low'],
            self.data.iloc[self.current_step]['Volume'],
            self.data.iloc[self.current_step]['SMA_20'] / self.data.iloc[self.current_step]['Close'] - 1,
            self.data.iloc[self.current_step]['SMA_50'] / self.data.iloc[self.current_step]['Close'] - 1,
            (self.data.iloc[self.current_step]['BB_upper'] - self.data.iloc[self.current_step]['Close']) / self.data.iloc[self.current_step]['Close'],
            (self.data.iloc[self.current_step]['Close'] - self.data.iloc[self.current_step]['BB_lower']) / self.data.iloc[self.current_step]['Close'],
            self.data.iloc[self.current_step]['RSI'] / 100,
            self.data.iloc[self.current_step]['Momentum'],
            self.data.iloc[self.current_step]['MACD'],
            self.data.iloc[self.current_step]['MACD_signal'],
            self.data.iloc[self.current_step]['Volatility'],
            self.data.iloc[self.current_step]['ATM_Call_Price'],
            self.data.iloc[self.current_step]['ATM_Put_Price'],
            self.data.iloc[self.current_step]['ATM_Call_IV'],
            self.data.iloc[self.current_step]['ATM_Put_IV'],
            self.data.iloc[self.current_step]['ATM_Call_Delta'],
            self.data.iloc[self.current_step]['ATM_Put_Delta'],
            self.data.iloc[self.current_step]['ATM_Gamma'],
            self.data.iloc[self.current_step]['PCR']
        ], dtype=np.float32)
        
        return obs
    
    def _calculate_cost(self, action: int) -> float:
        """
        Calculate the cost of an action
        
        Args:
            action: The action to take
            
        Returns:
            The cost of the action
        """
        cost = 0.0
        
        if action == 1:  # Buy Call
            cost = self.data.iloc[self.current_step]['ATM_Call_Price'] * self.lot_size
        elif action == 2:  # Buy Put
            cost = self.data.iloc[self.current_step]['ATM_Put_Price'] * self.lot_size
        
        # Add transaction cost
        cost += cost * self.transaction_cost
        
        return cost
    
    def _take_action(self, action: int) -> float:
        """
        Execute the action in the environment
        
        Args:
            action: The action to take
            
        Returns:
            The reward for the action
        """
        reward = 0.0
        
        if action == 0:  # Do nothing
            pass
        
        elif action == 1:  # Buy Call
            # Check if we have enough balance and not too many positions
            cost = self._calculate_cost(action)
            
            if cost <= self.current_balance and len(self.positions) < self.max_positions:
                # Record the position
                position = {
                    'type': 'call',
                    'entry_price': self.data.iloc[self.current_step]['ATM_Call_Price'],
                    'entry_step': self.current_step,
                    'quantity': self.lot_size,
                    'strike': self.data.iloc[self.current_step]['Close']  # Using spot price as approx strike
                }
                self.positions.append(position)
                
                # Update balance
                self.current_balance -= cost
                
                # Small negative reward for opening a position (risk)
                reward -= 0.001
        
        elif action == 2:  # Buy Put
            # Check if we have enough balance and not too many positions
            cost = self._calculate_cost(action)
            
            if cost <= self.current_balance and len(self.positions) < self.max_positions:
                # Record the position
                position = {
                    'type': 'put',
                    'entry_price': self.data.iloc[self.current_step]['ATM_Put_Price'],
                    'entry_step': self.current_step,
                    'quantity': self.lot_size,
                    'strike': self.data.iloc[self.current_step]['Close']  # Using spot price as approx strike
                }
                self.positions.append(position)
                
                # Update balance
                self.current_balance -= cost
                
                # Small negative reward for opening a position (risk)
                reward -= 0.001
        
        elif action == 3:  # Sell to close Call
            # Find call positions to close
            call_positions = [p for p in self.positions if p['type'] == 'call']
            
            if call_positions:
                # Close the oldest call position
                position = call_positions[0]
                
                # Calculate profit/loss
                current_price = self.data.iloc[self.current_step]['ATM_Call_Price']
                profit = (current_price - position['entry_price']) * position['quantity']
                
                # Apply transaction cost
                profit -= current_price * position['quantity'] * self.transaction_cost
                
                # Update balance
                self.current_balance += position['entry_price'] * position['quantity'] + profit
                
                # Remove the position
                self.positions.remove(position)
                
                # Reward is proportional to profit
                reward += profit / (position['entry_price'] * position['quantity']) * 10
        
        elif action == 4:  # Sell to close Put
            # Find put positions to close
            put_positions = [p for p in self.positions if p['type'] == 'put']
            
            if put_positions:
                # Close the oldest put position
                position = put_positions[0]
                
                # Calculate profit/loss
                current_price = self.data.iloc[self.current_step]['ATM_Put_Price']
                profit = (current_price - position['entry_price']) * position['quantity']
                
                # Apply transaction cost
                profit -= current_price * position['quantity'] * self.transaction_cost
                
                # Update balance
                self.current_balance += position['entry_price'] * position['quantity'] + profit
                
                # Remove the position
                self.positions.remove(position)
                
                # Reward is proportional to profit
                reward += profit / (position['entry_price'] * position['quantity']) * 10
        
        # Update portfolio value
        self._update_nav()
        
        # Record portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'balance': self.current_balance,
            'nav': self.current_nav,
            'action': action
        })
        
        return reward
    
    def _update_nav(self):
        """
        Update the net asset value (NAV) of the portfolio
        """
        # Start with cash balance
        nav = self.current_balance
        
        # Add value of open positions
        for position in self.positions:
            if position['type'] == 'call':
                current_price = self.data.iloc[self.current_step]['ATM_Call_Price']
                nav += current_price * position['quantity']
            elif position['type'] == 'put':
                current_price = self.data.iloc[self.current_step]['ATM_Put_Price']
                nav += current_price * position['quantity']
        
        self.current_nav = nav
    
    def _get_reward(self) -> float:
        """
        Calculate the reward based on portfolio performance
        
        Returns:
            The reward value
        """
        # Record portfolio history length - 2 entries (must have at least 2 entries)
        if len(self.portfolio_history) < 2:
            return 0.0
        
        # Calculate percentage change in NAV
        prev_nav = self.portfolio_history[-2]['nav']
        current_nav = self.current_nav
        
        reward = (current_nav - prev_nav) / prev_nav
        
        # Scale reward
        reward *= 100
        
        return reward
    
    def step(self, action):
        """
        Take a step in the environment
        
        Args:
            action: The action to take
            
        Returns:
            observation, reward, done, info
        """
        # Check if we're done
        if self.current_step >= len(self.data) - 1:
            self.done = True
            return self._get_observation(), 0, self.done, False, {}
        
        # Take action and get action-specific reward
        action_reward = self._take_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate overall reward
        reward = self._get_reward() + action_reward
        
        # Get new observation
        obs = self._get_observation()
        
        # Check if we're done
        if self.current_step >= len(self.data) - 1:
            self.done = True
        
        # Return observation, reward, done, info
        info = {
            'balance': self.current_balance,
            'nav': self.current_nav,
            'positions': len(self.positions),
            'action': action
        }
        
        return obs, reward, self.done, False, info
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment
        
        Returns:
            Initial observation
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.current_nav = self.initial_balance
        self.done = False
        self.positions = []
        self.portfolio_history = []
        
        return self._get_observation(), {}
    
    def render(self, mode='human'):
        """
        Render the environment
        
        Args:
            mode: The render mode
            
        Returns:
            The rendered frame
        """
        # Create a plot of the portfolio value over time
        if len(self.portfolio_history) > 0:
            steps = [h['step'] for h in self.portfolio_history]
            nav = [h['nav'] for h in self.portfolio_history]
            
            plt.figure(figsize=(10, 5))
            plt.plot(steps, nav)
            plt.title('Portfolio Value')
            plt.xlabel('Step')
            plt.ylabel('NAV')
            plt.grid(True)
            
            if mode == 'human':
                plt.show()
            elif mode == 'rgb_array':
                fig = plt.gcf()
                fig.canvas.draw()
                img = np.array(fig.canvas.renderer.buffer_rgba())
                plt.close()
                return img


class TensorboardCallback(BaseCallback):
    """
    Custom callback for logging training metrics to TensorBoard
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log reward
        if 'rewards' in self.locals:
            reward = self.locals['rewards'][0]
            self.logger.record('reward', reward)
        
        # Log other metrics if available
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            info = self.locals['infos'][0]
            if 'nav' in info:
                self.logger.record('portfolio/nav', info['nav'])
            if 'balance' in info:
                self.logger.record('portfolio/balance', info['balance'])
            if 'positions' in info:
                self.logger.record('portfolio/positions', info['positions'])
        
        return True


def train_agent(env, total_timesteps=100000, model_path='models/options_trading_ppo'):
    """
    Train a PPO agent
    
    Args:
        env: The training environment
        total_timesteps: Number of timesteps to train for
        model_path: Path to save the model
        
    Returns:
        The trained model
    """
    # Create a vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Create the model
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./tensorboard_log/"
    )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: env.envs[0]])
    
    # Set up callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path='./logs/',
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    tensorboard_callback = TensorboardCallback()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, tensorboard_callback]
    )
    
    # Save the final model
    model.save(f"{model_path}/final_model")
    
    return model


def evaluate_agent(model, env, num_episodes=10):
    """
    Evaluate a trained agent
    
    Args:
        model: The trained model
        env: The evaluation environment
        num_episodes: Number of episodes to evaluate
        
    Returns:
        Mean reward, std reward
    """
    # Reset the environment
    env.reset()
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=num_episodes,
        deterministic=True
    )
    
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward


def backtest_agent(model, env, render=False):
    """
    Backtest a trained agent
    
    Args:
        model: The trained model
        env: The backtest environment
        render: Whether to render the backtest
        
    Returns:
        DataFrame with backtest results
    """
    # Reset the environment
    obs, _ = env.reset()
    done = False
    
    # Lists to store results
    steps = []
    actions = []
    rewards = []
    balances = []
    navs = []
    
    # Run the backtest
    while not done:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, done, _, info = env.step(action)
        
        # Store results
        steps.append(env.current_step)
        actions.append(action)
        rewards.append(reward)
        balances.append(info['balance'])
        navs.append(info['nav'])
        
        # Render if requested
        if render:
            env.render()
    
    # Create results DataFrame
    results = pd.DataFrame({
        'step': steps,
        'action': actions,
        'reward': rewards,
        'balance': balances,
        'nav': navs
    })
    
    # Calculate performance metrics
    initial_nav = results['nav'].iloc[0]
    final_nav = results['nav'].iloc[-1]
    
    total_return = (final_nav / initial_nav - 1) * 100
    daily_returns = results['nav'].pct_change().dropna()
    annualized_return = ((1 + total_return / 100) ** (252 / len(results)) - 1) * 100
    annualized_volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    max_drawdown = (results['nav'] / results['nav'].cummax() - 1).min() * 100
    
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annualized_return:.2f}%")
    print(f"Annualized Volatility: {annualized_volatility:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")
    
    return results


def plot_backtest_results(results):
    """
    Plot the backtest results
    
    Args:
        results: DataFrame with backtest results
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    
    # Plot portfolio value
    axes[0].plot(results['step'], results['nav'])
    axes[0].set_title('Portfolio Value')
    axes[0].set_ylabel('NAV (₹)')
    axes[0].grid(True)
    
    # Plot returns
    daily_returns = results['nav'].pct_change().fillna(0)
    axes[1].bar(results['step'], daily_returns * 100)
    axes[1].set_title('Daily Returns')
    axes[1].set_ylabel('Return (%)')
    axes[1].grid(True)
    
    # Plot drawdown
    drawdown = (results['nav'] / results['nav'].cummax() - 1) * 100
    axes[2].fill_between(results['step'], drawdown, 0, color='red', alpha=0.3)
    axes[2].set_title('Drawdown')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].set_xlabel('Trading Day')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function to run the trading agent
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create data handler
    print("Initializing data handler...")
    data_handler = OptionsDataHandler(
        index_symbol="^NSEI",  # Nifty 50 (use "^NSEBANK" for Bank Nifty)
        start_date="2020-01-01",
        end_date="2023-12-31",
        expiry_gap=30  # 30-day options
    )
    
    # Fetch data
    data_handler.fetch_index_data()
    data_handler.fetch_options_data()
    
    # Engineer features
    print("Engineering features...")
    data = data_handler.engineer_features()
    
    # Split data into train and test sets
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]
    
    print(f"Training data: {len(train_data)} days")
    print(f"Testing data: {len(test_data)} days")
    
    # Create environments
    train_env = OptionsEnvironment(
        data=train_data,
        initial_balance=100000.0,
        transaction_cost=0.0005,
        lot_size=50,
        max_positions=5
    )
    
    test_env = OptionsEnvironment(
        data=test_data,
        initial_balance=100000.0,
        transaction_cost=0.0005,
        lot_size=50,
        max_positions=5
    )
    
    # Check if model exists
    model_path = 'models/options_trading_ppo'
    try:
        # Load existing model
        print("Loading existing model...")
        model = PPO.load(f"{model_path}/best_model")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"No existing model found or error loading: {e}")
        print("Training new model...")
        
        # Train agent
        model = train_agent(train_env, total_timesteps=100000, model_path=model_path)
        print("Training completed!")
    
    # Evaluate agent
    print("Evaluating agent...")
    evaluate_agent(model, test_env, num_episodes=10)
    
    # Backtest agent
    print("Backtesting agent...")
    backtest_results = backtest_agent(model, test_env, render=False)
    
    # Plot backtest results
    plot_backtest_results(backtest_results)


# Strategy refinement functions
def optimize_hyperparameters(train_env, test_env):
    """
    Optimize hyperparameters for the trading agent
    
    Args:
        train_env: Training environment
        test_env: Testing environment
        
    Returns:
        Best hyperparameters
    """
    # Define hyperparameter grid
    param_grid = {
        'learning_rate': [0.0001, 0.0003, 0.001],
        'n_steps': [1024, 2048, 4096],
        'batch_size': [32, 64, 128],
        'n_epochs': [5, 10, 15],
        'gamma': [0.95, 0.99, 0.995],
        'ent_coef': [0.005, 0.01, 0.02]
    }
    
    # Vectorize environment
    train_env = DummyVecEnv([lambda: train_env])
    
    # Track best parameters and performance
    best_params = {}
    best_reward = float('-inf')
    
    # Number of trials
    n_trials = 10
    
    for trial in range(n_trials):
        # Randomly sample hyperparameters
        params = {
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'n_steps': np.random.choice(param_grid['n_steps']),
            'batch_size': np.random.choice(param_grid['batch_size']),
            'n_epochs': np.random.choice(param_grid['n_epochs']),
            'gamma': np.random.choice(param_grid['gamma']),
            'ent_coef': np.random.choice(param_grid['ent_coef'])
        }
        
        print(f"Trial {trial+1}/{n_trials}: {params}")
        
        # Create model with these parameters
        model = PPO(
            'MlpPolicy',
            train_env,
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=params['ent_coef'],
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=0
        )
        
        # Train for fewer steps for quick evaluation
        model.learn(total_timesteps=50000)
        
        # Evaluate
        mean_reward, _ = evaluate_agent(model, test_env, num_episodes=5)
        
        # Update best parameters if performance improved
        if mean_reward > best_reward:
            best_reward = mean_reward
            best_params = params
            print(f"New best: {best_reward:.2f}")
        
        print(f"Trial reward: {mean_reward:.2f}")
        print("-" * 40)
    
    print(f"Best hyperparameters: {best_params}")
    print(f"Best reward: {best_reward:.2f}")
    
    return best_params


def analyze_trades(backtest_results, data):
    """
    Analyze the trades made by the agent
    
    Args:
        backtest_results: DataFrame with backtest results
        data: Original price data
        
    Returns:
        DataFrame with trade analysis
    """
    # Find all trades (action != 0)
    trades = backtest_results[backtest_results['action'] != 0].copy()
    
    # Add price data
    for idx, row in trades.iterrows():
        step = row['step']
        trades.loc[idx, 'date'] = data.index[step]
        trades.loc[idx, 'price'] = data.iloc[step]['Close']
    
    # Calculate trade statistics
    win_trades = trades[trades['reward'] > 0]
    loss_trades = trades[trades['reward'] < 0]
    
    win_rate = len(win_trades) / len(trades) * 100 if len(trades) > 0 else 0
    avg_win = win_trades['reward'].mean() if len(win_trades) > 0 else 0
    avg_loss = loss_trades['reward'].mean() if len(loss_trades) > 0 else 0
    
    # Action type statistics
    action_counts = trades['action'].value_counts()
    
    print(f"Total trades: {len(trades)}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Average win: {avg_win:.4f}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Profit factor: {abs(avg_win * len(win_trades) / (avg_loss * len(loss_trades))):.2f}" if len(loss_trades) > 0 else "Infinity")
    print("\nAction counts:")
    for action, count in action_counts.items():
        action_name = {
            1: "Buy Call",
            2: "Buy Put",
            3: "Sell Call",
            4: "Sell Put"
        }.get(action, "Unknown")
        print(f"  {action_name}: {count} ({count / len(trades) * 100:.2f}%)")
    
    return trades


if __name__ == "__main__":
    main()
