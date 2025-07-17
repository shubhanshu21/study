import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os
import json
from finta import TA
from tqdm import tqdm
from rich import print
import time
import logging
import warnings
import optuna
from optuna.samplers import TPESampler
from tabulate import tabulate
import torch
from dotenv import load_dotenv
import pytz
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scalping_training_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Scalping_RL_Training_Bot")

SYMBOLS = ["ADANIPORTS", "RELIANCE"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_dotenv()

# Trading parameters optimized for scalping
DEFAULT_INITIAL_BALANCE = float(os.getenv("DEFAULT_INITIAL_BALANCE", "100000"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "scalping_models")
DATA_PATH = os.getenv("DATA_PATH", "datasets_5min/{}_5min.csv")  # 5-minute data path
RESULTS_SAVE_PATH = os.getenv("RESULTS_SAVE_PATH", "scalping_results")

# Scalping-specific parameters
SCALP_TARGET_PROFIT = float(os.getenv("SCALP_TARGET_PROFIT", "0.5"))  # 0.5% target
SCALP_STOP_LOSS = float(os.getenv("SCALP_STOP_LOSS", "0.3"))  # 0.3% stop loss
MAX_POSITION_TIME_BARS = int(os.getenv("MAX_POSITION_TIME_BARS", "12"))  # 1 hour (12 * 5min)
SCALP_POSITION_SIZE_PCT = float(os.getenv("SCALP_POSITION_SIZE_PCT", "50"))  # 50% of capital

# Transaction costs (higher frequency = more impact)
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY", "0.0003"))
STT_INTRADAY = float(os.getenv("STT_INTRADAY", "0.00025"))
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE", "0.0000345"))
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES", "0.000001"))
STAMP_DUTY = float(os.getenv("STAMP_DUTY", "0.00015"))
GST = float(os.getenv("GST", "0.18"))

TIMEZONE = pytz.timezone('Asia/Kolkata')

# Logging levels
LOG_LEVELS = {
    'DATA_LOAD': logging.INFO,
    'MODEL_TRAINING': logging.INFO,
    'MODEL_EVALUATION': logging.INFO,
    'PERFORMANCE_METRICS': logging.INFO,
    'EXCEPTION': logging.ERROR
}

class ScalpingTradingEnv(gym.Env):
    """Scalping-optimized trading environment for 5-minute timeframe"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, symbol, initial_balance=DEFAULT_INITIAL_BALANCE, window_size=20,
                 scalp_target=SCALP_TARGET_PROFIT, scalp_stop=SCALP_STOP_LOSS, 
                 max_position_time=MAX_POSITION_TIME_BARS, position_size_pct=SCALP_POSITION_SIZE_PCT):
        super(ScalpingTradingEnv, self).__init__()
        
        # Data setup
        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Scalping parameters
        self.scalp_target = scalp_target / 100  # Convert percentage to decimal
        self.scalp_stop = scalp_stop / 100
        self.max_position_time = max_position_time
        self.position_size_pct = position_size_pct / 100
        
        # Account variables
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.position_entry_time = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = initial_balance
        
        # Scalping-specific tracking
        self.scalp_wins = 0
        self.scalp_losses = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        # Episode variables
        self.current_step = 0
        self.rewards = []
        self.trades = []
        self.net_worth_history = [initial_balance]
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=Hold, 1=Buy, 2=Sell
        
        # Observation space optimized for scalping (more features for short-term patterns)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(65,), dtype=np.float32  # Expanded for scalping indicators
        )

    def calculate_transaction_costs(self, trade_value, is_intraday=True):
        """Calculate transaction costs - intraday by default for scalping"""
        brokerage_rate = BROKERAGE_INTRADAY
        stt_rate = STT_INTRADAY
        
        brokerage = min(trade_value * brokerage_rate, 20)  # Cap at ₹20
        stt = trade_value * stt_rate
        exchange_charge = trade_value * EXCHANGE_TXN_CHARGE
        sebi_charges = trade_value * SEBI_CHARGES
        stamp_duty = trade_value * STAMP_DUTY
        gst = (brokerage + exchange_charge) * GST
        
        total_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst
        
        # Cap total costs at 2% for scalping (to prevent unrealistic costs)
        max_allowed_cost = trade_value * 0.02
        if total_cost > max_allowed_cost:
            total_cost = max_allowed_cost
            
        return round(total_cost, 2)

    def _get_observation(self):
        """Get scalping-optimized observation with short-term indicators"""
        current_price = self.df.iloc[self.current_step]["Close"]
        
        # Basic account state (8 features)
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100,
            self.cost_basis / current_price if self.cost_basis > 0 else 0,
            current_price / self.df["Close"].iloc[max(0, self.current_step-10):self.current_step+1].mean(),
            1 if self.shares_held > 0 else 0,  # In position
            (self.current_step - self.position_entry_time) / self.max_position_time if self.shares_held > 0 else 0,
            self.consecutive_losses / 10,  # Normalize consecutive losses
            self.scalp_wins / max(1, self.scalp_wins + self.scalp_losses),  # Win rate
        ])
        
        if self.current_step >= self.window_size:
            window_data = self.df.iloc[self.current_step-self.window_size+1:self.current_step+1]
            
            # Price and volume data (5 features)
            close_mean = window_data["Close"].mean()
            volume_mean = window_data["Volume"].mean()
            
            obs = np.append(obs, [
                window_data["Open"].iloc[-1] / close_mean,
                window_data["High"].iloc[-1] / close_mean,
                window_data["Low"].iloc[-1] / close_mean,
                window_data["Close"].iloc[-1] / close_mean,
                window_data["Volume"].iloc[-1] / volume_mean if volume_mean > 0 else 1,
            ])
            
            # Fast EMAs for scalping (4 features)
            for period in [3, 5, 8, 13]:
                ema_col = f'EMA{period}'
                if ema_col in window_data.columns:
                    ema = window_data[ema_col].iloc[-1] / close_mean
                else:
                    ema = window_data["Close"].ewm(span=period).mean().iloc[-1] / close_mean
                obs = np.append(obs, [ema])
            
            # Short-term SMAs (4 features)
            for period in [3, 5, 8, 13]:
                sma_col = f'SMA{period}'
                if sma_col in window_data.columns:
                    sma = window_data[sma_col].iloc[-1] / close_mean
                else:
                    sma = window_data["Close"].rolling(period).mean().iloc[-1] / close_mean
                obs = np.append(obs, [sma])
            
            # Momentum indicators (6 features)
            price_changes = window_data["Close"].pct_change()
            obs = np.append(obs, [
                price_changes.iloc[-1],  # Last 1-bar change
                price_changes.iloc[-3:].mean(),  # 3-bar average change
                price_changes.iloc[-5:].mean(),  # 5-bar average change
                price_changes.iloc[-3:].std(),  # 3-bar volatility
                (window_data["High"].iloc[-1] - window_data["Low"].iloc[-1]) / close_mean,  # Current bar range
                window_data["Close"].iloc[-5:].std() / close_mean,  # 5-bar price volatility
            ])
            
            # Volume indicators (4 features)
            volume_changes = window_data["Volume"].pct_change()
            obs = np.append(obs, [
                volume_changes.iloc[-1],  # Volume change
                window_data["Volume"].iloc[-3:].mean() / volume_mean if volume_mean > 0 else 1,
                (window_data["Volume"].iloc[-1] > window_data["Volume"].rolling(5).mean().iloc[-1]) * 1.0,
                window_data["Volume"].iloc[-3:].std() / volume_mean if volume_mean > 0 else 0,
            ])
            
            # RSI and fast oscillators (3 features)
            if 'RSI' in window_data.columns:
                rsi = window_data['RSI'].iloc[-1] / 100
            else:
                # Calculate fast RSI for scalping
                delta = window_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(5).mean()  # Faster RSI
                loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
                rs = gain / loss
                rsi = (100 - (100 / (1 + rs))).iloc[-1] / 100
            
            # Stochastic for scalping
            high_5 = window_data['High'].rolling(5).max()
            low_5 = window_data['Low'].rolling(5).min()
            stoch_k = ((window_data['Close'] - low_5) / (high_5 - low_5)).iloc[-1]
            stoch_d = stoch_k  # Simplified for speed
            
            obs = np.append(obs, [rsi, stoch_k, stoch_d])
            
            # MACD for trend (2 features)
            if 'MACD' in window_data.columns:
                macd = window_data['MACD'].iloc[-1] / close_mean
                signal = window_data['Signal'].iloc[-1] / close_mean
            else:
                ema_fast = window_data['Close'].ewm(span=5).mean()
                ema_slow = window_data['Close'].ewm(span=13).mean()
                macd = (ema_fast - ema_slow).iloc[-1] / close_mean
                signal = (ema_fast - ema_slow).ewm(span=3).mean().iloc[-1] / close_mean
            
            obs = np.append(obs, [macd, signal])
            
            # Bollinger Bands (3 features)
            if all(x in window_data.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                bb_upper = window_data['BB_Upper'].iloc[-1] / close_mean
                bb_middle = window_data['BB_Middle'].iloc[-1] / close_mean
                bb_lower = window_data['BB_Lower'].iloc[-1] / close_mean
            else:
                sma_bb = window_data['Close'].rolling(10).mean()
                std_bb = window_data['Close'].rolling(10).std()
                bb_upper = (sma_bb + 1.5 * std_bb).iloc[-1] / close_mean  # Tighter bands for scalping
                bb_middle = sma_bb.iloc[-1] / close_mean
                bb_lower = (sma_bb - 1.5 * std_bb).iloc[-1] / close_mean
            
            obs = np.append(obs, [bb_upper, bb_middle, bb_lower])
            
            # Support/Resistance levels (4 features)
            recent_highs = window_data['High'].rolling(5).max()
            recent_lows = window_data['Low'].rolling(5).min()
            resistance = recent_highs.iloc[-1] / close_mean
            support = recent_lows.iloc[-1] / close_mean
            
            # Distance to support/resistance
            dist_to_resistance = (resistance * close_mean - current_price) / current_price
            dist_to_support = (current_price - support * close_mean) / current_price
            
            obs = np.append(obs, [resistance, support, dist_to_resistance, dist_to_support])
            
            # Time-based features (3 features)
            # Market session indicators (assuming 5-min bars during market hours)
            bars_from_open = self.current_step % 75  # 375 minutes / 5 = 75 bars per day
            session_progress = bars_from_open / 75
            
            # Volatility regime
            current_volatility = window_data['Close'].pct_change().std()
            avg_volatility = self.df['Close'].pct_change().rolling(100).std().iloc[self.current_step]
            volatility_regime = current_volatility / avg_volatility if avg_volatility > 0 else 1
            
            obs = np.append(obs, [session_progress, volatility_regime, bars_from_open / 75])
            
            # EMA crossover signals (4 features)
            ema3 = window_data['Close'].ewm(span=3).mean().iloc[-1]
            ema5 = window_data['Close'].ewm(span=5).mean().iloc[-1]
            ema8 = window_data['Close'].ewm(span=8).mean().iloc[-1]
            ema13 = window_data['Close'].ewm(span=13).mean().iloc[-1]
            
            bullish_cross = (ema3 > ema5 > ema8) * 1.0
            bearish_cross = (ema3 < ema5 < ema8) * 1.0
            trend_strength = abs(ema3 - ema13) / close_mean
            trend_direction = (ema3 > ema13) * 1.0
            
            obs = np.append(obs, [bullish_cross, bearish_cross, trend_strength, trend_direction])
            
            # Price action patterns (3 features)
            # Engulfing patterns, doji, etc.
            body_size = abs(window_data['Close'].iloc[-1] - window_data['Open'].iloc[-1]) / close_mean
            upper_shadow = (window_data['High'].iloc[-1] - max(window_data['Close'].iloc[-1], window_data['Open'].iloc[-1])) / close_mean
            lower_shadow = (min(window_data['Close'].iloc[-1], window_data['Open'].iloc[-1]) - window_data['Low'].iloc[-1]) / close_mean
            
            obs = np.append(obs, [body_size, upper_shadow, lower_shadow])
            
        else:
            # Fill with default values if not enough data
            obs = np.append(obs, np.full(57, 0.5))  # 65 - 8 = 57 additional features
        
        # Ensure exactly 65 features
        if len(obs) < 65:
            obs = np.append(obs, np.full(65 - len(obs), 0.5))
        elif len(obs) > 65:
            obs = obs[:65]
        
        return np.nan_to_num(obs).astype(np.float32)

    def calculate_reward(self, action, exit_type=None):
        """Scalping-optimized reward function"""
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        net_worth_change = self.net_worth - prev_net_worth
        
        # Base reward from net worth change
        reward = net_worth_change / self.initial_balance * 1000  # Scale up for scalping
        
        # Scalping-specific rewards
        if action == 1:  # Buy
            reward -= 0.01  # Small entry penalty
            
        elif action == 2 and self.shares_held > 0:  # Sell
            holding_time = self.current_step - self.position_entry_time
            profit_pct = (current_price - self.cost_basis) / self.cost_basis if self.cost_basis > 0 else 0
            
            # Reward based on profit target achievement
            if profit_pct >= self.scalp_target:
                reward += 5.0  # Big reward for hitting target
                self.scalp_wins += 1
                self.consecutive_losses = 0
            elif profit_pct <= -self.scalp_stop:
                reward -= 2.0  # Penalty for hitting stop loss
                self.scalp_losses += 1
                self.consecutive_losses += 1
            else:
                # Partial profit/loss
                reward += profit_pct * 10  # Scale profit percentage
                if profit_pct > 0:
                    self.scalp_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.scalp_losses += 1
                    self.consecutive_losses += 1
            
            # Time-based rewards/penalties
            if holding_time < 3:  # Very quick scalp
                reward += 1.0
            elif holding_time > self.max_position_time:
                reward -= 1.0  # Penalty for holding too long
        
        # Position holding penalties/rewards
        if self.shares_held > 0:
            holding_time = self.current_step - self.position_entry_time
            unrealized_pnl_pct = (current_price - self.cost_basis) / self.cost_basis if self.cost_basis > 0 else 0
            
            # Penalty for holding losing positions too long
            if unrealized_pnl_pct < -self.scalp_stop and holding_time > 5:
                reward -= 0.5
            
            # Reward for letting winners run (but not too long)
            if unrealized_pnl_pct > self.scalp_target and holding_time < self.max_position_time:
                reward += 0.2
        
        # Consecutive loss penalty
        if self.consecutive_losses > 3:
            reward -= 0.5 * self.consecutive_losses
        
        # Update max consecutive losses
        self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # Update max net worth and drawdown
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        if self.net_worth < self.max_net_worth:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Drawdown penalty
            if current_drawdown > 0.05:  # 5% drawdown threshold for scalping
                reward -= current_drawdown * 10
        
        return reward

    def step(self, action):
        """Execute scalping action"""
        current_price = self.df.iloc[self.current_step]["Close"]
        exit_type = None
        
        # Check for automatic exits (stop loss, target, time limit)
        if self.shares_held > 0:
            holding_time = self.current_step - self.position_entry_time
            unrealized_pnl_pct = (current_price - self.cost_basis) / self.cost_basis if self.cost_basis > 0 else 0
            
            # Force exit conditions
            if (unrealized_pnl_pct <= -self.scalp_stop or  # Stop loss
                unrealized_pnl_pct >= self.scalp_target or  # Take profit
                holding_time >= self.max_position_time):    # Time limit
                
                action = 2  # Force sell
                if unrealized_pnl_pct <= -self.scalp_stop:
                    exit_type = 'stop_loss'
                elif unrealized_pnl_pct >= self.scalp_target:
                    exit_type = 'take_profit'
                else:
                    exit_type = 'time_limit'
        
        # Execute action
        if action == 0:  # Hold
            pass
            
        elif action == 1:  # Buy
            if self.balance > 0 and self.shares_held == 0:
                # Calculate position size based on percentage of capital
                max_investment = self.balance * self.position_size_pct
                max_shares = int(max_investment // current_price)
                
                if max_shares > 0:
                    shares_bought = max_shares
                    trade_value = shares_bought * current_price
                    txn_costs = self.calculate_transaction_costs(trade_value, is_intraday=True)
                    total_cost = trade_value + txn_costs
                    
                    if self.balance >= total_cost:
                        self.balance -= total_cost
                        self.shares_held = shares_bought
                        self.cost_basis = current_price
                        self.position_entry_time = self.current_step
                        self.total_trades += 1
                        
                        self.trades.append({
                            'step': self.current_step,
                            'type': 'buy',
                            'price': current_price,
                            'shares': shares_bought,
                            'cost': total_cost,
                            'txn_cost': txn_costs
                        })
                        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                shares_sold = self.shares_held
                revenue = shares_sold * current_price
                txn_costs = self.calculate_transaction_costs(revenue, is_intraday=True)
                net_revenue = revenue - txn_costs
                
                # Calculate profit/loss
                position_cost = shares_sold * self.cost_basis
                profit = revenue - position_cost - txn_costs
                profit_pct = profit / position_cost * 100 if position_cost > 0 else 0
                holding_time = self.current_step - self.position_entry_time
                
                self.balance += net_revenue
                self.shares_held = 0
                self.total_trades += 1
                
                self.trades.append({
                    'step': self.current_step,
                    'type': 'sell',
                    'price': current_price,
                    'shares': shares_sold,
                    'revenue': net_revenue,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_time': holding_time,
                    'exit_type': exit_type,
                    'txn_cost': txn_costs
                })
        
        # Calculate reward
        reward = self.calculate_reward(action, exit_type)
        self.rewards.append(reward)
        
        # Update net worth and history
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        truncated = False
        
        # Get observation
        obs = self._get_observation()
        
        # Info for analysis
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'scalp_wins': self.scalp_wins,
            'scalp_losses': self.scalp_losses,
            'win_rate': self.scalp_wins / max(1, self.scalp_wins + self.scalp_losses),
            'consecutive_losses': self.consecutive_losses
        }
        
        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """Reset environment for new episode"""
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.position_entry_time = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = self.initial_balance
        
        self.scalp_wins = 0
        self.scalp_losses = 0
        self.consecutive_losses = 0
        self.max_consecutive_losses = 0
        
        self.current_step = self.window_size
        self.rewards = []
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        
        return self._get_observation(), {}

    def calculate_performance_metrics(self):
        """Calculate scalping-specific performance metrics"""
        # Calculate returns
        returns = []
        for i in range(1, len(self.net_worth_history)):
            returns.append((self.net_worth_history[i] - self.net_worth_history[i-1]) / self.net_worth_history[i-1])
        
        total_return = (self.net_worth_history[-1] - self.initial_balance) / self.initial_balance
        
        # Annualize return (5-min data: 75 bars/day * 252 days = 18,900 bars/year)
        n_bars = len(self.net_worth_history)
        bars_per_year = 75 * 252  # 5-min bars per year
        annual_return = (1 + total_return) ** (bars_per_year / n_bars) - 1 if n_bars > 0 else 0
        
        # Sharpe ratio
        if len(returns) > 0:
            std_return = np.std(returns)
            avg_return = np.mean(returns)
            risk_free_rate = 0.05 / bars_per_year  # Daily risk-free rate
            sharpe_ratio = ((avg_return - risk_free_rate) / std_return) * np.sqrt(bars_per_year) if std_return > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Scalping-specific metrics
        total_completed_trades = self.scalp_wins + self.scalp_losses
        win_rate = self.scalp_wins / total_completed_trades if total_completed_trades > 0 else 0
        
        # Calculate profit factor and average trade metrics
        winning_trades = [t for t in self.trades if t['type'] == 'sell' and t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trades if t['type'] == 'sell' and t.get('profit', 0) <= 0]
        
        total_profit = sum(t.get('profit', 0) for t in winning_trades)
        total_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
        
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_profit = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        
        # Average holding time
        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        avg_holding_time = np.mean([t.get('holding_time', 0) for t in sell_trades]) if sell_trades else 0
        
        return {
            'symbol': self.symbol,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_completed_trades,
            'scalp_wins': self.scalp_wins,
            'scalp_losses': self.scalp_losses,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'avg_holding_time': avg_holding_time,
            'max_consecutive_losses': self.max_consecutive_losses,
            'net_worth': self.net_worth_history[-1],
            'total_transaction_costs': sum(t.get('txn_cost', 0) for t in self.trades),
        }

def calculate_scalping_indicators(df):
    """Calculate indicators optimized for 5-minute scalping"""
    print("Calculating scalping indicators for 5-minute data...")
    
    # Fast EMAs (3, 5, 8, 13, 21)
    for period in [3, 5, 8, 13, 21]:
        df[f'EMA{period}'] = TA.EMA(df, period=period)
        df[f'SMA{period}'] = TA.SMA(df, period=period)
    
    # Fast RSI (5 and 14 period)
    df['RSI5'] = TA.RSI(df, period=5)   # Fast RSI for scalping
    df['RSI'] = TA.RSI(df, period=14)   # Standard RSI
    
    # Fast Stochastic
    df['STOCH_K'] = TA.STOCH(df, period=5)  # Faster than 14
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
    
    # MACD with faster settings
    ema_fast = df['Close'].ewm(span=5).mean()
    ema_slow = df['Close'].ewm(span=13).mean()
    df['MACD'] = ema_fast - ema_slow
    df['Signal'] = df['MACD'].ewm(span=3).mean()
    
    # Bollinger Bands with tighter settings
    sma_bb = df['Close'].rolling(10).mean()
    std_bb = df['Close'].rolling(10).std()
    df['BB_Upper'] = sma_bb + 1.5 * std_bb  # Tighter bands
    df['BB_Middle'] = sma_bb
    df['BB_Lower'] = sma_bb - 1.5 * std_bb
    
    # Volume indicators
    df['OBV'] = TA.OBV(df)
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    
    # Price action
    df['High_Low_Pct'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['Body_Size'] = abs(df['Close'] - df['Open']) / df['Close'] * 100
    
    # Support/Resistance
    df['Resistance_5'] = df['High'].rolling(5).max()
    df['Support_5'] = df['Low'].rolling(5).min()
    
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(10).std() * 100
    
    # Fill NaN values
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    print("Scalping indicators calculation complete.")
    return df

def load_and_prepare_scalping_data(symbol, split_ratio=0.8):
    """Load and prepare 5-minute data for scalping"""
    logger.log(LOG_LEVELS['DATA_LOAD'], f"Loading 5-minute data for {symbol}...")
    
    # Load 5-minute data
    data_file = DATA_PATH.format(symbol)
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"5-minute data file not found: {data_file}")
    
    df = pd.read_csv(data_file)
    
    # Ensure datetime column
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    elif 'Datetime' in df.columns:
        df['Date'] = pd.to_datetime(df['Datetime'])
        df = df.drop('Datetime', axis=1)
    
    # Filter market hours (9:15 AM to 3:30 PM) if datetime info available
    if 'Date' in df.columns and df['Date'].dtype.name.startswith('datetime'):
        df = df[
            (df['Date'].dt.time >= pd.Timestamp('09:15:00').time()) &
            (df['Date'].dt.time <= pd.Timestamp('15:30:00').time())
        ].reset_index(drop=True)
    
    # Calculate scalping indicators
    df = calculate_scalping_indicators(df)
    
    # Split data
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
    
    logger.log(LOG_LEVELS['DATA_LOAD'], 
               f"Data preparation complete for {symbol}. "
               f"Train: {len(train_df)} bars, Test: {len(test_df)} bars")
    
    return train_df, test_df

def create_scalping_vec_env(df, symbol, **kwargs):
    """Create vectorized environment for scalping"""
    num_envs = 4  # Reduced for scalping (more complex environment)
    
    def make_env():
        def _init():
            return Monitor(ScalpingTradingEnv(df=df, symbol=symbol, **kwargs))
        return _init
    
    return SubprocVecEnv([make_env() for _ in range(num_envs)])

def optimize_scalping_hyperparameters(trial, train_df, symbol, **env_kwargs):
    """Optimize hyperparameters for scalping"""
    # Scalping-specific hyperparameter ranges
    learning_rate = trial.suggest_loguniform('learning_rate', 5e-5, 5e-3)
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048])  # Smaller steps for scalping
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.85, 0.99)
    gamma = trial.suggest_uniform('gamma', 0.95, 0.999)  # Higher gamma for scalping
    n_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.005, 0.1)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    
    # Scalping-optimized network architecture
    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]  # Larger networks for complex patterns
    )
    
    model = PPO(
        'MlpPolicy',
        create_scalping_vec_env(train_df, symbol, **env_kwargs),
        verbose=0,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gae_lambda=gae_lambda,
        gamma=gamma,
        n_epochs=n_epochs,
        ent_coef=ent_coef,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        device=device
    )
    
    # Train with fewer timesteps for faster optimization
    model.learn(total_timesteps=50_000)
    
    # Evaluate
    eval_env = ScalpingTradingEnv(df=train_df, symbol=symbol, **env_kwargs)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=3)
    
    return -mean_reward

def evaluate_scalping_model(model, test_df, symbol, num_episodes=1, **env_kwargs):
    """Evaluate scalping model"""
    logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Evaluating scalping model for {symbol}...")
    
    env = ScalpingTradingEnv(df=test_df, symbol=symbol, **env_kwargs)
    all_metrics = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        metrics = env.calculate_performance_metrics()
        all_metrics.append(metrics)
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Holding Time: {metrics['avg_holding_time']:.1f} bars")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
    
    # Calculate averages
    avg_metrics = {
        'symbol': symbol,
        'avg_total_return': np.mean([m['total_return'] for m in all_metrics]),
        'avg_annual_return': np.mean([m['annual_return'] for m in all_metrics]),
        'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
        'avg_profit_factor': np.mean([m['profit_factor'] for m in all_metrics if m['profit_factor'] != float('inf')]),
        'avg_sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics]),
        'avg_max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
        'avg_trades': np.mean([m['total_trades'] for m in all_metrics]),
        'avg_holding_time': np.mean([m['avg_holding_time'] for m in all_metrics]),
    }
    
    return all_metrics, avg_metrics

def main():
    """Main function for scalping model training"""
    print("🔥 Starting Scalping RL Model Training (5-minute timeframe)")
    
    # Create directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    
    for symbol in SYMBOLS:
        print(f"\n📊 Processing {symbol} for scalping...")
        
        try:
            # Load 5-minute data
            train_df, test_df = load_and_prepare_scalping_data(symbol)
            
            # Environment parameters for scalping
            env_kwargs = {
                'initial_balance': DEFAULT_INITIAL_BALANCE,
                'window_size': 20,
                'scalp_target': SCALP_TARGET_PROFIT,
                'scalp_stop': SCALP_STOP_LOSS,
                'max_position_time': MAX_POSITION_TIME_BARS,
                'position_size_pct': SCALP_POSITION_SIZE_PCT
            }
            
            model_path = f"{MODEL_SAVE_PATH}/{symbol}_scalping.zip"
            
            if os.path.exists(model_path):
                print(f"📁 Loading existing scalping model: {model_path}")
                model = PPO.load(model_path)
            else:
                print(f"🎯 Training new scalping model for {symbol}")
                
                # Hyperparameter optimization
                study_name = f"scalping_{symbol}"
                storage_url = f"sqlite:///scalping_optuna/{study_name}.db"
                
                os.makedirs("scalping_optuna", exist_ok=True)
                
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_url,
                    direction='minimize',
                    sampler=TPESampler(n_startup_trials=5),
                    load_if_exists=True
                )
                
                print("🔍 Optimizing hyperparameters for scalping...")
                study.optimize(
                    lambda trial: optimize_scalping_hyperparameters(trial, train_df, symbol, **env_kwargs),
                    n_trials=30,  # Reduced for faster development
                    n_jobs=1
                )
                
                best_params = study.best_trial.params
                print(f"✅ Best hyperparameters: {best_params}")
                
                # Train final model
                policy_kwargs = dict(
                    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
                )
                
                model = PPO(
                    'MlpPolicy',
                    create_scalping_vec_env(train_df, symbol, **env_kwargs),
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    **best_params
                )
                
                print("🚀 Training scalping model...")
                model.learn(total_timesteps=200_000, progress_bar=True)
                
                model.save(model_path)
                print(f"💾 Model saved: {model_path}")
            
            # Evaluate model
            print(f"📈 Evaluating scalping model for {symbol}")
            all_metrics, avg_metrics = evaluate_scalping_model(
                model, test_df, symbol, num_episodes=3, **env_kwargs
            )
            
            # Save results
            results_file = f"{RESULTS_SAVE_PATH}/{symbol}_scalping_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'avg_metrics': avg_metrics,
                    'all_metrics': all_metrics
                }, f, indent=4, default=str)
            
            print(f"💾 Results saved: {results_file}")
            print(f"📊 Average Performance Summary for {symbol}:")
            print(f"   💰 Total Return: {avg_metrics['avg_total_return']:.2%}")
            print(f"   🎯 Win Rate: {avg_metrics['avg_win_rate']:.2%}")
            print(f"   📈 Profit Factor: {avg_metrics['avg_profit_factor']:.2f}")
            print(f"   ⚡ Avg Trades: {avg_metrics['avg_trades']:.0f}")
            print(f"   ⏱️ Avg Hold Time: {avg_metrics['avg_holding_time']:.1f} bars")
            
        except Exception as e:
            logger.log(LOG_LEVELS['EXCEPTION'], f"Error processing {symbol}: {str(e)}")
            print(f"❌ Error processing {symbol}: {str(e)}")
            continue
    
    print("\n🎉 Scalping model training completed!")

if __name__ == "__main__":
    main()
