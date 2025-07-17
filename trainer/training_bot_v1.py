import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os
import json
from finta import TA
from tqdm import tqdm
from rich import print
from rich.live import Live
from rich.table import Table
import time
import logging
import warnings
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tabulate import tabulate
import torch
from stable_baselines3.common.monitor import Monitor
from dotenv import load_dotenv
import pytz

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RL_Trading_Bot")

SYMBOLS = ["ADANIPORTS","BHARTIARTL"]

# SYMBOLS = ['BHARTIARTL','ZEEL','BAJFINANCE','BRITANNIA','TATAMOTORS','COALINDIA','HCLTECH','TITAN','MM','CIPLA','IOC','SUNPHARMA','DRREDDY','POWERGRID','KOTAKBANK','ULTRACEMCO','AXISBANK','SHREECEM','ITC','SBIN','INFY','TCS','GRASIM','UPL','ADANIPORTS','JSWSTEEL','RELIANCE','EICHERMOT','VEDL','LT','GAIL','NTPC','HINDUNILVR','BAJAJFINSV','BAJAJ-AUTO','ICICIBANK','NESTLEIND','MARUTI','TECHM','BPCL','ASIANPAINT','TATASTEEL','HEROMOTOCO','HINDALCO','ONGC','WIPRO','INDUSINDBK']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
load_dotenv()
API_KEY = os.getenv("ZERODHA_API_KEY")
API_SECRET = os.getenv("ZERODHA_API_SECRET")
ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN")
DEFAULT_INITIAL_BALANCE = float(os.getenv("DEFAULT_INITIAL_BALANCE", "100000"))
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", "models")
DATA_PATH = os.getenv("DATA_PATH", "datasets/{}.csv")
RESULTS_SAVE_PATH = os.getenv("RESULTS_SAVE_PATH", "results")
BROKERAGE_INTRADAY = float(os.getenv("BROKERAGE_INTRADAY", "0.0003"))
BROKERAGE_DELIVERY = float(os.getenv("BROKERAGE_DELIVERY", "0.0"))
STT_INTRADAY = float(os.getenv("STT_INTRADAY", "0.00025"))
STT_DELIVERY = float(os.getenv("STT_DELIVERY", "0.001"))
EXCHANGE_TXN_CHARGE = float(os.getenv("EXCHANGE_TXN_CHARGE", "0.0000345"))
SEBI_CHARGES = float(os.getenv("SEBI_CHARGES", "0.000001"))
STAMP_DUTY = float(os.getenv("STAMP_DUTY", "0.00015"))
GST = float(os.getenv("GST", "0.18"))

# Tax Settings
SHORT_TERM_CAPITAL_GAINS_TAX = float(os.getenv("SHORT_TERM_CAPITAL_GAINS_TAX", "0.15"))
LONG_TERM_CAPITAL_GAINS_TAX = float(os.getenv("LONG_TERM_CAPITAL_GAINS_TAX", "0.10"))
LONG_TERM_THRESHOLD_DAYS = float(os.getenv("LONG_TERM_THRESHOLD_DAYS", "365"))

TIMEZONE = pytz.timezone('Asia/Kolkata')

# Logging levels for specific events
LOG_LEVELS = {
    'DATA_LOAD': logging.INFO,
    'MODEL_TRAINING': logging.INFO,
    'MODEL_EVALUATION': logging.INFO,
    'PERFORMANCE_METRICS': logging.INFO,
    'EXCEPTION': logging.ERROR
}

# Trading Environment
class IndianStockTradingEnv(gym.Env):
    """Custom Environment for trading Indian stocks using RL with advanced features"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, symbol, initial_balance=DEFAULT_INITIAL_BALANCE, window_size=20,
                 use_trailing_stop=True, atr_periods=14, trailing_stop_atr_multiplier=2,
                 target_atr_multiplier=3, min_rr_ratio=1.5):
        super(IndianStockTradingEnv, self).__init__()
        
        # Data setup
        self.df = df
        self.symbol = symbol
        self.window_size = window_size
        self.initial_balance = initial_balance
        
        # Stop loss and target parameters
        self.use_trailing_stop = use_trailing_stop
        self.atr_periods = atr_periods
        self.trailing_stop_atr_multiplier = trailing_stop_atr_multiplier
        self.target_atr_multiplier = target_atr_multiplier
        self.min_rr_ratio = min_rr_ratio
        
        # Calculate ATR for dynamic stop loss and targets
        self._calculate_atr()
        
        # Account variables
        self.balance = initial_balance
        self.net_worth = initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = initial_balance
        
        # Position management
        self.stop_loss = 0
        self.trailing_stop = 0
        self.target_price = 0
        self.highest_price_since_buy = 0
        self.position_type = None  # 'long' or 'short'
        self.last_txn_cost = 0
        
        # Episode variables
        self.current_step = 0
        self.rewards = []
        self.trades = []
        self.net_worth_history = [initial_balance]
        
        # Define action and observation spaces
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: balance, shares, cost_basis, current_price, technical indicators, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(53,), dtype=np.float32
        )

    def calculate_transaction_costs(self, trade_value, is_intraday=False, verbose=False):
        brokerage_rate = min(BROKERAGE_INTRADAY if is_intraday else BROKERAGE_DELIVERY, 0.05)
        stt_rate = min(STT_INTRADAY if is_intraday else STT_DELIVERY, 0.05)
        exchange_rate = min(EXCHANGE_TXN_CHARGE, 0.01)
        sebi_rate = min(SEBI_CHARGES, 0.01)
        stamp_rate = min(STAMP_DUTY, 0.01)
        gst_rate = min(GST, 0.30) if GST < 1 else 0.18

        brokerage = trade_value * brokerage_rate
        stt = trade_value * stt_rate
        exchange_charge = trade_value * exchange_rate
        sebi_charges = trade_value * sebi_rate
        stamp_duty = trade_value * stamp_rate
        gst = (brokerage + exchange_charge) * gst_rate

        total_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst

        max_allowed_cost = trade_value * 0.10
        if total_cost > max_allowed_cost:
            if verbose:
                print(f"WARNING: Transaction costs capped from {total_cost:.2f} to {max_allowed_cost:.2f}")
            total_cost = max_allowed_cost

        return round(total_cost, 2)


    
    def calculate_post_tax_profit(self, buy_price, sell_price, quantity, holding_period_days):
        if quantity <= 0:
            return 0.0

        gross_profit = (sell_price - buy_price) * quantity
        buy_cost = self.calculate_transaction_costs(buy_price * quantity)
        sell_cost = self.calculate_transaction_costs(sell_price * quantity)
        pre_tax_profit = gross_profit - buy_cost - sell_cost

        if holding_period_days >= LONG_TERM_THRESHOLD_DAYS:
            tax_rate = LONG_TERM_CAPITAL_GAINS_TAX
        else:
            tax_rate = SHORT_TERM_CAPITAL_GAINS_TAX

        capital_gains_tax = max(0, pre_tax_profit * tax_rate)
        return pre_tax_profit - capital_gains_tax

    
    def _calculate_atr(self):
        """Calculate Average True Range using Finta"""
        if 'ATR' not in self.df.columns:
            self.df['ATR'] = TA.ATR(self.df, period=self.atr_periods)
        
        # Replace inf/-inf with default value (2% of price)
        atr_default = self.df['Close'] * 0.02
        self.df['ATR'] = self.df['ATR'].replace([np.inf, -np.inf], np.nan)
        self.df['ATR'] = self.df['ATR'].fillna(atr_default).bfill()

        # Safety: still ensure no NaNs
        self.df['ATR'].fillna(method='bfill', inplace=True)



    def _calculate_dynamic_stop_loss(self, price, position_type):
        """Calculate ATR-based dynamic stop-loss with reasonable bounds"""
        current_atr = self.df.iloc[self.current_step]['ATR']
        atr_multiplier = self.trailing_stop_atr_multiplier

        stop_distance = current_atr * atr_multiplier
        max_stop_pct = 0.2  # Cap stop distance to 20% of price
        stop_distance = min(stop_distance, price * max_stop_pct)

        if position_type == 'long':
            return price - stop_distance
        else:
            return price + stop_distance

    
    def _calculate_dynamic_target(self, entry_price, position_type):
        """Calculate target price using ATR and risk-reward ratio"""
        current_atr = self.df.iloc[self.current_step]['ATR']
        stop_distance = current_atr * self.trailing_stop_atr_multiplier

        # Choose max of risk-reward and target multiplier logic
        target_distance = max(
            stop_distance * self.min_rr_ratio,
            current_atr * self.target_atr_multiplier
        )

        # Optional cap to avoid unrealistic targets
        max_target_pct = 0.4
        target_distance = min(target_distance, entry_price * max_target_pct)

        return entry_price + target_distance if position_type == 'long' else entry_price - target_distance
    
    def _update_trailing_stop(self, current_price):
        """Update trailing stop loss based on favorable price move"""
        if not self.position_type:
            return

        if self.position_type == 'long':
            if current_price > self.highest_price_since_buy:
                self.highest_price_since_buy = current_price
                new_stop = self._calculate_dynamic_stop_loss(current_price, 'long')
                self.trailing_stop = max(self.trailing_stop, new_stop)
        else:  # short
            if current_price < self.highest_price_since_buy:
                self.highest_price_since_buy = current_price
                new_stop = self._calculate_dynamic_stop_loss(current_price, 'short')
                self.trailing_stop = min(self.trailing_stop, new_stop)

    
    def _check_stop_and_target(self, current_price):
        """Check if stop-loss or target was hit"""
        if not self.position_type:
            return False, None

        exit_type = None

        if self.position_type == 'long':
            if current_price <= self.trailing_stop:
                exit_type = 'stop_loss'
            elif current_price >= self.target_price:
                exit_type = 'target'
        else:  # short
            if current_price >= self.trailing_stop:
                exit_type = 'stop_loss'
            elif current_price <= self.target_price:
                exit_type = 'target'

        return exit_type is not None, exit_type

    
    def _get_observation(self):
        """Get the current trading state observation with additional indicators"""
        # Get current price data
        current_price = self.df.iloc[self.current_step]["Close"]
        
        # Basic account state features
        obs = np.array([
            self.balance / self.initial_balance,
            self.shares_held / 100,  # Normalize shares
            self.cost_basis / current_price if self.cost_basis > 0 else 0,
            current_price / self.df["Close"].iloc[max(0, self.current_step-20):self.current_step+1].mean(),  # Normalized price
            1 if self.position_type == 'long' else 0,  # Is in long position
            1 if self.position_type == 'short' else 0,  # Is in short position
            self.trailing_stop / current_price if self.trailing_stop > 0 else 0,  # Normalized trailing stop
            self.target_price / current_price if self.target_price > 0 else 0,  # Normalized target
        ])
        
        # Add technical indicators from current row
        if self.current_step >= self.window_size:
            window_data = self.df.loc[self.current_step-self.window_size+1:self.current_step+1, :]

            # Normalize OHLC
            close_mean = window_data["Close"].mean()
            obs = np.append(obs, [
                window_data.loc[window_data.index[-1], "Open"] / close_mean,
                window_data.loc[window_data.index[-1], "High"] / close_mean,
                window_data.loc[window_data.index[-1], "Low"] / close_mean,
                window_data.loc[window_data.index[-1], "Close"] / close_mean,
                window_data.loc[window_data.index[-1], "Volume"] / window_data["Volume"].mean(),
            ])
            
            # Add normalized SMA 5, 10, 20, 50 days
            for period in [5, 10, 20, 50]:
                if f'SMA{period}' in window_data.columns:
                    sma = window_data[f'SMA{period}'].iloc[-1] / close_mean
                else:
                    sma = window_data["Close"].rolling(window=min(period, len(window_data))).mean().iloc[-1] / close_mean
                obs = np.append(obs, [sma])
            
            # Add EMA 5, 10, 20, 50 days
            for period in [5, 10, 20, 50]:
                if f'EMA{period}' in window_data.columns:
                    ema = window_data[f'EMA{period}'].iloc[-1] / close_mean
                else:
                    ema = window_data["Close"].ewm(span=min(period, len(window_data)), adjust=False).mean().iloc[-1] / close_mean
                obs = np.append(obs, [ema])
            
            # Add RSI
            if 'RSI' in window_data.columns:
                rsi = window_data['RSI'].iloc[-1] / 100  # Normalize RSI
            else:
                rsi = 0.5  # Default middle value if not available
            obs = np.append(obs, [rsi])
            
            # Add MACD
            if all(x in window_data.columns for x in ['MACD', 'Signal']):
                macd = window_data['MACD'].iloc[-1] / close_mean
                signal = window_data['Signal'].iloc[-1] / close_mean
            else:
                macd = 0
                signal = 0
            obs = np.append(obs, [macd, signal])
            
            # Add Bollinger Bands
            if all(x in window_data.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                upper_band = window_data['BB_Upper'].iloc[-1] / close_mean
                middle_band = window_data['BB_Middle'].iloc[-1] / close_mean
                lower_band = window_data['BB_Lower'].iloc[-1] / close_mean
            else:
                upper_band = 1.1
                middle_band = 1.0
                lower_band = 0.9
            obs = np.append(obs, [upper_band, middle_band, lower_band])
            
            # Add price momentum indicators
            ret_1d = window_data['Close'].pct_change(1).iloc[-1]
            ret_5d = window_data['Close'].pct_change(5).iloc[-1] if len(window_data) > 5 else 0
            ret_20d = window_data['Close'].pct_change(20).iloc[-1] if len(window_data) > 20 else 0
            obs = np.append(obs, [ret_1d, ret_5d, ret_20d])
            
            # === NEW INDICATORS ===
            # 1. Volume-Based Indicators
            # OBV (On-Balance Volume)
            if 'OBV' in window_data.columns:
                obv = window_data['OBV'].iloc[-1] / window_data['OBV'].max() if window_data['OBV'].max() != 0 else 0
            else:
                obv = 0.5
            obs = np.append(obs, [obv])
            
            # Chaikin Money Flow
            if 'CMF' in window_data.columns:
                cmf = window_data['CMF'].iloc[-1]  # Already normalized between -1 and 1
                cmf = (cmf + 1) / 2  # Normalize to 0-1 range
            else:
                cmf = 0.5
            obs = np.append(obs, [cmf])
            
            # 2. Trend Indicators
            # ADX (Average Directional Index)
            if 'ADX' in window_data.columns:
                adx = window_data['ADX'].iloc[-1] / 100  # Normalize ADX to 0-1
            else:
                adx = 0.5
            obs = np.append(obs, [adx])
            
            # Parabolic SAR
            if 'SAR' in window_data.columns:
                sar = window_data['SAR'].iloc[-1] / close_mean  # Normalize relative to close
                # Direction indicator: 1 if SAR below price (bullish), 0 if above (bearish)
                sar_dir = 1 if window_data['SAR'].iloc[-1] < window_data['Close'].iloc[-1] else 0
            else:
                sar = 1
                sar_dir = 0.5
            obs = np.append(obs, [sar, sar_dir])
            
            # Ichimoku Cloud
            if all(x in window_data.columns for x in ['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A', 'ICHIMOKU_SENKOU_B']):
                tenkan = window_data['ICHIMOKU_TENKAN'].iloc[-1] / close_mean
                kijun = window_data['ICHIMOKU_KIJUN'].iloc[-1] / close_mean
                senkou_a = window_data['ICHIMOKU_SENKOU_A'].iloc[-1] / close_mean
                senkou_b = window_data['ICHIMOKU_SENKOU_B'].iloc[-1] / close_mean
                # Cloud direction: 1 if bullish (A > B), 0 if bearish (A < B)
                cloud_direction = 1 if senkou_a > senkou_b else 0
                # Price relative to cloud: 1 if above both, 0.5 if between, 0 if below both
                current_close = window_data['Close'].iloc[-1] / close_mean
                if current_close > max(senkou_a, senkou_b):
                    price_cloud_position = 1
                elif current_close < min(senkou_a, senkou_b):
                    price_cloud_position = 0
                else:
                    price_cloud_position = 0.5
            else:
                tenkan, kijun, senkou_a, senkou_b = 1, 1, 1, 1
                cloud_direction, price_cloud_position = 0.5, 0.5
            obs = np.append(obs, [tenkan, kijun, cloud_direction, price_cloud_position])
            
            # 3. Oscillators
            # Stochastic Oscillator
            if all(x in window_data.columns for x in ['STOCH_K', 'STOCH_D']):
                stoch_k = window_data['STOCH_K'].iloc[-1] / 100  # Normalize to 0-1
                stoch_d = window_data['STOCH_D'].iloc[-1] / 100  # Normalize to 0-1
            else:
                stoch_k, stoch_d = 0.5, 0.5
            obs = np.append(obs, [stoch_k, stoch_d])
            
            # CCI (Commodity Channel Index)
            if 'CCI' in window_data.columns:
                # Normalize CCI from typical -100 to +100 range to 0-1
                cci = window_data['CCI'].iloc[-1]
                cci_norm = (min(max(cci, -200), 200) + 200) / 400  # Clip at -200/+200 and normalize
            else:
                cci_norm = 0.5
            obs = np.append(obs, [cci_norm])
            
            # Williams %R
            if 'WILLIAMS' in window_data.columns:
                # Williams %R ranges from -100 to 0, normalize to 0-1
                williams = window_data['WILLIAMS'].iloc[-1]
                williams_norm = (williams + 100) / 100
            else:
                williams_norm = 0.5
            obs = np.append(obs, [williams_norm])
            
            # 4. Volatility Indicators
            # Standard Deviation
            if 'STD20' in window_data.columns:
                # Normalize relative to price
                std = window_data['STD20'].iloc[-1] / close_mean
            else:
                std = 0.02  # Default ~2% volatility
            obs = np.append(obs, [std])
            
            # Keltner Channels
            if all(x in window_data.columns for x in ['KELTNER_UPPER', 'KELTNER_MIDDLE', 'KELTNER_LOWER']):
                keltner_upper = window_data['KELTNER_UPPER'].iloc[-1] / close_mean
                keltner_middle = window_data['KELTNER_MIDDLE'].iloc[-1] / close_mean
                keltner_lower = window_data['KELTNER_LOWER'].iloc[-1] / close_mean
                # Position relative to Keltner: 1 if above upper, 0 if below lower, between 0-1 otherwise
                current_close = window_data['Close'].iloc[-1]
                keltner_position = (current_close - window_data['KELTNER_LOWER'].iloc[-1]) / \
                                (window_data['KELTNER_UPPER'].iloc[-1] - window_data['KELTNER_LOWER'].iloc[-1])
                keltner_position = max(0, min(1, keltner_position))  # Clip to 0-1
            else:
                keltner_upper, keltner_middle, keltner_lower = 1.1, 1, 0.9
                keltner_position = 0.5
            obs = np.append(obs, [keltner_upper, keltner_middle, keltner_lower, keltner_position])
            
            # 5. Advanced Price Patterns
            # Moving Average Crossovers
            if all(x in window_data.columns for x in ['GOLDEN_CROSS', 'DEATH_CROSS']):
                golden_cross = window_data['GOLDEN_CROSS'].iloc[-1]
                death_cross = window_data['DEATH_CROSS'].iloc[-1]
            else:
                golden_cross, death_cross = 0, 0
            obs = np.append(obs, [golden_cross, death_cross])
            
            # Price Rate of Change (ROC)
            if 'ROC5' in window_data.columns and 'ROC20' in window_data.columns:
                # Normalize ROC to 0-1 range (assuming typical range of -20% to +20%)
                roc5 = (window_data['ROC5'].iloc[-1] + 20) / 40
                roc20 = (window_data['ROC20'].iloc[-1] + 20) / 40
                roc5 = max(0, min(1, roc5))  # Clip to 0-1
                roc20 = max(0, min(1, roc20))  # Clip to 0-1
            else:
                roc5, roc20 = 0.5, 0.5
            obs = np.append(obs, [roc5, roc20])
            
            # 6. Market Regime Indicators
            # Volatility Regime
            if 'VOL_REGIME' in window_data.columns:
                # Normalize volatility regime (usually around 1.0)
                vol_regime = min(window_data['VOL_REGIME'].iloc[-1], 3) / 3  # Cap at 3x
            else:
                vol_regime = 0.33  # Default neutral
            obs = np.append(obs, [vol_regime])
            
        else:
            # Fill with zeros if not enough data
            # 22 from original features + 30 from new features
            obs = np.append(obs, np.zeros(52))
        
        # return obs.astype(np.float32)
        return np.nan_to_num(obs).astype(np.float32)


    def calculate_reward(self, action, exit_type=None):
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.shares_held * current_price
        net_worth_change = self.net_worth - prev_net_worth

        reward = net_worth_change / self.initial_balance * 100

        # Encourage holding profitable positions
        if action == 0 and self.shares_held > 0 and net_worth_change > 0:
            reward += 0.1

        # Buy
        if action == 1:
            self.purchase_step = self.current_step
            self.purchase_price = current_price
            reward -= 0.1  # transaction penalty
            reward += 0.2  # small incentive

        # Sell
        elif action == 2 and self.shares_held > 0:
            if hasattr(self, 'purchase_step') and hasattr(self, 'purchase_price'):
                holding_period_days = self.current_step - self.purchase_step
                quantity = self.shares_held
                post_tax_profit = self.calculate_post_tax_profit(
                    self.purchase_price, current_price, quantity, holding_period_days
                )
                reward = post_tax_profit / self.initial_balance * 100
                if exit_type == 'target':
                    reward += 2.0
                elif exit_type == 'stop_loss':
                    reward -= 0.25

        # Holding drawdown penalty
        if self.shares_held > 0:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            if current_drawdown > 0.1:
                reward -= current_drawdown * 5

        # Overtrading penalty
        if hasattr(self, 'recent_actions'):
            self.recent_actions.append(action)
            if len(self.recent_actions) > 10:
                self.recent_actions.pop(0)
            if self.recent_actions.count(1) + self.recent_actions.count(2) > 5:
                reward -= 0.5

        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        return reward


    def step(self, action):
        """Execute one step in the environment"""
        # Get current price
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        exit_type = None
        
        # Check if stop loss or target hit before handling action
        if self.shares_held > 0:
            self._update_trailing_stop(current_price)
            stop_or_target_hit, exit_type = self._check_stop_and_target(current_price)
            
            if stop_or_target_hit:
                # Force sell if stop loss or target hit
                action = 2
        
        # Execute trading action
        if action == 0:  # Hold
            pass
        
        elif action == 1:  # Buy
            if self.balance > 0 and self.shares_held == 0:  # Only buy if not in position
                # Calculate max shares to buy (round down to nearest whole share)
                max_shares = int(self.balance // current_price)

                # print(f"Step {self.current_step}: Attempting to BUY at price {current_price}")

                if max_shares > 0:
                    shares_bought = max_shares

                    trade_value = shares_bought * current_price
                    txn_costs = self.calculate_transaction_costs(trade_value, is_intraday=False)  
                    self.last_txn_cost = txn_costs
                    cost = trade_value + txn_costs
                    
                    # Update account
                    self.balance -= cost
                    self.shares_held += shares_bought
                    self.total_shares_bought += shares_bought
                    self.cost_basis = current_price
                    self.total_trades += 1
                    
                    # Set position type
                    self.position_type = 'long'
                    
                    # Set stop loss and target
                    self.stop_loss = self._calculate_dynamic_stop_loss(current_price, 'long')
                    self.trailing_stop = self.stop_loss
                    self.target_price = self._calculate_dynamic_target(current_price, 'long')
                    self.highest_price_since_buy = current_price

                    # print(f"Bought {shares_bought} shares at {current_price}, cost: {cost}, txn_costs: {txn_costs}")

                    # Log trade
                    self.trades.append({
                        'step': self.current_step,
                        'date': self.df.iloc[self.current_step]['Date'] if 'Date' in self.df.columns else self.current_step,
                        'price': current_price,
                        'type': 'buy',
                        'shares': shares_bought,
                        'cost': cost,
                        'stop_loss': self.stop_loss,
                        'trailing_stop': self.trailing_stop,
                        'target': self.target_price
                    })
        
        elif action == 2:  # Sell
            if self.shares_held > 0:
                # print(f"Step {self.current_step}: Attempting to SELL at price {current_price}")

                # Sell all shares
                shares_sold = self.shares_held
                revenue = shares_sold * current_price

                # Calculate transaction costs
                txn_costs = self.calculate_transaction_costs(revenue, is_intraday=False)
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
                
                # Reset position tracking
                position_type = self.position_type
                self.position_type = None
                self.stop_loss = 0
                self.trailing_stop = 0
                self.target_price = 0
                self.highest_price_since_buy = 0

                # print(f"Sold {shares_sold} shares at {current_price}, revenue: {revenue}, txn_costs: {txn_costs}")
                
                # Log trade with exit type
                self.trades.append({
                    'step': self.current_step,
                    'date': self.df.iloc[self.current_step]['Date'] if 'Date' in self.df.columns else self.current_step,
                    'price': current_price,
                    'type': 'sell',
                    'shares': shares_sold,
                    'revenue': revenue,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'exit_type': exit_type if exit_type else 'manual',
                    'position_type': position_type
                })
        
        # Calculate reward
        reward = self.calculate_reward(action, exit_type)
        self.rewards.append(reward)
        
        # Update net worth history
        self.net_worth = self.balance + self.shares_held * current_price
        self.net_worth_history.append(self.net_worth)
        
        # Update maximum drawdown
        if self.net_worth < self.max_net_worth:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.df) - 1
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
            'target_price': self.target_price
        }
   
        return obs, reward, done, truncated, info

    
    def reset(self, seed=None, options=None):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_bought = 0
        self.total_shares_sold = 0
        self.total_trades = 0
        self.max_drawdown = 0
        self.max_net_worth = self.initial_balance

        self.stop_loss = 0
        self.trailing_stop = 0
        self.target_price = 0
        self.highest_price_since_buy = 0
        self.position_type = None

        self.current_step = self.window_size
        self.rewards = []
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        self.total_transaction_costs = 0
        self.total_trade_value = 0
        self.last_txn_cost = 0

        self.recent_actions = []

        return self._get_observation(), {}

    
    def render(self, mode='human', close=False):
        if mode == 'human':
            profit = self.net_worth - self.initial_balance
            print(f"Step: {self.current_step}")
            print(f"Symbol: {self.symbol}")
            print(f"Balance: ₹{self.balance:.2f}")
            print(f"Shares held: {self.shares_held}")
            print(f"Net worth: ₹{self.net_worth:.2f}")
            print(f"Profit: ₹{profit:.2f} ({(profit/self.initial_balance)*100:.2f}%)")
            print(f"Max Drawdown: {self.max_drawdown:.2%}")
            if self.position_type:
                print(f"Position: {self.position_type}")
                print(f"Trailing Stop: ₹{self.trailing_stop:.2f}")
                print(f"Target: ₹{self.target_price:.2f}")
    
    def calculate_performance_metrics(self):
        """Calculate trading performance metrics"""
        # Calculate daily returns
        returns = [0]
        for i in range(1, len(self.net_worth_history)):
            returns.append((self.net_worth_history[i] - self.net_worth_history[i-1]) / self.net_worth_history[i-1])
        
        # Calculate metrics
        total_return = (self.net_worth_history[-1] - self.initial_balance - self.total_transaction_costs) / self.initial_balance
        
        # Annualize return (assuming 252 trading days in a year)
        n_days = len(self.net_worth_history)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        # Calculate daily returns std and Sharpe ratio
        std_daily_return = np.std(returns[1:])
        risk_free_rate = 0.05  # 5% annual risk-free rate
        daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
        sharpe_ratio = ((np.mean(returns[1:]) - daily_risk_free) / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0
        
        # Calculate max drawdown
        max_dd = self.max_drawdown
        
        # Calculate win rate and stats
        win_count = 0
        loss_count = 0
        total_profit = 0
        total_loss = 0
        
        for trade in self.trades:
            if trade['type'] == 'sell':
                if 'profit' in trade and trade['profit'] > 0:
                    win_count += 1
                    total_profit += trade['profit']
                else:
                    loss_count += 1
                    if 'profit' in trade:
                        total_loss += abs(trade['profit'])
        
        total_completed_trades = win_count + loss_count
        win_rate = win_count / total_completed_trades if total_completed_trades > 0 else 0
        
        # Average profit/loss
        avg_profit = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Risk-adjusted return - Calmar Ratio (Annual Return / Max Drawdown)
        calmar_ratio = annual_return / max_dd if max_dd > 0 else float('inf')
        
        # Exit type statistics
        exit_types = {'target': 0, 'stop_loss': 0, 'manual': 0}
        for trade in self.trades:
            if trade['type'] == 'sell' and 'exit_type' in trade:
                exit_type = trade['exit_type']
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
        
        return {
            'symbol': self.symbol,
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
            'exit_types': exit_types,
            'net_worth': self.net_worth_history[-1],
            'total_transaction_costs': self.total_transaction_costs,
            'transaction_cost_ratio': self.total_transaction_costs / self.total_trade_value if self.total_trade_value > 0 else 0,
        }

    def plot_performance(self, save_path=None):
        """Plot trading performance"""
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Get price data and dates
        price_data = self.df.iloc[self.window_size:self.current_step+1]['Close'].values
        
        if 'Date' in self.df.columns:
            dates = pd.to_datetime(self.df.iloc[self.window_size:self.current_step+1]['Date'])
            x_values = dates
        else:
            x_values = range(len(price_data))
        
        # Plot net worth
        ax1.plot(x_values, self.net_worth_history, 'b-', label=f'Net Worth ({self.symbol})')
        ax1.set_title(f'{self.symbol} - Net Worth Over Time')
        ax1.set_ylabel('Net Worth (₹)')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Format date axis if dates are available
        if 'Date' in self.df.columns:
            plt.setp(ax1.get_xticklabels(), rotation=45)
            ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        
        # Plot stock price
        ax2.plot(x_values, price_data, 'k-', label=f'{self.symbol} Price')
        
        # Add stop loss and target lines for each trade
        buy_steps = []
        sell_steps = []
        stop_losses = []
        targets = []
        
        for trade in self.trades:
            step_idx = trade['step'] - self.window_size
            if step_idx >= 0 and step_idx < len(price_data):
                if trade['type'] == 'buy':
                    buy_steps.append(step_idx)
                    # Add stop loss and target to lists
                    stop_losses.append(trade['stop_loss'])
                    targets.append(trade['target'])
                    # Plot stop loss and target lines
                    if 'Date' in self.df.columns:
                        ax2.axhline(y=trade['stop_loss'], color='r', linestyle='--', alpha=0.3)
                        ax2.axhline(y=trade['target'], color='g', linestyle='--', alpha=0.3)
                    else:
                        ax2.axhline(y=trade['stop_loss'], color='r', linestyle='--', alpha=0.3)
                        ax2.axhline(y=trade['target'], color='g', linestyle='--', alpha=0.3)
                elif trade['type'] == 'sell':
                    sell_steps.append(step_idx)
        
        # Plot buy and sell markers
        if 'Date' in self.df.columns:
            dates = pd.to_datetime(self.df.iloc[self.window_size:self.current_step+1]['Date']).reset_index(drop=True)

            buy_dates = [dates[i] for i in buy_steps]
            sell_dates = [dates[i] for i in sell_steps]
            buy_prices = [price_data[i] for i in buy_steps]
            sell_prices = [price_data[i] for i in sell_steps]
            
            ax2.plot(buy_dates, buy_prices, '^', markersize=8, color='g', label='Buy')
            ax2.plot(sell_dates, sell_prices, 'v', markersize=8, color='r', label='Sell')
        else:
            buy_prices = [price_data[i] for i in buy_steps]
            sell_prices = [price_data[i] for i in sell_steps]
            
            ax2.plot(buy_steps, buy_prices, '^', markersize=8, color='g', label='Buy')
            ax2.plot(sell_steps, sell_prices, 'v', markersize=8, color='r', label='Sell')
        
        ax2.set_title(f'{self.symbol} - Price and Trades')
        ax2.set_ylabel('Price (₹)')
        ax2.legend(loc='upper left')
        ax2.grid(True)
        
        if 'Date' in self.df.columns:
            plt.setp(ax2.get_xticklabels(), rotation=45)
            ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        
        # Plot drawdown
        drawdowns = []
        peak = self.initial_balance
        for worth in self.net_worth_history:
            if worth > peak:
                peak = worth
            drawdown = (peak - worth) / peak
            drawdowns.append(drawdown)
        
        ax3.plot(x_values, drawdowns, 'r-', label='Drawdown')
        ax3.fill_between(x_values, drawdowns, color='red', alpha=0.3)
        ax3.set_title(f'{self.symbol} - Drawdown')
        ax3.set_ylabel('Drawdown (%)')
        ax3.set_ylim(0, max(drawdowns) * 1.1)  # Add 10% margin to top
        ax3.legend(loc='upper left')
        ax3.grid(True)
        
        if 'Date' in self.df.columns:
            plt.setp(ax3.get_xticklabels(), rotation=45)
            ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
            
        plt.close()
        
        return fig

def create_vec_env(df, symbol, **kwargs):
    """Create a vectorized environment"""
    num_envs = 16
    def make_env():
        return lambda: Monitor(IndianStockTradingEnv(df=df, symbol=symbol, **kwargs))

    return DummyVecEnv([make_env() for _ in range(num_envs)])

# Custom Chaikin Money Flow implementation
def chaikin_money_flow(df, period=20):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(period).sum() / df['Volume'].rolling(period).sum()
    return cmf

def calculate_additional_indicators(df):
    """Calculate all the additional technical indicators using Finta"""
    print("Calculating additional technical indicators...")
    
    # 1. Volume-Based Indicators
    # OBV (On-Balance Volume)
    if 'OBV' not in df.columns:
        df['OBV'] = TA.OBV(df)
        
    # Chaikin Money Flow
    if 'CMF' not in df.columns:
        df['CMF'] = chaikin_money_flow(df)
        
    # 2. Trend Indicators
    # ADX (Average Directional Index)
    if 'ADX' not in df.columns:
        adx = TA.ADX(df)
        df['ADX'] = adx
        
    # Parabolic SAR
    if 'SAR' not in df.columns:
        df['SAR'] = TA.SAR(df)
        
    # Ichimoku Cloud components
    if 'ICHIMOKU_TENKAN' not in df.columns:
        ichimoku = TA.ICHIMOKU(df)
        
        df['ICHIMOKU_TENKAN'] = ichimoku['TENKAN']
        df['ICHIMOKU_KIJUN'] = ichimoku['KIJUN']
        df['ICHIMOKU_SENKOU_A'] = ichimoku['senkou_span_a']  # lowercase!
        df['ICHIMOKU_SENKOU_B'] = ichimoku['SENKOU']
        df['ICHIMOKU_CHIKOU'] = ichimoku['CHIKOU']
        
    # 3. Oscillators
    # Stochastic Oscillator
    if 'STOCH_K' not in df.columns:
        stoch_k = TA.STOCH(df)  # Series of %K only
        df['STOCH_K'] = stoch_k
        df['STOCH_D'] = stoch_k.rolling(3).mean()  # Typical smoothing for %D
        
    # CCI (Commodity Channel Index)
    if 'CCI' not in df.columns:
        df['CCI'] = TA.CCI(df)
        
    # Williams %R
    if 'WILLIAMS' not in df.columns:
        df['WILLIAMS'] = TA.WILLIAMS(df)
        
    # 4. Volatility Indicators
    # Standard Deviation
    if 'STD20' not in df.columns:
        # Using rolling standard deviation of close prices
        df['STD20'] = df['Close'].rolling(window=20).std()
        
    # Keltner Channels
    if 'KELTNER_UPPER' not in df.columns:
        # Parameters
        ema_period = 20
        atr_period = 14
        multiplier = 2

        # Middle Line: EMA
        df['KC_MIDDLE'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

        # ATR Calculation (finta has ATR but we'll calculate it manually)
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TR'].rolling(window=atr_period).mean()

        # Keltner Channels
        df['KC_UPPER'] = df['KC_MIDDLE'] + multiplier * df['ATR']
        df['KC_LOWER'] = df['KC_MIDDLE'] - multiplier * df['ATR']

        # Drop helper columns (optional)
        df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR'], inplace=True)
        
    # 5. Advanced Price Patterns
    # Moving Average Crossovers
    if 'SMA10' in df.columns and 'SMA50' in df.columns:
        # Golden Cross (SMA10 crosses above SMA50)
        df['GOLDEN_CROSS'] = np.where(
            (df['SMA10'].shift(1) < df['SMA50'].shift(1)) & 
            (df['SMA10'] > df['SMA50']), 
            1, 0
        )
        # Death Cross (SMA10 crosses below SMA50)
        df['DEATH_CROSS'] = np.where(
            (df['SMA10'].shift(1) > df['SMA50'].shift(1)) & 
            (df['SMA10'] < df['SMA50']), 
            1, 0
        )
    
    # Price Rate of Change (ROC)
    if 'ROC5' not in df.columns:
        df['ROC5'] = TA.ROC(df, period=5)
        df['ROC20'] = TA.ROC(df, period=20)
    
    # 6. Market Regime Indicators
    # Volatility Regime (using ATR ratio)
    if 'VOL_REGIME' not in df.columns and 'ATR' in df.columns:
        # Short-term vs long-term ATR ratio as volatility regime indicator
        # ATR was already calculated in your original code
        short_period = 5
        long_period = 20
        
        # Create short-term ATR if not exists
        if f'ATR{short_period}' not in df.columns:
            df[f'ATR{short_period}'] = TA.ATR(df, period=short_period)
            
        # Volatility regime = short-term ATR / long-term ATR
        df['VOL_REGIME'] = df[f'ATR{short_period}'] / df['ATR']
    
    # Fill NaN values
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    print("Technical indicator calculation complete.")
    return df

def update_load_and_prepare_data(symbol, split_ratio=0.8):
    """Updated load_and_prepare_data function with additional indicators"""
    logger.log(LOG_LEVELS['DATA_LOAD'], f"Loading data for {symbol}...")

    # Load data
    data_file = DATA_PATH.format(symbol)
    df = pd.read_csv(data_file)
    
    # Ensure date column exists and is properly formatted
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate technical indicators if not already present
    if 'RSI' not in df.columns:
        df['RSI'] = TA.RSI(df, period=14)
    if 'MACD' not in df.columns:
        macd = TA.MACD(df)
        df['MACD'] = macd['MACD']
        df['Signal'] = macd['SIGNAL']
    if 'BB_Upper' not in df.columns:
        bb = TA.BBANDS(df)
        df['BB_Upper'] = bb['BB_UPPER']
        df['BB_Middle'] = bb['BB_MIDDLE']
        df['BB_Lower'] = bb['BB_LOWER']
    
    # Calculate SMAs and EMAs
    for period in [5, 10, 20, 50]:
        if f'SMA{period}' not in df.columns:
            df[f'SMA{period}'] = TA.SMA(df, period=period)
        if f'EMA{period}' not in df.columns:
            df[f'EMA{period}'] = TA.EMA(df, period=period)
    
    # Calculate all additional indicators
    df = calculate_additional_indicators(df)
    
    # Fill NaN values
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    
    # Split data into training and testing sets
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

    logger.log(LOG_LEVELS['DATA_LOAD'], f"Data preparation complete for {symbol}.")

    return train_df, test_df


def evaluate_model(model, test_df, symbol, num_episodes=1, render=False, save_results=True, **env_kwargs):
    """Evaluate trained model on test data"""
    logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Evaluating model for {symbol}...")
    
    # Create environment
    env = IndianStockTradingEnv(df=test_df, symbol=symbol, **env_kwargs)
    
    # Storage for results
    all_metrics = []
    figures = []
    
    # For detailed trade tracking
    detailed_trades = []
    capital_history = []
    drawdown_history = []
    holding_periods = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs, _ = env.reset()
        done = False
        
        # Trade tracking variables
        active_positions = {}  # Track active positions for holding period calculation
        highest_capital = env.net_worth  # For drawdown calculation
        episode_trades = []
        episode_capital = []
        episode_drawdowns = []
        
        # Get initial timestamp
        current_timestamp = test_df.iloc[env.current_step].name if hasattr(test_df.iloc[env.current_step], 'name') else f"Step_{env.current_step}"

        current_price = test_df.iloc[env.current_step]['Close'] if 'Close' in test_df.columns else None
        
        # Record initial capital
        capital_history.append({
            'episode': episode,
            'timestamp': current_timestamp,
            'capital': env.net_worth,
            'balance': env.balance,
            'shares_held': env.shares_held,
            'shares_value': env.shares_held * current_price if current_price else 0,
            'price': current_price
        })
        
        episode_capital.append({
            'episode': episode,
            'timestamp': current_timestamp,
            'capital': env.net_worth,
            'balance': env.balance,
            'shares_held': env.shares_held,
            'shares_value': env.shares_held * current_price if current_price else 0,
            'price': current_price
        })
        
        # Run episode
        while not done:
            # Get model action
            action, _states_ = model.predict(obs, deterministic=False)

            # print(f"Action distribution: {action}")

            # Record pre-action state
            pre_step_capital = env.net_worth
            pre_step_shares = env.shares_held
            pre_step_balance = env.balance
            
            # Take action in environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Get current timestamp and price
            current_timestamp = test_df.iloc[env.current_step].name if hasattr(test_df.iloc[env.current_step], 'name') else f"Step_{env.current_step}"
            current_price = test_df.iloc[env.current_step]['Close'] if 'Close' in test_df.columns else None
            
            # Record capital after action
            capital_history.append({
                'episode': episode,
                'timestamp': current_timestamp,
                'capital': env.net_worth,
                'balance': env.balance,
                'shares_held': env.shares_held,
                'shares_value': env.shares_held * current_price if current_price else 0,
                'price': current_price
            })
            
            episode_capital.append({
                'episode': episode,
                'timestamp': current_timestamp,
                'capital': env.net_worth,
                'balance': env.balance,
                'shares_held': env.shares_held,
                'shares_value': env.shares_held * current_price if current_price else 0,
                'price': current_price
            })
            
            # Calculate and record drawdown
            if env.net_worth > highest_capital:
                highest_capital = env.net_worth
            
            current_drawdown = (highest_capital - env.net_worth) / highest_capital if highest_capital > 0 else 0
            drawdown_history.append({
                'episode': episode,
                'timestamp': current_timestamp,
                'capital': env.net_worth,
                'highest_capital': highest_capital,
                'drawdown': current_drawdown,
                'price': current_price
            })
            
            episode_drawdowns.append({
                'episode': episode,
                'timestamp': current_timestamp,
                'capital': env.net_worth,
                'highest_capital': highest_capital,
                'drawdown': current_drawdown,
                'price': current_price
            })
            
            # Check for trade execution
            shares_change = env.shares_held - pre_step_shares
            
            if shares_change != 0:
                # Determine trade type and details
                trade_type = "BUY" if shares_change > 0 else "SELL"
                shares_traded = abs(shares_change)
                trade_value = shares_traded * current_price if current_price else 0
                
                # Record trade details
                trade_info = {
                    'episode': episode,
                    'timestamp': current_timestamp,
                    'action': trade_type,
                    'price': current_price,
                    'shares': shares_traded,
                    'value': trade_value,
                    'capital_before': pre_step_capital,
                    'capital_after': env.net_worth,
                    'balance_before': pre_step_balance,
                    'balance_after': env.balance,
                    'shares_after': env.shares_held,
                    'stop_loss': env.stop_loss,
                    'trailing_stop': env.trailing_stop,
                    'target_price': env.target_price,
                    'position_type': env.position_type,
                    'transaction_costs': env.last_txn_cost,
                    'net_value': trade_value - env.last_txn_cost if trade_type == "SELL" else -(trade_value + env.last_txn_cost),
                }

                # Calculate detailed transaction costs (optional)
                trade_value = shares_traded * current_price if current_price else 0
                is_intraday = False  # Adjust based on your strategy

                if trade_value > 0:
                    brokerage_rate = BROKERAGE_INTRADAY if is_intraday else BROKERAGE_DELIVERY
                    stt_rate = STT_INTRADAY if is_intraday else STT_DELIVERY
                    
                    brokerage = trade_value * brokerage_rate
                    stt = trade_value * stt_rate
                    exchange_charge = trade_value * EXCHANGE_TXN_CHARGE
                    sebi_charges = trade_value * SEBI_CHARGES
                    stamp_duty = trade_value * STAMP_DUTY
                    gst = (brokerage + exchange_charge) * GST
                    total_txn_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst
                else:
                    brokerage = stt = exchange_charge = sebi_charges = stamp_duty = gst = total_txn_cost = 0

                # Add to trade_info
                trade_info.update({
                    'brokerage': brokerage,
                    'stt': stt,
                    'exchange_charge': exchange_charge,
                    'sebi_charges': sebi_charges,
                    'stamp_duty': stamp_duty,
                    'gst': gst,
                    'total_txn_cost': total_txn_cost
                })
                
                detailed_trades.append(trade_info)
                episode_trades.append(trade_info)
                
                # Track entry for holding period calculation
                if trade_type == "BUY":
                    active_positions[current_timestamp] = {
                        'entry_time': current_timestamp,
                        'entry_price': current_price,
                        'shares': shares_traded,
                        'stop_loss': env.stop_loss,
                        'target_price': env.target_price
                    }
                elif trade_type == "SELL" and active_positions:
                    # Match this sell with the earliest open position
                    entry_timestamp = next(iter(active_positions))
                    entry_info = active_positions[entry_timestamp]

                    # Calculate holding period
                    if isinstance(current_timestamp, (int, float)) and isinstance(entry_info['entry_time'], (int, float)):
                        # Calculate holding period in terms of steps/bars
                        holding_time = current_timestamp - entry_info['entry_time']
                        holding_time_str = f"{holding_time} bars"
                        
                        holding_time_days = holding_time
                        holding_time_str = f"{holding_time_days} day(s)"
                        
                    else:
                                        
                        # Calculate holding period
                        if isinstance(current_timestamp, pd.Timestamp) and isinstance(entry_info['entry_time'], pd.Timestamp):
                            holding_time = current_timestamp - entry_info['entry_time']
                            holding_time_days = holding_time.days
                            holding_time_hours = holding_time.seconds // 3600
                            holding_time_str = f"{holding_time_days} days, {holding_time_hours} hours"
                        else:
                            holding_time = "N/A"
                            holding_time_str = "N/A"
                    
                    # Calculate profit/loss
                    profit_loss = (current_price - entry_info['entry_price']) * min(shares_traded, entry_info['shares'])
                    profit_loss_pct = (current_price - entry_info['entry_price']) / entry_info['entry_price'] if entry_info['entry_price'] else 0
                    
                    # Determine exit reason
                    exit_reason = "Regular Sell"
                    if env.stop_loss > 0 and current_price <= env.stop_loss:
                        exit_reason = "Stop Loss"
                    elif env.trailing_stop > 0 and current_price <= env.trailing_stop:
                        exit_reason = "Trailing Stop"
                    elif env.target_price > 0 and current_price >= env.target_price:
                        exit_reason = "Take Profit"
                    
                    # Record holding period
                    holding_periods.append({
                        'episode': episode,
                        'entry_time': entry_info['entry_time'],
                        'exit_time': current_timestamp,
                        'holding_period': holding_time,
                        'holding_period_str': holding_time_str,
                        'entry_price': entry_info['entry_price'],
                        'exit_price': current_price,
                        'shares': min(shares_traded, entry_info['shares']),
                        'profit_loss': profit_loss,
                        'profit_loss_pct': profit_loss_pct,
                        'initial_stop_loss': entry_info['stop_loss'],
                        'initial_target': entry_info['target_price'],
                        'final_stop_loss': env.stop_loss,
                        'final_trailing_stop': env.trailing_stop,
                        'exit_reason': exit_reason
                    })
                    
                    # Update or remove position
                    remaining_shares = entry_info['shares'] - shares_traded
                    if remaining_shares <= 0:
                        del active_positions[entry_timestamp]
                    else:
                        active_positions[entry_timestamp]['shares'] = remaining_shares
            
            # Render if requested
            if render:
                env.render()
        
        # Calculate performance metrics
        metrics = env.calculate_performance_metrics()
        all_metrics.append(metrics)
        logger.log(LOG_LEVELS['PERFORMANCE_METRICS'], f"Episode {episode+1} Metrics for {symbol}: {metrics}")
        
        # Save trade history in metrics
        metrics['detailed_trades'] = episode_trades
        metrics['capital_history'] = episode_capital
        metrics['drawdown_history'] = episode_drawdowns
        
        # Plot performance
        if save_results:
            # Create results directory if it doesn't exist
            os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
            
            # Plot and save
            fig_path = f"{RESULTS_SAVE_PATH}/{symbol}_ep{episode}.png"
            fig = env.plot_performance(save_path=fig_path)
            figures.append(fig)
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Figure saved to {fig_path} for {symbol}, Episode {episode+1}")
            
            # Save metrics to JSON (excluding numpy arrays and handling timestamp serialization)
            metrics_path = f"{RESULTS_SAVE_PATH}/{symbol}_ep{episode}_metrics.json"
            with open(metrics_path, 'w') as f:
                sanitized_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, np.ndarray):
                        continue
                    elif isinstance(v, dict):
                        sanitized_metrics[k] = {str(kk): str(vv) if isinstance(vv, (pd.Timestamp, np.ndarray)) else vv 
                                               for kk, vv in v.items()}
                    else:
                        sanitized_metrics[k] = str(v) if isinstance(v, (pd.Timestamp, np.ndarray)) else v
                json.dump(sanitized_metrics, f, indent=4)
            
            # Save detailed trade history to CSV
            trades_df = pd.DataFrame(episode_trades)
            if not trades_df.empty:
                trades_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_ep{episode}_trades.csv"
                trades_df.to_csv(trades_csv_path, index=False)
                logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Trade history saved to {trades_csv_path} for {symbol}, Episode {episode+1}")
            
            # Save capital history to CSV
            capital_df = pd.DataFrame(episode_capital)
            capital_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_ep{episode}_capital.csv"
            capital_df.to_csv(capital_csv_path, index=False)
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Capital history saved to {capital_csv_path} for {symbol}, Episode {episode+1}")
            
            # Save drawdown history to CSV
            drawdown_df = pd.DataFrame(episode_drawdowns)
            drawdown_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_ep{episode}_drawdowns.csv"
            drawdown_df.to_csv(drawdown_csv_path, index=False)
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Drawdown history saved to {drawdown_csv_path} for {symbol}, Episode {episode+1}")
            
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Metrics saved to {metrics_path} for {symbol}, Episode {episode+1}")

    # Calculate average metrics across episodes
    avg_metrics = {
        'symbol': symbol,
        'avg_total_return': np.mean([m['total_return'] for m in all_metrics]),
        'avg_annual_return': np.mean([m['annual_return'] for m in all_metrics]),
        'avg_sharpe_ratio': np.mean([m['sharpe_ratio'] for m in all_metrics]),
        'avg_max_drawdown': np.mean([m['max_drawdown'] for m in all_metrics]),
        'avg_win_rate': np.mean([m['win_rate'] for m in all_metrics]),
        'avg_profit_factor': np.mean([m['profit_factor'] for m in all_metrics if m['profit_factor'] != float('inf')]),
        'avg_calmar_ratio': np.mean([m['calmar_ratio'] for m in all_metrics if m['calmar_ratio'] != float('inf')]),
        'net_worth': np.mean([m['net_worth'] for m in all_metrics])
    }

    logger.log(LOG_LEVELS['PERFORMANCE_METRICS'], f"Average Metrics for {symbol}: {avg_metrics}")

    if save_results:
        avg_metrics_path = f"{RESULTS_SAVE_PATH}/{symbol}_avg_metrics.json"
        with open(avg_metrics_path, 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Average Metrics saved to {avg_metrics_path} for {symbol}")
        
        # Save all holding periods to CSV
        if holding_periods:
            holding_df = pd.DataFrame(holding_periods)
            holding_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_holding_periods.csv"
            holding_df.to_csv(holding_csv_path, index=False)
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Holding periods saved to {holding_csv_path} for {symbol}")
        
        # Save all trades to CSV
        all_trades_df = pd.DataFrame(detailed_trades)
        if not all_trades_df.empty:
            all_trades_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_all_trades.csv"
            all_trades_df.to_csv(all_trades_csv_path, index=False)
            logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"All trades saved to {all_trades_csv_path} for {symbol}")
        
        # Save all capital history to CSV
        all_capital_df = pd.DataFrame(capital_history)
        all_capital_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_all_capital.csv"
        all_capital_df.to_csv(all_capital_csv_path, index=False)
        logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"All capital history saved to {all_capital_csv_path} for {symbol}")
        
        # Save all drawdown history to CSV
        all_drawdown_df = pd.DataFrame(drawdown_history)
        all_drawdown_csv_path = f"{RESULTS_SAVE_PATH}/{symbol}_all_drawdowns.csv"
        all_drawdown_df.to_csv(all_drawdown_csv_path, index=False)
        logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"All drawdown history saved to {all_drawdown_csv_path} for {symbol}")

    # Add detailed trading information to the return values
    detailed_info = {
        'trades': detailed_trades,
        'holding_periods': holding_periods,
        'capital_history': capital_history,
        'drawdown_history': drawdown_history
    }
    
    return all_metrics, avg_metrics, figures, detailed_info

def backtest_buy_and_hold(df, symbol, **env_kwargs):
    """Run a simple buy and hold backtest for comparison"""
    try:
        # Create environment
        env = IndianStockTradingEnv(df=df, symbol=symbol, **env_kwargs)

        start_price = df.iloc[0]['Close']
        end_price = df.iloc[-1]['Close']
        
        # Calculate shares to buy at the start (no transaction fee adjustment)
        shares = int(env_kwargs['initial_balance'] // start_price)
        
        # Calculate trade cost using proper transaction cost calculation
        trade_value = shares * start_price
        txn_cost = env.calculate_transaction_costs(trade_value, is_intraday=False)  # Assuming delivery
        
        cost = trade_value + txn_cost  # Total cost including transaction charges
        
        # Calculate final value (after selling)
        sell_trade_value = shares * end_price
        sell_txn_cost = env.calculate_transaction_costs(sell_trade_value, is_intraday=False)  # Selling costs
        
        final_value = sell_trade_value - sell_txn_cost
        
        # Calculate return
        profit = final_value - cost
        total_return = profit / env_kwargs['initial_balance']
        
        # Annualize return (assuming 252 trading days)
        days = len(df)
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        return {
            'symbol': symbol,
            'strategy': 'buy_and_hold',
            'shares_held': shares,
            'initial_cost': cost,
            'final_value': final_value,
            'profit': profit,
            'total_return': total_return,
            'annual_return': annual_return
        }
    
    except Exception as e:
        logger.log(LOG_LEVELS['EXCEPTION'], f"Error in Buy and Hold Backtest for {symbol}: {str(e)}")
        return None

def optimize_hyperparameters(trial, train_df, symbol, **env_kwargs):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)  # Wider range
    n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192])  # Add smaller step option
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
    gamma = trial.suggest_uniform('gamma', 0.9, 0.99)
    n_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.001, 0.1)  # Log-uniform for entropy
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    policy_kwargs = dict(
        net_arch=[dict(pi=[128, 64, 32], vf=[128, 64, 32])]
    )

    # Create a PPO model with the suggested hyperparameters
    model = PPO(
        'MlpPolicy',
        create_vec_env(train_df, symbol, **env_kwargs),
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
        # tensorboard_log=f"./tensorboard/{symbol}/",
        device=device
    )

    # Train the model and evaluate its performance
    model.learn(total_timesteps=100_000)
    eval_env = IndianStockTradingEnv(df=train_df, symbol=symbol, **env_kwargs)
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)

    # Return the negative mean reward (Optuna minimizes the objective function)
    return -mean_reward

def main():
    """Main execution function"""
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(RESULTS_SAVE_PATH, exist_ok=True)
    
    # Run for each symbol
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        logger.log(LOG_LEVELS['DATA_LOAD'], f"Processing {symbol}...")

        # Load and prepare data
        train_df, test_df = update_load_and_prepare_data(symbol)
        
        # Environment parameters
        env_kwargs = {
            'initial_balance': DEFAULT_INITIAL_BALANCE,
            'window_size': 20,
            'use_trailing_stop': True,
            'atr_periods': 14,
            'trailing_stop_atr_multiplier': 2.0,
            'target_atr_multiplier': 3.0,
            'min_rr_ratio': 1.5
        }

        # Train model
        model_path = f"{MODEL_SAVE_PATH}/{symbol}.zip"
        
        # Check if model exists
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            logger.log(LOG_LEVELS['MODEL_TRAINING'], f"Loading existing model from {model_path}")
            model = PPO.load(model_path)
        else:
            print(f"Training new model for {symbol}")
            logger.log(LOG_LEVELS['MODEL_TRAINING'], f"Training new model for {symbol}")

             # Use more efficient sampler and pruner
            sampler = TPESampler(n_startup_trials=10)
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=5, 
                max_resource=15, 
                reduction_factor=3
            )
            
            # Setup common database
            db_folder = "optuna_db"
            os.makedirs(db_folder, exist_ok=True)

            # Dynamic study name based on symbol
            study_name = f"optuna_{symbol}"

            # Path to database file
            storage_url = f"sqlite:///{db_folder}/{study_name}.db"

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction='minimize', 
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )

            print(f"Using study: {study_name}")
            print(f"Database stored at: {storage_url}")

            n_trials = 500
            n_jobs = 1  # Number of processes to run in parallel
            # n_jobs = max(1, os.cpu_count() - 3)  # Use all but one CPU core
            print(f"Running {n_trials} trials with {n_jobs} parallel jobs")
            
            completed_trials = len(study.trials)
            remaining_trials = max(n_trials - completed_trials, 0)
            with tqdm(
                total=n_trials,
                desc="Hyperparameter Tuning",
                unit="trials",
                initial=completed_trials
            ) as pbar:
                study.optimize(
                    lambda trial: optimize_hyperparameters(trial, train_df, symbol, **env_kwargs),
                    n_trials=remaining_trials,
                    n_jobs=n_jobs,
                    callbacks=[lambda study, trial: pbar.update(1)]
                )

            # Get the best hyperparameters and the corresponding mean reward
            best_trial = study.best_trial
            best_hyperparameters = best_trial.params
            best_mean_reward = -best_trial.value

            print(f"Best Hyperparameters for {symbol}: {best_hyperparameters}")
            print(f"Best Mean Reward for {symbol}: {best_mean_reward:.2f}")

            # Save to a JSON file
            with open(f"{symbol}_best_hyperparameters.json", "w") as f:
                json.dump(best_hyperparameters, f, indent=4)

            print(f"✅ Best hyperparameters saved to {symbol}_best_hyperparameters.json")

            # Now you can use best_hyperparameters in PPO()
            with open(f"{symbol}_best_hyperparameters.json", "r") as f:
                best_hyperparameters = json.load(f)

            # Train a new model with the best hyperparameters and evaluate its performance
            best_model = PPO(
                'MlpPolicy',
                create_vec_env(train_df, symbol, **env_kwargs),
                verbose=0,
                **best_hyperparameters,
                # tensorboard_log=f"./tensorboard/{symbol}/",
                device=device
            )

            total_timesteps = 500_000
            log_interval = 1000
            batch_size = 2048

            # Replace Rich Live with simple progress tracking
            print("Starting model training...")
            for i in range(0, total_timesteps, 2048):
                # Train model
                start_time = time.time()
                best_model.learn(total_timesteps=min(2048, total_timesteps - i), reset_num_timesteps=False, progress_bar=True, log_interval=log_interval)
                end_time = time.time()

                # Evaluate model to get reward
                mean_reward, _ = evaluate_policy(best_model, create_vec_env(train_df, symbol, **env_kwargs), n_eval_episodes=5, warn=False)

                # Print progress instead of using Rich Live
                print(f"Step: {i+batch_size}, Reward: {mean_reward:.2f}, Time: {end_time - start_time:.2f} sec")

                # --- Save model
                best_model.save(model_path)
                    
            # After the hyperparameter optimization
            best_model.save(model_path)
            logger.log(LOG_LEVELS['MODEL_TRAINING'], f"Model saved to {model_path} for {symbol}.")
            model = best_model  # Use this model for evaluation
        
        # Evaluate model
        print(f"Evaluating model for {symbol}")
        logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Evaluating model for {symbol}")

        metrics, avg_metrics, figures, detailed_info = evaluate_model(
            model=model,
            test_df=test_df,
            symbol=symbol,
            num_episodes=5,
            render=False,
            save_results=True,
            **env_kwargs
        )

        # Run buy and hold backtest for comparison
        bh_metrics = backtest_buy_and_hold(
            df=test_df,
            symbol=symbol,
            **env_kwargs
        )
        
        if bh_metrics is not None:
                # Save buy and hold metrics
                bh_metrics_path = f"{RESULTS_SAVE_PATH}/{symbol}_buy_and_hold_metrics.json"
                with open(bh_metrics_path, 'w') as f:
                    json.dump(bh_metrics, f, indent=4)
                logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Buy and Hold Metrics saved to {bh_metrics_path} for {symbol}")
        else:
            print("Error: Buy and Hold backtest failed. No metrics available.")
            
        # Print comparison
        logger.log(LOG_LEVELS['PERFORMANCE_METRICS'], "\nPerformance Comparison:")
        logger.log(LOG_LEVELS['PERFORMANCE_METRICS'], f"RL Strategy - Total Return: {avg_metrics['avg_total_return']:.2%}, Annual Return: {avg_metrics['avg_annual_return']:.2%}, Sharpe: {avg_metrics['avg_sharpe_ratio']:.2f}")

        if bh_metrics is not None:
            logger.log(LOG_LEVELS['PERFORMANCE_METRICS'], f"Buy & Hold  - Total Return: {bh_metrics['total_return']:.2%}, Annual Return: {bh_metrics['annual_return']:.2%}")
        
        logger.log(LOG_LEVELS['DATA_LOAD'], f"Completed processing {symbol}\n")
        
        # Print comparison
        print("\nPerformance Comparison:")
        print(f"RL Strategy -")
        print(f"  Total Return: {avg_metrics['avg_total_return']:.2%}")
        print(f"  Annual Return: {avg_metrics['avg_annual_return']:.2%}")
        print(f"  Sharpe Ratio: {avg_metrics['avg_sharpe_ratio']:.2f}")
        print(f"  Final Capital (₹): {avg_metrics['net_worth']:.2f}")
        print(f"  Capital Change (₹): {avg_metrics['net_worth'] - DEFAULT_INITIAL_BALANCE:.2f} ({'Gain' if avg_metrics['net_worth'] > DEFAULT_INITIAL_BALANCE else 'Loss'})")

        print(f"Buy & Hold  -")
        print(f"  Total Return: {bh_metrics['total_return']:.2%}")
        print(f"  Annual Return: {bh_metrics['annual_return']:.2%}")
        print(f"  Final Capital (₹): {bh_metrics['final_value']:.2f}")
        print(f"  Capital Change (₹): {bh_metrics['final_value'] - DEFAULT_INITIAL_BALANCE:.2f} ({'Gain' if bh_metrics['final_value'] > DEFAULT_INITIAL_BALANCE else 'Loss'})")

        table = [
            ["Metric", "RL Strategy", "Buy & Hold"],
            ["Total Return", f"{avg_metrics['avg_total_return']:.2%}", f"{bh_metrics['total_return']:.2%}"],
            ["Annual Return", f"{avg_metrics['avg_annual_return']:.2%}", f"{bh_metrics['annual_return']:.2%}"],
            ["Sharpe Ratio", f"{avg_metrics['avg_sharpe_ratio']:.2f}", "-"],  # No Sharpe Ratio for Buy & Hold
            ["Final Capital (₹)", f"{avg_metrics['net_worth']:.2f}", f"{bh_metrics['final_value']:.2f}"],
            ["Capital Change (₹)", 
            f"{avg_metrics['net_worth'] - DEFAULT_INITIAL_BALANCE:.2f} ({'Gain' if avg_metrics['net_worth'] > DEFAULT_INITIAL_BALANCE else 'Loss'})", 
            f"{bh_metrics['final_value'] - DEFAULT_INITIAL_BALANCE:.2f} ({'Gain' if bh_metrics['final_value'] > DEFAULT_INITIAL_BALANCE else 'Loss'})"]
        ]

        print(tabulate(table, headers="firstrow", tablefmt="grid"))

        # Print detailed trade information
        if detailed_info['trades']:
            print("\nDetailed Trade Information:")
            trade_table = []
            trade_headers = ["Episode", "Timestamp", "Action", "Price (₹)", "Shares", "Value (₹)", "Capital After (₹)", "Stop Loss (₹)", "Target (₹)"]
            
            for trade in detailed_info['trades']:
                trade_row = [
                    trade['episode'],
                    trade['timestamp'],
                    trade['action'],
                    f"{trade['price']:.2f}" if trade['price'] else "N/A",
                    trade['shares'],
                    f"{trade['value']:.2f}" if 'value' in trade else "N/A",
                    f"{trade['capital_after']:.2f}" if 'capital_after' in trade else "N/A",
                    f"{trade['stop_loss']:.2f}" if trade.get('stop_loss') and trade['stop_loss'] > 0 else "-",
                    f"{trade['target_price']:.2f}" if trade.get('target_price') and trade['target_price'] > 0 else "-"
                ]
                trade_table.append(trade_row)
            
            print(tabulate(trade_table, headers=trade_headers, tablefmt="grid"))

        # Print holding periods
        if detailed_info['holding_periods']:
            print("\nHolding Periods:")
            holding_table = []
            holding_headers = ["Episode", "Entry Time", "Exit Time", "Holding Period", "Entry Price (₹)","Exit Price (₹)", "Shares", "P/L (₹)", "P/L %", "Exit Reason"]
            
            for period in detailed_info['holding_periods']:
                holding_table.append([
                    period['episode'],
                    period['entry_time'],
                    period['exit_time'],
                    period.get('holding_period_str', period['holding_period']),
                    f"{period['entry_price']:.2f}" if period['entry_price'] else "N/A",
                    f"{period['exit_price']:.2f}" if period['exit_price'] else "N/A",
                    period['shares'],
                    f"{period['profit_loss']:.2f}" if 'profit_loss' in period else "N/A",
                    f"{period['profit_loss_pct']:.2%}" if 'profit_loss_pct' in period else "N/A",
                    period.get('exit_reason', 'Regular Sell')
                ])
            
            print(tabulate(holding_table, headers=holding_headers, tablefmt="grid"))

        # Calculate and print drawdown statistics
        if detailed_info['drawdown_history']:
            drawdown_df = pd.DataFrame(detailed_info['drawdown_history'])
            max_drawdown = drawdown_df['drawdown'].max()
            
            print("\nDrawdown Analysis:")
            drawdown_stats = [
                ["Maximum Drawdown", f"{max_drawdown:.2%}"],
                ["Average Drawdown", f"{drawdown_df['drawdown'].mean():.2%}"],
                ["Median Drawdown", f"{drawdown_df['drawdown'].median():.2%}"],
                ["Drawdown Volatility", f"{drawdown_df['drawdown'].std():.2%}"],
                ["Time in Drawdown", f"{len(drawdown_df[drawdown_df['drawdown'] > 0]) / len(drawdown_df):.2%}" if len(drawdown_df) > 0 else "0%"]
            ]
            
            print(tabulate(drawdown_stats, headers=["Metric", "Value"], tablefmt="grid"))

        # Print trade statistics
        if detailed_info['trades']:
            trades_df = pd.DataFrame(detailed_info['trades'])
            
            # Filter buy and sell trades
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Calculate win/loss metrics
            if not detailed_info['holding_periods']:
                win_rate = "N/A"
                avg_win = "N/A"
                avg_loss = "N/A"
            else:
                holding_df = pd.DataFrame(detailed_info['holding_periods'])
                winning_trades = holding_df[holding_df['profit_loss'] > 0]
                losing_trades = holding_df[holding_df['profit_loss'] <= 0]
                
                win_rate = f"{len(winning_trades) / len(holding_df):.2%}" if len(holding_df) > 0 else "N/A"
                avg_win = f"₹{winning_trades['profit_loss'].mean():.2f}" if not winning_trades.empty else "N/A"
                avg_loss = f"₹{losing_trades['profit_loss'].mean():.2f}" if not losing_trades.empty else "N/A"
            
            print("\nTrade Statistics:")
            trade_stats = [
                ["Total Trades", len(detailed_info['trades'])],
                ["Buy Trades", len(buy_trades) if not buy_trades.empty else 0],
                ["Sell Trades", len(sell_trades) if not sell_trades.empty else 0],
                ["Win Rate", win_rate],
                ["Average Win", avg_win],
                ["Average Loss", avg_loss],
                ["Avg Buy Price", f"₹{buy_trades['price'].mean():.2f}" if not buy_trades.empty and 'price' in buy_trades else "N/A"],
                ["Avg Sell Price", f"₹{sell_trades['price'].mean():.2f}" if not sell_trades.empty and 'price' in sell_trades else "N/A"],
                ["Avg Trade Size (Shares)", f"{trades_df['shares'].mean():.2f}" if 'shares' in trades_df and not trades_df.empty else "N/A"],
                ["Avg Trade Value", f"₹{trades_df['value'].mean():.2f}" if 'value' in trades_df and not trades_df.empty else "N/A"]
            ]
            
            print(tabulate(trade_stats, headers=["Metric", "Value"], tablefmt="grid"))

        print(f"Completed processing {symbol}\n")

# Run main function if script is executed directly
if __name__ == "__main__":
    main()