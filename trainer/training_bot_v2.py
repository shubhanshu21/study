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
import sqlite3

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

# --- Setup SQLite DB ---
conn = sqlite3.connect('training_progress.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS model_training_progress (
        symbol TEXT PRIMARY KEY,
        last_trained_steps INTEGER
    )
''')
conn.commit()

SYMBOLS = ["ADANIPORTS","ASIANPAINT","BAJFINANCE"]

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

# Optimize risk parameters for better performance
MAX_POSITION_SIZE_PCT = 0.20  # Significantly increased from 0.30
MAX_OPEN_POSITIONS = 8        # Further increased from 6
MAX_TRADES_PER_DAY = 3       # Further increased from 8
DAILY_LOSS_LIMIT_PCT = 0.025  # Slightly increased from 0.03
DAILY_PROFIT_TARGET_PCT = 0.05 # Further increased from 0.08
MAX_DRAWDOWN_ALLOWED = 0.15   # Slightly increased from 0.12

# Enhanced trading modes
SWING_TRADING_MODE = True     # Enable swing trading mode for longer holds

# Enhanced technical parameters
SWING_TRADING_MODE = True     # Enable swing trading mode (hold positions longer)
USE_MARKET_REGIME = True      # Adapt strategy to market regime (trending vs ranging)
USE_BOLLINGER_BREAKOUTS = True  # Use Bollinger band breakouts for entries
USE_RSI_FILTER = True         # Use RSI filter for entries
USE_VOLUME_FILTER = True      # Use volume confirmation for entries
USE_PROFIT_TRAILING = True    # Use trailing stops for profits
USE_MULTIPLE_TIMEFRAMES = True  # Use multiple timeframe analysis

# Market timing settings - avoid trading in high volatility periods
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 15
MARKET_CLOSE_HOUR = 15
MARKET_CLOSE_MINUTE = 30
AVOID_FIRST_MINUTES = 60  # Avoid first 15 minutes of trading (high volatility)
AVOID_LAST_MINUTES = 60   # Avoid last 15 minutes of trading (high volatility)

TIMEZONE = pytz.timezone('Asia/Kolkata')

# Logging levels for specific events
LOG_LEVELS = {
    'DATA_LOAD': logging.INFO,
    'MODEL_TRAINING': logging.INFO,
    'MODEL_EVALUATION': logging.INFO,
    'PERFORMANCE_METRICS': logging.INFO,
    'EXCEPTION': logging.ERROR,
    'RISK_MANAGEMENT': logging.WARNING
}

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
                "trailing_stop_atr_multiplier": 3.0,  # Increased from 2.0
                "target_atr_multiplier": 5.0,  # Slightly decreased 
                "position_size_pct": 0.20,  # Reduced from 0.35
                "use_trailing_stop": True,
                "min_rr_ratio": 2.0  # Increased from 1.5
            },
            "trending_down": {
                "trailing_stop_atr_multiplier": 2.5,  # Increased from 1.8
                "target_atr_multiplier": 4.0,  # Slightly decreased
                "position_size_pct": 0.15,  # Reduced from 0.20
                "use_trailing_stop": True,
                "min_rr_ratio": 2.5  # Increased from 1.8
            },
            "ranging": {
                "trailing_stop_atr_multiplier": 2.0,  # Increased from 1.0
                "target_atr_multiplier": 3.0,  # Same
                "position_size_pct": 0.15,  # Reduced from 0.25
                "use_trailing_stop": True,
                "min_rr_ratio": 2.0  # Increased from 1.5
            },
            "volatile": {
                "trailing_stop_atr_multiplier": 3.0,  # Increased from 2.5
                "target_atr_multiplier": 5.0,  # Decreased from 7.0
                "position_size_pct": 0.10,  # Reduced from 0.18
                "use_trailing_stop": True,
                "min_rr_ratio": 2.5  # Increased from 1.8
            },
            "unknown": {
                "trailing_stop_atr_multiplier": 2.5,  # Increased from 1.8
                "target_atr_multiplier": 4.0,  # Same
                "position_size_pct": 0.15,  # Reduced from 0.20
                "use_trailing_stop": True,
                "min_rr_ratio": 2.0  # Increased from 1.5
            }
        }
        
        return params.get(regime, params["unknown"])

# Trading Environment
class IndianStockTradingEnv(gym.Env):
    """Custom Environment for trading Indian stocks using RL with advanced features"""
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, symbol, initial_balance=DEFAULT_INITIAL_BALANCE, window_size=20,
             use_trailing_stop=True, atr_periods=14, trailing_stop_atr_multiplier=2,
             target_atr_multiplier=3, min_rr_ratio=1.5, max_position_size_pct=MAX_POSITION_SIZE_PCT,
             max_trades_per_day=MAX_TRADES_PER_DAY, daily_loss_limit_pct=DAILY_LOSS_LIMIT_PCT):
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
        
        # Risk management parameters
        self.max_position_size_pct = max_position_size_pct
        self.max_trades_per_day = max_trades_per_day
        self.daily_loss_limit_pct = daily_loss_limit_pct
        
        # Market regime detector
        self.regime_detector = MarketRegimeDetector(lookback=20)
        
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
        self.initial_stop_loss = 0  # Track initial stop for risk calculation
        self.target_price = 0
        self.highest_price_since_buy = 0
        self.position_type = None  # 'long' or 'short'
        self.last_txn_cost = 0
        self.entry_date = None
        self.entry_time = None
        
        # Enhanced risk management features
        self.daily_trades = 0
        self.daily_high_capital = initial_balance
        self.daily_low_capital = initial_balance
        self.trading_stopped_for_day = False
        self.trading_stopped_reason = None
        self.drawdown_reduction_applied = False
        self.position_size_pct = max_position_size_pct
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
        self.positions = {}  # Dictionary to track multiple positions
        self.open_positions_count = 0

        # Trading signals
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Episode variables
        self.current_step = 0
        self.rewards = []
        self.trades = []
        self.net_worth_history = [initial_balance]
        
        # Define action and observation spaces
        # Actions: 0 = Hold, 1 = Buy with 25% capital, 2 = Buy with 50% capital, 
        # 3 = Buy with 75% capital, 4 = Buy with 100% capital, 5 = Sell all
        self.action_space = spaces.Discrete(6)
        
        # Observation space: balance, shares, cost_basis, current_price, technical indicators, etc.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32
        )

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
        
        # Total transaction cost - print for debugging
        total_cost = brokerage + stt + exchange_charge + sebi_charges + stamp_duty + gst
        
        # Safety check - cap transaction costs at a reasonable percentage
        if total_cost > trade_value * 0.1:  # Cap at 10% of trade value
            print(f"WARNING: Transaction costs capped from {total_cost} to {trade_value * 0.1}")
            total_cost = trade_value * 0.1
            
        return total_cost
    
    def calculate_post_tax_profit(self, buy_price, sell_price, quantity, holding_period_days):
        """Calculate profit after accounting for transaction costs and capital gains tax"""
        # Pre-tax profit 
        gross_profit = (sell_price - buy_price) * quantity
        
        # Transaction costs (both buy and sell)
        buy_cost = self.calculate_transaction_costs(buy_price * quantity)
        sell_cost = self.calculate_transaction_costs(sell_price * quantity)
        
        # Pre-tax net profit
        pre_tax_profit = gross_profit - buy_cost - sell_cost
        
        # Only apply capital gains tax on profits
        tax_rate = LONG_TERM_CAPITAL_GAINS_TAX if holding_period_days >= LONG_TERM_THRESHOLD_DAYS else SHORT_TERM_CAPITAL_GAINS_TAX
        
        capital_gains_tax = max(0, pre_tax_profit * tax_rate) if pre_tax_profit > 0 else 0
        
        # Final profit after all costs and taxes
        post_tax_profit = pre_tax_profit - capital_gains_tax
        
        return post_tax_profit
    
    def _calculate_atr(self):
        """Calculate Average True Range using Finta"""
        # Using Finta's ATR function
        if 'ATR' not in self.df.columns:
            self.df['ATR'] = TA.ATR(self.df, period=self.atr_periods)
            
        # Fill NaN values with a reasonable default
        self.df.loc[:, 'ATR'].bfill(inplace=True)  # Updated for bfill
        self.df.fillna({'ATR': self.df['Close'] * 0.02}, inplace=True)
        # logger.log(LOG_LEVELS['DATA_LOAD'], f"ATR calculation complete for {self.symbol}.")


    def _calculate_dynamic_stop_loss(self, price, position_type, regime_params=None):
        """Calculate dynamic stop loss based on ATR and market regime"""
        current_atr = self.df.iloc[self.current_step]['ATR']
        
        # Use regime-specific multiplier if provided
        atr_multiplier = regime_params['trailing_stop_atr_multiplier'] if regime_params else self.trailing_stop_atr_multiplier
        
        # Adjust for consecutive losses (tighten stops)
        if self.consecutive_losses >= 3:
            atr_multiplier *= 0.8  # Reduce by 20% after 3 consecutive losses
        
        if position_type == 'long':
            stop_loss = price - (current_atr * atr_multiplier)
            
            # Enhanced: Add support level consideration
            if 'Support' in self.df.columns:
                support_level = self.df.iloc[self.current_step]['Support']
                if support_level < price and support_level > stop_loss:
                    # Place stop just below support
                    stop_loss = support_level * 0.995
        else:  # short position
            stop_loss = price + (current_atr * atr_multiplier)
            
            # Enhanced: Add resistance level consideration
            if 'Resistance' in self.df.columns:
                resistance_level = self.df.iloc[self.current_step]['Resistance']
                if resistance_level > price and resistance_level < stop_loss:
                    # Place stop just above resistance
                    stop_loss = resistance_level * 1.005

        return stop_loss
    
    def _calculate_dynamic_target(self, entry_price, position_type, regime_params):
        """Calculate dynamic target based on ATR and min risk-reward ratio"""
        current_atr = self.df.iloc[self.current_step]['ATR']
        stop_distance = current_atr * self.trailing_stop_atr_multiplier
        
        # Ensure risk-reward ratio is at least min_rr_ratio
        target_distance = stop_distance * self.min_rr_ratio
        # Also consider the target_atr_multiplier
        target_distance = max(target_distance, current_atr * self.target_atr_multiplier)
        
        if position_type == 'long':
            target = entry_price + target_distance
        else:  # short position
            target = entry_price - target_distance

        # logger.log(LOG_LEVELS['DATA_LOAD'], f"Dynamic target calculation complete. {target}.")        
        return target
    
    def _update_trailing_stop(self, current_price, regime_params=None):
        """Update trailing stop if price moves in favorable direction"""
        if not self.position_type:
            return
            
        # Use regime-specific trailing settings
        use_trailing = regime_params['use_trailing_stop'] if regime_params else self.use_trailing_stop
        
        if not use_trailing:
            return
            
        if self.position_type == 'long':
            if current_price > self.highest_price_since_buy:
                self.highest_price_since_buy = current_price
                
                # Enhanced trailing stop logic
                profit_pct = (current_price - self.cost_basis) / self.cost_basis
                
                # Tighten trail as profit increases
                trail_multiplier = self.trailing_stop_atr_multiplier
                if regime_params:
                    trail_multiplier = regime_params['trailing_stop_atr_multiplier']
                
                # Tighten trail as profit grows
                if profit_pct > 0.10:  # Over 10% profit
                    trail_multiplier *= 0.7  # Tighter trail to protect profits
                elif profit_pct > 0.05:  # Over 5% profit
                    trail_multiplier *= 0.8
                
                # Update trailing stop
                new_stop = self._calculate_dynamic_stop_loss(current_price, 'long', 
                                                        {'trailing_stop_atr_multiplier': trail_multiplier})
                
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    
                    # Move to breakeven after sufficient profit
                    if profit_pct > 0.03 and self.trailing_stop < self.cost_basis:
                        self.trailing_stop = max(self.trailing_stop, self.cost_basis * 1.001)  # 0.1% above entry
                    
        else:  # short position
            if current_price < self.highest_price_since_buy:  # For shorts, this would be lowest price
                self.highest_price_since_buy = current_price
                
                # Enhanced trailing stop logic for shorts
                profit_pct = (self.cost_basis - current_price) / self.cost_basis
                
                # Tighten trail as profit increases
                trail_multiplier = self.trailing_stop_atr_multiplier
                if regime_params:
                    trail_multiplier = regime_params['trailing_stop_atr_multiplier']
                
                # Tighten trail as profit grows
                if profit_pct > 0.10:  # Over 10% profit
                    trail_multiplier *= 0.7  # Tighter trail to protect profits
                elif profit_pct > 0.05:  # Over 5% profit
                    trail_multiplier *= 0.8
                
                # Update trailing stop
                new_stop = self._calculate_dynamic_stop_loss(current_price, 'short', 
                                                        {'trailing_stop_atr_multiplier': trail_multiplier})
                
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    
                    # Move to breakeven after sufficient profit
                    if profit_pct > 0.03 and self.trailing_stop > self.cost_basis:
                        self.trailing_stop = min(self.trailing_stop, self.cost_basis * 0.999)  # 0.1% below entry
    

    def _check_stop_and_target(self, current_price):
        """Check if price hit stop loss or target with improved exit criteria"""
        if not self.position_type:
            return False, None
            
        hit_stop = False
        hit_target = False
        
        # Get time of day for time-based exits
        has_date_time = 'Date' in self.df.columns and 'Time' in self.df.columns
        is_near_close = False
        
        if has_date_time:
            current_date = self.df.iloc[self.current_step]['Date']
            current_time = self.df.iloc[self.current_step]['Time']
            
            # Check if near market close
            if hasattr(current_time, 'hour') and hasattr(current_time, 'minute'):
                is_near_close = (current_time.hour == MARKET_CLOSE_HOUR and 
                            current_time.minute >= (MARKET_CLOSE_MINUTE - AVOID_LAST_MINUTES))
        
        # Force exit if near market close
        if is_near_close and self.position_type:
            return True, "market_close"
            
        # Normal stop and target checks
        if self.position_type == 'long':
            if current_price <= self.trailing_stop:
                hit_stop = True
                exit_type = 'stop_loss'
            elif current_price >= self.target_price:
                hit_target = True
                exit_type = 'target'
                
            # Calculate current profit percentage
            profit_pct = (current_price - self.cost_basis) / self.cost_basis
            
            # IMPROVED: Be more patient with exits in trending markets, faster in ranging
            if self.current_market_regime == "volatile" and profit_pct > 0.05:  # Increased threshold
                # Take profits in volatile markets if decent gain
                hit_target = True
                exit_type = 'regime_volatile_take_profit'
                
            elif self.current_market_regime == "ranging" and profit_pct > 0.04:  # Slightly reduced threshold
                # Take profits sooner in ranging markets
                hit_target = True
                exit_type = 'regime_ranging_take_profit'
                
            # For trending markets, hold for bigger moves
            elif self.current_market_regime == "trending_up":
                # Only exit if momentum really slows and we have significant profit
                if profit_pct > 0.10 and 'EMA5' in self.df.columns and current_price < self.df.iloc[self.current_step]['EMA5']:
                    hit_target = True
                    exit_type = 'trend_momentum_loss'
                    
            # IMPROVED: Exit faster on losing trades (cut losses)
            if profit_pct < -0.03 and self.days_in_trade > 2:
                hit_stop = True
                exit_type = 'time_based_exit_loss'
                    
            # IMPROVED: Time-based exit for swing trading - allow longer holds for profitable trades
            if self.days_in_trade > 12:
                if profit_pct < 0.01:  # Exit if minimal profit after 12 days
                    hit_stop = True
                    exit_type = 'time_based_exit'
                elif profit_pct > 0.15:  # Take profits if large gain after holding period
                    hit_target = True
                    exit_type = 'time_based_profit_target'
                    
            # IMPROVED: Exit if RSI extremely overbought (potential reversal)
            if 'RSI' in self.df.columns and self.df.iloc[self.current_step]['RSI'] > 80 and profit_pct > 0.05:
                hit_target = True
                exit_type = 'rsi_extreme'
                    
        else:  # short position
            if current_price >= self.trailing_stop:
                hit_stop = True
                exit_type = 'stop_loss'
            elif current_price <= self.target_price:
                hit_target = True
                exit_type = 'target'
                
            # Calculate current profit percentage
            profit_pct = (self.cost_basis - current_price) / self.cost_basis
            
            # IMPROVED: Be more patient with exits in trending markets, faster in ranging
            if self.current_market_regime == "volatile" and profit_pct > 0.05:  # Increased threshold
                # Take higher profits in volatile markets
                hit_target = True
                exit_type = 'regime_volatile_take_profit'
                
            elif self.current_market_regime == "ranging" and profit_pct > 0.04:  # Slightly reduced
                # Take profits in ranging markets
                hit_target = True
                exit_type = 'regime_ranging_take_profit'
                
            # Protect profits in trending markets if momentum slows
            elif self.current_market_regime == "trending_down":
                # Check if momentum is slowing (price above 5-day EMA)
                if profit_pct > 0.10 and 'EMA5' in self.df.columns and current_price > self.df.iloc[self.current_step]['EMA5']:
                    hit_target = True
                    exit_type = 'trend_momentum_loss'
                        
            # IMPROVED: Exit faster on losing trades
            if profit_pct < -0.03 and self.days_in_trade > 2:
                hit_stop = True
                exit_type = 'time_based_exit_loss'
                    
            # IMPROVED: Time-based exit for swing trading
            if self.days_in_trade > 12:
                if profit_pct < 0.01:  # Exit if minimal profit after 12 days
                    hit_stop = True
                    exit_type = 'time_based_exit'
                elif profit_pct > 0.15:  # Take profits if large gain after holding period
                    hit_target = True
                    exit_type = 'time_based_profit_target'
                    
            # IMPROVED: Exit if RSI extremely oversold (potential reversal)
            if 'RSI' in self.df.columns and self.df.iloc[self.current_step]['RSI'] < 20 and profit_pct > 0.05:
                hit_target = True
                exit_type = 'rsi_extreme'
        
        # IMPROVED: Check for volume-based exits (major reversal signals)
        if self.position_type and 'Volume' in self.df.columns:
            avg_volume = self.df.iloc[self.current_step-10:self.current_step]['Volume'].mean()
            current_volume = self.df.iloc[self.current_step]['Volume']
            
            # Exit on extreme volume spike with adverse price movement
            if current_volume > avg_volume * 3.5:  # Very large volume spike (increased threshold)
                if (self.position_type == 'long' and current_price < self.df.iloc[self.current_step-1]['Close'] and profit_pct > 0.03) or \
                (self.position_type == 'short' and current_price > self.df.iloc[self.current_step-1]['Close'] and profit_pct > 0.03):
                    hit_target = True
                    exit_type = 'volume_reversal'
        
        # IMPROVED: Check for parabolic movement followed by reversal
        if self.position_type and profit_pct > 0.12:  # Big profit (>12%)
            # Check for potential reversal after strong move
            if (self.position_type == 'long' and 
                current_price < self.df.iloc[self.current_step-1]['Close'] and 
                self.df.iloc[self.current_step-1]['Close'] < self.df.iloc[self.current_step-2]['Close']):
                # Two consecutive down days after strong rally
                hit_target = True
                exit_type = 'parabolic_reversal'
                
            elif (self.position_type == 'short' and 
                current_price > self.df.iloc[self.current_step-1]['Close'] and 
                self.df.iloc[self.current_step-1]['Close'] > self.df.iloc[self.current_step-2]['Close']):
                # Two consecutive up days after strong decline
                hit_target = True
                exit_type = 'parabolic_reversal'
        
        # IMPROVED: Check for key resistance/support levels
        if 'BB_Upper' in self.df.columns and 'BB_Lower' in self.df.columns:
            upper_band = self.df.iloc[self.current_step]['BB_Upper']
            lower_band = self.df.iloc[self.current_step]['BB_Lower']
            
            # Long position at upper band resistance with profit
            if self.position_type == 'long' and current_price >= upper_band * 0.98 and profit_pct > 0.04:
                hit_target = True
                exit_type = 'bollinger_resistance'
                
            # Short position at lower band support with profit
            elif self.position_type == 'short' and current_price <= lower_band * 1.02 and profit_pct > 0.04:
                hit_target = True
                exit_type = 'bollinger_support'
        
        return hit_stop or hit_target, exit_type if (hit_stop or hit_target) else None
    

    def _detect_market_regime(self):
        """Detect current market regime and update trading parameters"""
        self.current_market_regime = self.regime_detector.detect_regime(self.df, self.current_step)
        
        # Get optimal parameters for current regime
        regime_params = self.regime_detector.get_regime_parameters(self.current_market_regime)
        
        # Update trading parameters based on market regime
        self.trailing_stop_atr_multiplier = regime_params['trailing_stop_atr_multiplier']
        self.target_atr_multiplier = regime_params['target_atr_multiplier']
        self.min_rr_ratio = regime_params['min_rr_ratio']
        self.position_size_pct = min(regime_params['position_size_pct'], self.max_position_size_pct)
        
        return regime_params

    def calculate_enhanced_reward(self, action, exit_type=None):
        """Calculate reward based on action and outcome with enhanced risk-adjusted metrics"""
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        
        # Update net worth based on current price
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate base reward as normalized net worth change
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change / self.initial_balance * 100  # Scale relative to initial balance
        
        # Add trading action rewards/penalties
        if action >= 1 and action <= 4:  # Buy actions
            # MAJOR CHANGE: Strongly penalize excessive trades
            if self.daily_trades >= self.max_trades_per_day - 1:
                reward -= 5.0  # Much stronger penalty for excessive trading
                
            # Higher threshold for entry - only enter on strong signals
            if self.buy_signal_strength < 4:  # Require stronger signals
                reward -= 3.0  # Strongly discourage weak entries
                
            # Track purchase information
            self.purchase_step = self.current_step
            self.purchase_price = current_price
            
            # Higher transaction cost penalty
            reward -= 0.3  # Increased from 0.05 to discourage frequent trading
            
        elif action == 5 and self.shares_held > 0:  # Sell
            # Calculate profit metrics
            if hasattr(self, 'purchase_step') and hasattr(self, 'purchase_price'):
                holding_period_days = self.days_in_trade
                quantity = self.shares_held
                
                # Calculate post-tax profit
                post_tax_profit = self.calculate_post_tax_profit(
                    self.purchase_price, 
                    current_price, 
                    quantity, 
                    holding_period_days
                )
                
                # MAJOR CHANGE: Reward longer-term profitable trades much more
                if post_tax_profit > 0:
                    # Scale reward by holding period to encourage longer holds
                    holding_multiplier = min(1 + (holding_period_days * 0.1), 3.0)  # Cap at 3x
                    reward = (post_tax_profit / self.initial_balance * 100) * holding_multiplier
                else:
                    reward = post_tax_profit / self.initial_balance * 100
                
                # Additional rewards/penalties based on exit type
                if exit_type == 'target':
                    reward += 2.0  # Reduced from 4.0 to be more balanced
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    self.daily_profit += post_tax_profit
                elif exit_type == 'stop_loss':
                    # Smaller penalty for disciplined risk management
                    reward -= 0.5  # Increased penalty slightly
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    self.daily_loss += abs(post_tax_profit)
        
        # MAJOR CHANGE: Penalize drawdowns more heavily
        if self.shares_held > 0:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            if current_drawdown > 0.05:  # Lower threshold for drawdown penalty
                reward -= current_drawdown * 5  # Increased from 3
        
        # Update max net worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        return reward

    def _update_risk_management(self):
        """Update risk management parameters"""
        # Update daily tracking
        current_date = None
        
        # Extract date if available
        if 'Date' in self.df.columns:
            current_date = self.df.iloc[self.current_step]['Date']
        else:
            # Use step as a proxy for day
            current_date = self.current_step // 375  # ~375 minutes in a trading day
            
        # Reset daily counters on new day
        if self.current_date != current_date:
            # Previous date
            self.previous_date = self.current_date
            
            # New date
            self.current_date = current_date
            self.daily_trades = 0
            self.daily_high_capital = self.net_worth
            self.daily_low_capital = self.net_worth
            self.trading_stopped_for_day = False
            self.trading_stopped_reason = None
            self.daily_profit = 0
            self.daily_loss = 0
            
            # Increment days in trade if position open
            if self.position_type:
                self.days_in_trade += 1
        
        # Update daily capital extremes
        self.daily_high_capital = max(self.daily_high_capital, self.net_worth)
        self.daily_low_capital = min(self.daily_low_capital, self.net_worth)
        
        # Daily drawdown calculation
        daily_drawdown = (self.daily_high_capital - self.net_worth) / self.daily_high_capital if self.daily_high_capital > 0 else 0
        
        # Check daily loss limit
        daily_pnl_pct = (self.net_worth - self.initial_balance) / self.initial_balance
        
        # Dynamic daily trade limit based on market regime
        if self.current_market_regime == "trending_up":
            self.max_trades_per_day = MAX_TRADES_PER_DAY + 3  # More trades in uptrends
        elif self.current_market_regime == "volatile":
            self.max_trades_per_day = MAX_TRADES_PER_DAY - 1  # Fewer trades in volatile markets
        else:
            self.max_trades_per_day = MAX_TRADES_PER_DAY  # Default
        
        # Stop trading if daily loss limit hit
        if daily_pnl_pct < -self.daily_loss_limit_pct and not self.trading_stopped_for_day:
            self.trading_stopped_for_day = True
            self.trading_stopped_reason = "daily_loss_limit"
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Trading stopped for day due to daily loss limit: {daily_pnl_pct:.2%}")
        
        # Don't stop trading on daily profit target in strong uptrends
        if self.current_market_regime == "trending_up" and daily_pnl_pct > DAILY_PROFIT_TARGET_PCT:
            # Keep trading in strong uptrends even after hitting profit target
            self.trading_stopped_for_day = False
            self.trading_stopped_reason = None
        # Stop trading if daily profit target hit in non-trending markets
        elif daily_pnl_pct > DAILY_PROFIT_TARGET_PCT and not self.trading_stopped_for_day and self.current_market_regime != "trending_up":
            self.trading_stopped_for_day = True
            self.trading_stopped_reason = "daily_profit_target"
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Trading stopped for day due to reaching profit target: {daily_pnl_pct:.2%}")
        
        # Dynamic position sizing based on drawdown
        if self.max_drawdown > MAX_DRAWDOWN_ALLOWED and not self.drawdown_reduction_applied:
            # Reduce position size after large drawdown
            self.position_size_pct = self.max_position_size_pct * 0.5
            self.drawdown_reduction_applied = True
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Position size reduced due to drawdown of {self.max_drawdown:.2%}")
        
        # Adaptive position sizing based on winning streak
        if self.consecutive_wins >= 3:
            # Increase position size after 3 consecutive wins
            self.position_size_pct = min(self.position_size_pct * 1.2, self.max_position_size_pct * 1.5)
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Position size increased to {self.position_size_pct:.2%} after {self.consecutive_wins} consecutive wins")
        
        # Reset position size faster after recovery
        if self.drawdown_reduction_applied and self.max_drawdown < MAX_DRAWDOWN_ALLOWED/3:
            self.position_size_pct = self.max_position_size_pct * 1.2  # Go even more aggressive
            self.drawdown_reduction_applied = False
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Position size restored and boosted after drawdown recovery")
        
        # Scale back position sizing after consecutive losses
        if self.consecutive_losses >= 2:
            # Reduce position size after consecutive losses
            self.position_size_pct = max(self.position_size_pct * 0.8, self.max_position_size_pct * 0.4)
            # logger.log(LOG_LEVELS['RISK_MANAGEMENT'], 
            #         f"Position size reduced to {self.position_size_pct:.2%} after {self.consecutive_losses} consecutive losses")
            
    def _get_observation(self):
        """Get the current trading state observation with enhanced features"""
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
            # New features - Enhanced risk management
            self.days_in_trade / 10 if hasattr(self, 'days_in_trade') and self.days_in_trade > 0 else 0,  # Days in trade
            1 if hasattr(self, 'trading_stopped_for_day') and self.trading_stopped_for_day else 0,  # Trading stopped
            self.open_positions_count / MAX_OPEN_POSITIONS if hasattr(self, 'open_positions_count') else 0,  # Position ratio
            self.daily_trades / MAX_TRADES_PER_DAY if hasattr(self, 'daily_trades') else 0,  # Daily trades ratio
            self.max_drawdown * 10,  # Amplify drawdown signal
            self.consecutive_losses / 5 if hasattr(self, 'consecutive_losses') and self.consecutive_losses > 0 else 0,
            self.consecutive_wins / 5 if hasattr(self, 'consecutive_wins') and self.consecutive_wins > 0 else 0,
            # Technical signals and market regime
            self.buy_signal_strength / 10 if hasattr(self, 'buy_signal_strength') else 0,  # Buy signal strength
            self.sell_signal_strength / 10 if hasattr(self, 'sell_signal_strength') else 0,  # Sell signal strength
            # Market regime (one-hot encoding)
            1 if hasattr(self, 'current_market_regime') and self.current_market_regime == "trending_up" else 0,
            1 if hasattr(self, 'current_market_regime') and self.current_market_regime == "trending_down" else 0,
            1 if hasattr(self, 'current_market_regime') and self.current_market_regime == "ranging" else 0,
            1 if hasattr(self, 'current_market_regime') and self.current_market_regime == "volatile" else 0
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
            
            # Volume-Based Indicators
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
            
            # ADX (Average Directional Index)
            if 'ADX' in window_data.columns:
                adx = window_data['ADX'].iloc[-1] / 100  # Normalize ADX to 0-1
            else:
                adx = 0.5
            obs = np.append(obs, [adx])
            
            # Ensure we have exactly 60 elements in the observation
            if len(obs) > 60:
                # Truncate if necessary
                obs = obs[:60]
            elif len(obs) < 60:
                # Pad with zeros if necessary
                obs = np.append(obs, np.zeros(60 - len(obs)))
        else:
            # If not enough data, fill with zeros to maintain array size
            obs = np.append(obs, np.zeros(60 - len(obs)))
        
        return obs.astype(np.float32)

    def calculate_reward(self, action, exit_type=None):
        """Calculate reward based on action and outcome"""
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        
        # Update net worth based on current price
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate base reward as normalized net worth change
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change / self.initial_balance * 100  # Scale relative to initial balance
        
        # Give a small positive reward for holding when price is rising
        if action == 0 and self.shares_held > 0 and net_worth_change > 0:
            reward += 0.1  # Small incentive for holding winning positions
        
        # Add trading action rewards/penalties
        if action == 1:  # Buy
            # Track purchase information
            self.purchase_step = self.current_step
            self.purchase_price = current_price
            
            # Apply a small transaction cost penalty
            reward -= 0.1  # Small fixed penalty for any trade
            
            # Add small incentive to enter positions
            reward += 0.2
        
        elif action == 2 and self.shares_held > 0:  # Sell with shares
            # Calculate post-tax profit instead of raw net worth change
            if hasattr(self, 'purchase_step') and hasattr(self, 'purchase_price'):
                holding_period_days = self.current_step - self.purchase_step
                quantity = self.shares_held
                
                # Calculate post-tax profit using the separate function
                post_tax_profit = self.calculate_post_tax_profit(
                    self.purchase_price, 
                    current_price, 
                    quantity, 
                    holding_period_days
                )
                
                # Use post-tax profit for reward calculation
                reward = post_tax_profit / self.initial_balance * 100
                
                # Additional rewards/penalties based on exit type
                if exit_type == 'target':
                    reward += 2.0
                elif exit_type == 'stop_loss':
                    reward -= 0.25
        
        # Moderate penalty for holding during significant drawdowns
        if self.shares_held > 0:
            current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
            if current_drawdown > 0.1:  # Only penalize for large drawdowns (10%+)
                reward -= current_drawdown * 5
        
        # Update max net worth
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        
        return reward
    
    def step(self, action):
        """Execute one step in the environment with improved trading logic"""
        # Update risk management and market regime
        self._update_risk_management()
        regime_params = self._detect_market_regime()
        
        # Get current price
        current_price = self.df.iloc[self.current_step]["Close"]
        prev_net_worth = self.net_worth
        exit_type = None
        
        # Check if stop loss or target hit before handling action
        if self.shares_held > 0:
            self._update_trailing_stop(current_price, regime_params)
            stop_or_target_hit, exit_type = self._check_stop_and_target(current_price)
            
            if stop_or_target_hit:
                # Force sell if stop loss or target hit
                action = 5  # Sell action
        
        # IMPROVED: Add time filter check
        should_avoid_trading = False
        
        # Check time of day if available
        if 'Time' in self.df.columns:
            current_time = self.df.iloc[self.current_step]['Time']
            
            if hasattr(current_time, 'hour') and hasattr(current_time, 'minute'):
                # Convert to minutes since market open
                minutes_since_open = ((current_time.hour - MARKET_OPEN_HOUR) * 60 + 
                                (current_time.minute - MARKET_OPEN_MINUTE))
                
                # Check if in avoid period - first hour
                if minutes_since_open < AVOID_FIRST_MINUTES:
                    should_avoid_trading = True
                
                # Minutes until close - last hour
                minutes_until_close = ((MARKET_CLOSE_HOUR - current_time.hour) * 60 + 
                                (MARKET_CLOSE_MINUTE - current_time.minute))
                
                if minutes_until_close < AVOID_LAST_MINUTES:
                    should_avoid_trading = True
        
        # IMPROVED: Update technical signals for better decision making
        self._update_technical_signals()
        
        # IMPROVED: Check if trading is allowed based on time, risk management, etc.
        can_trade = not self.trading_stopped_for_day and self.daily_trades < self.max_trades_per_day and not should_avoid_trading
        can_buy = can_trade and self.open_positions_count < MAX_OPEN_POSITIONS
        
        # IMPROVED: Check signal strength - only allow buying on strong signals
        signal_allows_buy = self.buy_signal_strength >= 5  # Require stronger signals
        signal_allows_sell = self.sell_signal_strength >= 4
        
        # IMPROVED: Only allow buying in favorable market regimes
        regime_allows_buy = self.current_market_regime in ["trending_up", "ranging"]
        if self.current_market_regime == "volatile" and self.buy_signal_strength < 7:
            regime_allows_buy = False  # Require even stronger signals in volatile markets
            
        # IMPROVED: Normalize action based on ALL conditions
        if not can_trade and action >= 1 and action <= 4:
            action = 0  # Force hold if trading stopped or in avoid period
        
        if not can_buy and action >= 1 and action <= 4:
            action = 0  # Force hold if max positions reached
            
        if not signal_allows_buy and action >= 1 and action <= 4:
            action = 0  # Force hold if signal strength insufficient
            
        if not regime_allows_buy and action >= 1 and action <= 4:
            action = 0  # Force hold if market regime unfavorable
        
        # Execute trading action
        if action == 0:  # Hold
            pass
        
        elif action >= 1 and action <= 4 and can_buy and signal_allows_buy and regime_allows_buy:  # Buy with different position sizes
            if self.balance > 0 and self.shares_held == 0:  # Only buy if not in position
                # Calculate position size based on action
                position_size_multiplier = action * 0.25  # 25%, 50%, 75%, or 100% of allowed size
                
                # IMPROVED: Apply dynamic position sizing - use smaller size in riskier regimes
                if self.current_market_regime == "volatile":
                    self.position_size_pct *= 0.7  # Reduce position size in volatile markets
                elif self.current_market_regime == "trending_up" and self.buy_signal_strength >= 8:
                    self.position_size_pct *= 1.2  # Increase position size in strong uptrends w/ strong signals
                    self.position_size_pct = min(self.position_size_pct, MAX_POSITION_SIZE_PCT)  # Cap at max
                    
                # Apply dynamic position sizing
                position_size = self.balance * self.position_size_pct * position_size_multiplier
                position_size = min(position_size, self.balance * 0.95)  # Cap at 95% of balance
                
                # Calculate shares to buy
                shares_bought = int(position_size // current_price)

                if shares_bought > 0:
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
                    self.daily_trades += 1
                    self.open_positions_count += 1
                    
                    # Set position type
                    self.position_type = 'long'
                    
                    # IMPROVED: Set more conservative stops and realistic targets
                    self.stop_loss = self._calculate_dynamic_stop_loss(current_price, 'long', regime_params)
                    self.initial_stop_loss = self.stop_loss  # Store initial stop for risk calculation
                    self.trailing_stop = self.stop_loss
                    self.target_price = self._calculate_dynamic_target(current_price, 'long', regime_params)
                    self.highest_price_since_buy = current_price
                    
                    # Reset days in trade counter
                    self.days_in_trade = 0
                    
                    # Store entry date if available
                    if 'Date' in self.df.columns:
                        self.entry_date = self.df.iloc[self.current_step]['Date']
                    if 'Time' in self.df.columns:
                        self.entry_time = self.df.iloc[self.current_step]['Time']

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
                        'target': self.target_price,
                        'position_size_pct': position_size_multiplier * self.position_size_pct,
                        'market_regime': self.current_market_regime,
                        'buy_signal_strength': self.buy_signal_strength,
                        'sell_signal_strength': self.sell_signal_strength
                    })
        
        elif action == 5:  # Sell
            if self.shares_held > 0:
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
                self.daily_trades += 1
                self.open_positions_count -= 1
                
                # Reset position tracking
                position_type = self.position_type
                self.position_type = None
                self.stop_loss = 0
                self.trailing_stop = 0
                self.target_price = 0
                self.highest_price_since_buy = 0
                
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
                    'days_held': self.days_in_trade,
                    'exit_type': exit_type if exit_type else 'manual',
                    'position_type': position_type,
                    'market_regime': self.current_market_regime,
                    'buy_signal_strength': self.buy_signal_strength,
                    'sell_signal_strength': self.sell_signal_strength
                })
                
                # IMPROVED: Update win/loss counters and adjust position sizing accordingly
                if profit > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    
                    # Increase position size after consecutive wins, but cap it
                    if self.consecutive_wins >= 3:
                        self.position_size_pct = min(self.position_size_pct * 1.1, MAX_POSITION_SIZE_PCT * 1.1)
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    
                    # Decrease position size after consecutive losses
                    if self.consecutive_losses >= 2:
                        self.position_size_pct = max(self.position_size_pct * 0.7, MAX_POSITION_SIZE_PCT * 0.4)
        
        # IMPROVED: Implement swing trading mode enhancements
        if SWING_TRADING_MODE and self.position_type == 'long':
            # Only adjust stops/targets if we've been in the trade for at least 2 days
            if self.days_in_trade >= 2:
                current_profit_pct = (current_price - self.cost_basis) / self.cost_basis
                
                # For profitable positions in strong uptrends
                if self.current_market_regime == "trending_up" and current_profit_pct > 0.05:
                    # Allow for more room to breathe with a less aggressive trailing stop
                    new_trailing_stop = current_price * (1 - 0.06)  # 6% retracement allowed (increased from 5%)
                    if new_trailing_stop > self.cost_basis * 1.01 and new_trailing_stop > self.trailing_stop:
                        self.trailing_stop = new_trailing_stop
                    
                    # Set a more ambitious target in strong uptrends
                    if self.target_price < self.cost_basis * 1.20:  # If target is less than 20% gain
                        self.target_price = self.cost_basis * 1.20  # Increase to 20% gain target (from 15%)
        
        # IMPROVED: Calculate reward with enhanced risk-adjustment
        reward = self.calculate_enhanced_reward(action, exit_type)
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
            'target_price': self.target_price,
            'market_regime': self.current_market_regime,
            'daily_trades': self.daily_trades,
            'open_positions': self.open_positions_count,
            'max_drawdown': self.max_drawdown,
            'buy_signal_strength': self.buy_signal_strength,
            'sell_signal_strength': self.sell_signal_strength
        }

        return obs, reward, done, truncated, info

    def detect_support_resistance(df, window=20, threshold=0.02):
        """Detect support and resistance levels"""
        df = df.copy()
        
        # Initialize support and resistance columns
        df['Support'] = np.nan
        df['Resistance'] = np.nan
        
        for i in range(window, len(df)):
            # Get current window
            window_data = df.iloc[i-window:i]
            
            # Find local minima and maxima
            local_min = window_data['Low'].min()
            local_max = window_data['High'].max()
            
            # Check if current price is near support/resistance
            current_price = df.iloc[i]['Close']
            
            # Support
            if abs(current_price - local_min) / local_min < threshold:
                df.loc[df.index[i], 'Support'] = local_min
            
            # Resistance
            if abs(local_max - current_price) / current_price < threshold:
                df.loc[df.index[i], 'Resistance'] = local_max
        
        # Forward fill support/resistance levels
        df['Support'] = df['Support'].ffill()
        df['Resistance'] = df['Resistance'].ffill()
        
        return df
    

    def _update_technical_signals(self):
        """Calculate and update technical trading signals with stronger filters"""
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Skip if not enough data
        if self.current_step < 20:
            return
            
        # Get current window of data
        window = self.df.iloc[self.current_step-20:self.current_step+1]
        current_price = window['Close'].iloc[-1]
        
        # IMPROVED: Track confirmation factors for entry signals
        trend_confirmed = False
        volume_confirmed = False
        rsi_confirmed = False
        pattern_confirmed = False
        macd_confirmed = False
        bollinger_confirmed = False
        
        # 1. Check for trend confirmation (EMA alignment)
        if 'EMA20' in window.columns and 'EMA50' in window.columns and 'EMA100' in window.columns:
            # Strong uptrend: EMA20 > EMA50 > EMA100
            if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1] > window['EMA100'].iloc[-1]:
                trend_confirmed = True
                self.buy_signal_strength += 2  # Base points for aligned EMAs
                
                # Check for steepening trend (EMAs moving apart)
                ema20_50_diff = (window['EMA20'].iloc[-1] - window['EMA50'].iloc[-1]) / window['EMA50'].iloc[-1]
                ema20_50_prev = (window['EMA20'].iloc[-5] - window['EMA50'].iloc[-5]) / window['EMA50'].iloc[-5] if len(window) > 5 else 0
                
                if ema20_50_diff > ema20_50_prev:
                    self.buy_signal_strength += 1  # Additional point for accelerating trend
            
            # Strong downtrend: EMA20 < EMA50 < EMA100    
            elif window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1] < window['EMA100'].iloc[-1]:
                self.sell_signal_strength += 2  # Base points for aligned EMAs in downtrend
                
        # 2. Check basic trend using EMA crossovers
        if 'EMA20' in window.columns and 'EMA50' in window.columns:
            # Bullish trend
            if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1]:
                self.buy_signal_strength += 1
                
                # Bullish crossover (EMA20 crosses above EMA50) - fresh signal
                if window['EMA20'].iloc[-2] <= window['EMA50'].iloc[-2]:
                    self.buy_signal_strength += 3
                    trend_confirmed = True
                    
            # Bearish trend        
            elif window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1]:
                self.sell_signal_strength += 1
                
                # Bearish crossover (EMA20 crosses below EMA50) - fresh signal
                if window['EMA20'].iloc[-2] >= window['EMA50'].iloc[-2]:
                    self.sell_signal_strength += 3
        
        # 3. IMPROVED: Check volume confirmation more carefully
        if 'Volume' in window.columns:
            avg_volume = window['Volume'].iloc[-20:].mean()
            current_volume = window['Volume'].iloc[-1]
            
            # Strong volume spike (for entries, we want above average volume)
            if current_volume > avg_volume * 1.5:
                volume_confirmed = True
                
                # Volume spike with price increase (bullish)
                if window['Close'].iloc[-1] > window['Close'].iloc[-2]:
                    self.buy_signal_strength += 2
                    
                    # Check for multiple consecutive up days on increasing volume (very bullish)
                    if (window['Close'].iloc[-2] > window['Close'].iloc[-3] and 
                        window['Volume'].iloc[-2] > avg_volume * 1.2):
                        self.buy_signal_strength += 2
                        
                # Volume spike with price decrease (bearish)
                elif window['Close'].iloc[-1] < window['Close'].iloc[-2]:
                    self.sell_signal_strength += 2
                    
                    # Check for multiple consecutive down days on increasing volume (very bearish)
                    if (window['Close'].iloc[-2] < window['Close'].iloc[-3] and 
                        window['Volume'].iloc[-2] > avg_volume * 1.2):
                        self.sell_signal_strength += 2
        
        # 4. IMPROVED: Check RSI conditions
        if 'RSI' in window.columns:
            rsi = window['RSI'].iloc[-1]
            prev_rsi = window['RSI'].iloc[-2] if len(window) > 2 else 50
            
            # Oversold zone entry (bullish)
            if rsi < 30:
                rsi_confirmed = True
                self.buy_signal_strength += 3
                
                # RSI turning up from oversold (stronger signal)
                if rsi > prev_rsi:
                    self.buy_signal_strength += 1
                    
            # Overbought zone entry (bearish)    
            elif rsi > 70:
                self.sell_signal_strength += 3
                
                # RSI turning down from overbought (stronger signal)
                if rsi < prev_rsi:
                    self.sell_signal_strength += 1
                    
            # RSI divergence checks
            if len(window) >= 10:
                # Bullish divergence: lower price lows but higher RSI lows
                price_lower = window['Close'].iloc[-1] < window['Close'].iloc[-10]
                rsi_higher = rsi > window['RSI'].iloc[-10]
                
                if price_lower and rsi_higher and rsi < 40:  # Price down but RSI up from low levels
                    self.buy_signal_strength += 4
                    rsi_confirmed = True
                    
                # Bearish divergence: higher price highs but lower RSI highs    
                price_higher = window['Close'].iloc[-1] > window['Close'].iloc[-10]
                rsi_lower = rsi < window['RSI'].iloc[-10]
                
                if price_higher and rsi_lower and rsi > 60:  # Price up but RSI down from high levels
                    self.sell_signal_strength += 4
        
        # 5. IMPROVED: MACD signals
        if all(x in window.columns for x in ['MACD', 'Signal']):
            macd = window['MACD'].iloc[-1]
            signal = window['Signal'].iloc[-1]
            prev_macd = window['MACD'].iloc[-2]
            prev_signal = window['Signal'].iloc[-2]
            
            # Bullish MACD crossover (MACD crosses above Signal)
            if prev_macd < prev_signal and macd > signal:
                self.buy_signal_strength += 3
                macd_confirmed = True
                
            # Bearish MACD crossover (MACD crosses below Signal)    
            elif prev_macd > prev_signal and macd < signal:
                self.sell_signal_strength += 3
                
            # MACD zero line crosses (additional confirmation)
            if prev_macd < 0 and macd >= 0:  # Crosses above zero (bullish)
                self.buy_signal_strength += 2
                macd_confirmed = True
                
            elif prev_macd > 0 and macd <= 0:  # Crosses below zero (bearish)
                self.sell_signal_strength += 2
                
            # MACD histogram expanding (trend strengthening)
            if abs(macd - signal) > abs(prev_macd - prev_signal):
                if macd > signal:  # Bullish momentum increasing
                    self.buy_signal_strength += 1
                else:  # Bearish momentum increasing
                    self.sell_signal_strength += 1
        
        # 6. IMPROVED: Bollinger Band signals
        if all(x in window.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            upper = window['BB_Upper'].iloc[-1]
            middle = window['BB_Middle'].iloc[-1]
            lower = window['BB_Lower'].iloc[-1]
            
            # Price touching lower band (potential buy signal)
            if window['Close'].iloc[-1] <= lower * 1.01:
                self.buy_signal_strength += 2
                bollinger_confirmed = True
                
                # Price bouncing off lower band (stronger buy signal)
                if window['Close'].iloc[-2] <= lower * 1.01 and window['Close'].iloc[-1] > window['Close'].iloc[-2]:
                    self.buy_signal_strength += 2
                    
            # Price touching upper band (potential sell signal)        
            elif window['Close'].iloc[-1] >= upper * 0.99:
                self.sell_signal_strength += 2
                
                # Price bouncing off upper band (stronger sell signal)
                if window['Close'].iloc[-2] >= upper * 0.99 and window['Close'].iloc[-1] < window['Close'].iloc[-2]:
                    self.sell_signal_strength += 2
                    
            # Bollinger Band squeeze (narrowing bands - potential breakout)
            bb_width = (upper - lower) / middle
            prev_upper = window['BB_Upper'].iloc[-10] if len(window) > 10 else upper
            prev_lower = window['BB_Lower'].iloc[-10] if len(window) > 10 else lower
            prev_middle = window['BB_Middle'].iloc[-10] if len(window) > 10 else middle
            prev_bb_width = (prev_upper - prev_lower) / prev_middle
            
            if bb_width < prev_bb_width * 0.8:  # Band width narrowing by 20%+
                # If EMA alignment suggests uptrend
                if 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1]:
                    self.buy_signal_strength += 2
                    bollinger_confirmed = True
                # If EMA alignment suggests downtrend
                elif 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1]:
                    self.sell_signal_strength += 2
        
        # 7. IMPROVED: Candlestick pattern recognition
        if all(x in window.columns for x in ['Open', 'High', 'Low', 'Close']):
            # Bullish engulfing
            if (window['Open'].iloc[-2] > window['Close'].iloc[-2] and  # Previous red candle
                window['Open'].iloc[-1] < window['Close'].iloc[-1] and  # Current green candle
                window['Open'].iloc[-1] <= window['Close'].iloc[-2] and
                window['Close'].iloc[-1] > window['Open'].iloc[-2]):
                self.buy_signal_strength += 3
                pattern_confirmed = True
                
            # Bearish engulfing    
            if (window['Open'].iloc[-2] < window['Close'].iloc[-2] and  # Previous green candle
                window['Open'].iloc[-1] > window['Close'].iloc[-1] and  # Current red candle
                window['Open'].iloc[-1] >= window['Close'].iloc[-2] and
                window['Close'].iloc[-1] < window['Open'].iloc[-2]):
                self.sell_signal_strength += 3
                
            # Hammer (bullish reversal)
            body_size = abs(window['Open'].iloc[-1] - window['Close'].iloc[-1])
            lower_shadow = min(window['Open'].iloc[-1], window['Close'].iloc[-1]) - window['Low'].iloc[-1]
            upper_shadow = window['High'].iloc[-1] - max(window['Open'].iloc[-1], window['Close'].iloc[-1])
            
            if (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < 0.2 * body_size and  # Minimal upper shadow
                window['Close'].iloc[-5:-1].min() < window['Close'].iloc[-1]):  # In downtrend
                self.buy_signal_strength += 3
                pattern_confirmed = True
            
            # Shooting star (bearish reversal)
            if (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < 0.2 * body_size and  # Minimal lower shadow
                window['Close'].iloc[-5:-1].max() > window['Close'].iloc[-1]):  # In uptrend
                self.sell_signal_strength += 3
        
        # 8. IMPROVED: Support/Resistance tests
        if 'Support' in window.columns and 'Resistance' in window.columns:
            support = window['Support'].iloc[-1]
            resistance = window['Resistance'].iloc[-1]
            
            # Price bouncing off support
            if current_price <= support * 1.02 and window['Close'].iloc[-1] > window['Close'].iloc[-2]:
                self.buy_signal_strength += 3
                pattern_confirmed = True
                
            # Price bouncing off resistance    
            if current_price >= resistance * 0.98 and window['Close'].iloc[-1] < window['Close'].iloc[-2]:
                self.sell_signal_strength += 3
        
        # 9. IMPROVED: Check ADX for trend strength
        if 'ADX' in window.columns:
            adx = window['ADX'].iloc[-1]
            
            # Strong trend present
            if adx > 25:
                # In uptrend
                if 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1]:
                    self.buy_signal_strength += 2
                    trend_confirmed = True
                # In downtrend
                elif 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1]:
                    self.sell_signal_strength += 2
                    
            # Very strong trend
            if adx > 35:
                # In uptrend
                if 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1]:
                    self.buy_signal_strength += 2
                    trend_confirmed = True
                # In downtrend
                elif 'EMA20' in window.columns and 'EMA50' in window.columns and window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1]:
                    self.sell_signal_strength += 2
        
        # IMPROVED: Only generate strong buy signals when multiple conditions align
        # Count how many confirmation types we have
        buy_confirmations = sum([trend_confirmed, volume_confirmed, rsi_confirmed, pattern_confirmed, macd_confirmed, bollinger_confirmed])
        sell_confirmations = 0  # Could implement similar logic for sell signals
        
        # Require more confirmations for valid signals
        if buy_confirmations < 2:
            # Severely reduce signal if not enough confirmations
            self.buy_signal_strength = max(0, self.buy_signal_strength - 4)
        elif buy_confirmations >= 3:
            # Bonus for strong multi-factor confirmation
            self.buy_signal_strength += 2
            
        # IMPROVED: Normalize signal strength to avoid excessive values
        self.buy_signal_strength = min(self.buy_signal_strength, 10)
        self.sell_signal_strength = min(self.sell_signal_strength, 10)
                    

    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode with enhanced risk management"""
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
        
        # Reset enhanced risk management
        self.daily_trades = 0
        self.daily_high_capital = self.initial_balance
        self.daily_low_capital = self.initial_balance
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
        self.open_positions_count = 0
        
        # Reset technical signals
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Reset episode variables
        self.current_step = self.window_size
        self.rewards = []
        self.trades = []
        self.net_worth_history = [self.initial_balance]
        self.total_transaction_costs = 0
        self.total_trade_value = 0
        self.last_txn_cost = 0
        
        return self._get_observation(), {}
    
    def render(self, mode='human', close=False):
        """Render the environment"""
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
    
    def calculate_performance_metrics(self):
        """Calculate trading performance metrics with enhanced risk metrics"""
        # Calculate daily returns
        returns = [0]
        for i in range(1, len(self.net_worth_history)):
            returns.append((self.net_worth_history[i] - self.net_worth_history[i-1]) / self.net_worth_history[i-1])
        
        # Calculate metrics
        total_return = (self.net_worth_history[-1] - self.initial_balance) / self.initial_balance
        
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
        exit_types = {'target': 0, 'stop_loss': 0, 'manual': 0, 'market_close': 0, 
                    'time_based_exit': 0, 'regime_volatile_take_profit': 0, 'regime_ranging_take_profit': 0}
                    
        for trade in self.trades:
            if trade['type'] == 'sell' and 'exit_type' in trade:
                exit_type = trade['exit_type']
                exit_types[exit_type] = exit_types.get(exit_type, 0) + 1
        
        # Enhanced metrics
        avg_win_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else float('inf')
        recovery_factor = total_return / max_dd if max_dd > 0 else float('inf')
        
        # Holding period statistics
        holding_periods = []
        for trade in self.trades:
            if trade['type'] == 'sell' and 'days_held' in trade:
                holding_periods.append(trade['days_held'])
                
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # Market regime performance
        regime_returns = {'trending_up': [], 'trending_down': [], 'ranging': [], 'volatile': [], 'unknown': []}
        regime_trades = {'trending_up': 0, 'trending_down': 0, 'ranging': 0, 'volatile': 0, 'unknown': 0}
        
        for trade in self.trades:
            if trade['type'] == 'sell' and 'market_regime' in trade and 'profit_pct' in trade:
                regime = trade['market_regime']
                regime_returns[regime].append(trade['profit_pct'])
                regime_trades[regime] += 1
                
        regime_performance = {}
        for regime, returns in regime_returns.items():
            if returns:
                regime_performance[regime] = {
                    'avg_return': np.mean(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                    'trade_count': len(returns)
                }
            else:
                regime_performance[regime] = {'avg_return': 0, 'win_rate': 0, 'trade_count': 0}
        
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
            'avg_win_loss_ratio': avg_win_loss_ratio,
            'recovery_factor': recovery_factor,
            'avg_holding_period': avg_holding_period,
            'regime_performance': regime_performance
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

def train_enhanced_model(train_df, symbol, **env_kwargs):
    """Train an enhanced model with improved hyperparameters"""
    logger.log(LOG_LEVELS['MODEL_TRAINING'], f"Training enhanced model for {symbol}...")
    
    # Set up enhanced parameters
    enhanced_params = {
        'learning_rate': 3e-5,
        'n_steps': 2048,
        'batch_size': 128,
        'gae_lambda': 0.95,
        'gamma': 0.9995,  # Higher discount factor for long-term thinking
        'n_epochs': 15,
        'ent_coef': 0.01,  # Higher entropy for more exploration
        'clip_range': 0.2,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'policy_kwargs': dict(
            net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
        )
    }
    
    # Create environment
    env = create_vec_env(train_df, symbol, **env_kwargs)
    
    # Create model
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        **enhanced_params,
        device=device
    )
    
    # Train model
    total_timesteps = 100000
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model_path = f"{MODEL_SAVE_PATH}/enhanced_{symbol}.zip"
    model.save(model_path)
    logger.log(LOG_LEVELS['MODEL_TRAINING'], f"Enhanced model saved to {model_path}")
    
    return model, model_path

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

def portfolio_strategy(symbols, test_dfs, models, initial_balance=DEFAULT_INITIAL_BALANCE, allocation_method='equal'):
    """Run a portfolio strategy across multiple stocks"""
    logger.log(LOG_LEVELS['MODEL_EVALUATION'], f"Running portfolio strategy for {len(symbols)} symbols...")
    
    # Initialize portfolio
    portfolio = {
        'initial_balance': initial_balance,
        'current_balance': initial_balance,
        'allocations': {},
        'positions': {},
        'history': []
    }
    
    # Allocate capital
    if allocation_method == 'equal':
        allocation_per_symbol = initial_balance / len(symbols)
        for symbol in symbols:
            portfolio['allocations'][symbol] = allocation_per_symbol
    elif allocation_method == 'volatility_weighted':
        # Calculate inverse volatility for each symbol
        volatilities = {}
        for symbol, df in test_dfs.items():
            returns = df['Close'].pct_change().dropna()
            vol = returns.std()
            volatilities[symbol] = 1/vol if vol > 0 else 0
            
        # Normalize weights
        total_inv_vol = sum(volatilities.values())
        for symbol in symbols:
            weight = volatilities[symbol] / total_inv_vol if total_inv_vol > 0 else 1/len(symbols)
            portfolio['allocations'][symbol] = initial_balance * weight
    
    # Run simulation for each symbol
    results = {}
    for symbol in symbols:
        # Run trading with allocated capital
        env_kwargs = {
            'initial_balance': portfolio['allocations'][symbol],
            'window_size': 20,
            'use_trailing_stop': True,
            'trailing_stop_atr_multiplier': 2.0,
            'target_atr_multiplier': 3.0,
            'min_rr_ratio': 1.5,
            'max_position_size_pct': MAX_POSITION_SIZE_PCT,
            'max_trades_per_day': MAX_TRADES_PER_DAY,
            'daily_loss_limit_pct': DAILY_LOSS_LIMIT_PCT
        }
        
        # Create environment
        env = EnhancedIndianStockTradingEnv(df=test_dfs[symbol], symbol=symbol, **env_kwargs)
        
        # Run evaluation
        obs, _ = env.reset()
        done = False
        rewards = []
        
        while not done:
            action, _ = models[symbol].predict(obs, deterministic=False)
            obs, reward, done, _, info = env.step(action)
            rewards.append(reward)
            
            # Track portfolio value
            if len(portfolio['history']) <= env.current_step:
                portfolio['history'].append({})
                
            portfolio['history'][env.current_step][symbol] = info['net_worth']
        
        # Store results
        results[symbol] = {
            'final_balance': env.net_worth,
            'total_return': (env.net_worth - portfolio['allocations'][symbol]) / portfolio['allocations'][symbol],
            'max_drawdown': env.max_drawdown,
            'sharpe_ratio': env.calculate_performance_metrics()['sharpe_ratio'],
            'win_rate': env.calculate_performance_metrics()['win_rate']
        }
        
        # Update portfolio balance
        portfolio['current_balance'] += env.net_worth - portfolio['allocations'][symbol]
    
    # Calculate portfolio metrics
    portfolio_returns = []
    for step in range(len(portfolio['history'])):
        if step > 0:
            prev_total = sum(portfolio['history'][step-1].values())
            curr_total = sum(portfolio['history'][step].values())
            if prev_total > 0:
                portfolio_returns.append((curr_total - prev_total) / prev_total)
    
    # Calculate portfolio performance
    portfolio_sharpe = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252) if np.std(portfolio_returns) > 0 else 0
    portfolio_total_return = (portfolio['current_balance'] - initial_balance) / initial_balance
    
    # Calculate max drawdown for portfolio
    portfolio_values = [sum(day.values()) for day in portfolio['history']]
    portfolio_peak = np.maximum.accumulate(portfolio_values)
    portfolio_drawdowns = (portfolio_peak - portfolio_values) / portfolio_peak
    portfolio_max_drawdown = np.max(portfolio_drawdowns) if len(portfolio_drawdowns) > 0 else 0
    
    # Return portfolio results
    return {
        'initial_balance': initial_balance,
        'final_balance': portfolio['current_balance'],
        'total_return': portfolio_total_return,
        'annual_return': (1 + portfolio_total_return) ** (252 / len(portfolio['history'])) - 1,
        'sharpe_ratio': portfolio_sharpe,
        'max_drawdown': portfolio_max_drawdown,
        'symbols': symbols,
        'symbol_results': results
    }

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
    """
    Optimize hyperparameters for the RL trading strategy with enhanced parameters.
    
    This function runs optimization for a single trial with Optuna, exploring a wide
    range of both model and trading-specific hyperparameters to find optimal settings.
    """
    # ---- MODEL HYPERPARAMETERS ----
    # Define the hyperparameter search space with improved ranges
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 5e-3)  # Lower range for more stability
    n_steps = trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096, 8192])  # More options including larger steps
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])  # Include larger batch sizes
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.9, 0.995)  # Tighter range for stability
    gamma = trial.suggest_uniform('gamma', 0.95, 0.9999)  # Higher discount factors to focus on long-term rewards
    n_epochs = trial.suggest_int('n_epochs', 5, 20)  # More epochs options
    ent_coef = trial.suggest_loguniform('ent_coef', 0.0001, 0.05)  # Lower entropy to reduce unnecessary exploration
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)  # Standard PPO clipping range
    
    # ---- TRADING STRATEGY HYPERPARAMETERS ----
    # Risk management parameters
    trailing_stop_atr_multiplier = trial.suggest_uniform('trailing_stop_atr_multiplier', 2.0, 4.0)  # Wider stops
    target_atr_multiplier = trial.suggest_uniform('target_atr_multiplier', 3.0, 8.0)  # Higher targets
    min_rr_ratio = trial.suggest_uniform('min_rr_ratio', 1.5, 3.5)  # Higher risk-reward requirements
    max_position_size_pct = trial.suggest_uniform('max_position_size_pct', 0.05, 0.30)  # Lower position sizes
    max_trades_per_day = trial.suggest_int('max_trades_per_day', 1, 5)  # Fewer trades per day
    
    # Advanced trading parameters
    use_bollinger_breakouts = trial.suggest_categorical('use_bollinger_breakouts', [True, False])
    use_rsi_filter = trial.suggest_categorical('use_rsi_filter', [True, False])
    use_volume_filter = trial.suggest_categorical('use_volume_filter', [True, False])
    use_profit_trailing = trial.suggest_categorical('use_profit_trailing', [True, False])
    min_signal_strength = trial.suggest_int('min_signal_strength', 3, 7)  # Minimum signal strength for entry
    
    # Reward function parameters
    transaction_cost_penalty = trial.suggest_uniform('transaction_cost_penalty', 0.1, 1.0)
    holding_reward_multiplier = trial.suggest_uniform('holding_reward_multiplier', 0.05, 0.3)
    drawdown_penalty_multiplier = trial.suggest_uniform('drawdown_penalty_multiplier', 2.0, 8.0)
    excessive_trading_penalty = trial.suggest_uniform('excessive_trading_penalty', 1.0, 8.0)
    
    # Swing trading parameters
    swing_mode = trial.suggest_categorical('swing_mode', [True, False])
    early_exit_threshold = trial.suggest_uniform('early_exit_threshold', -0.05, -0.02)  # When to exit losing trades
    
    # ---- NETWORK ARCHITECTURE PARAMETERS ----
    # Try different network architectures
    network_width = trial.suggest_categorical('network_width', [64, 128, 256, 512])
    network_depth = trial.suggest_int('network_depth', 2, 4)
    
    # Build network architecture dynamically
    if network_depth == 2:
        network = [network_width, network_width // 2]
    elif network_depth == 3:
        network = [network_width, network_width // 2, network_width // 4]
    else:  # depth == 4
        network = [network_width, network_width // 2, network_width // 3, network_width // 4]
    
    policy_kwargs = dict(
        net_arch=[dict(pi=network, vf=network)]
    )
    
    trading_params = {
            'trailing_stop_atr_multiplier': trailing_stop_atr_multiplier,
            'target_atr_multiplier': target_atr_multiplier,
            'min_rr_ratio': min_rr_ratio,
            'max_position_size_pct': max_position_size_pct,
            'max_trades_per_day': max_trades_per_day,
        }
    
    # Store additional parameters separately for use in the custom environment
    additional_params = {
        'USE_BOLLINGER_BREAKOUTS': use_bollinger_breakouts,
        'USE_RSI_FILTER': use_rsi_filter,
        'USE_VOLUME_FILTER': use_volume_filter,
        'USE_PROFIT_TRAILING': use_profit_trailing,
        'SWING_TRADING_MODE': swing_mode,
    }
    
    # Update environment kwargs with only the parameters accepted by IndianStockTradingEnv
    updated_env_kwargs = env_kwargs.copy()
    updated_env_kwargs.update(trading_params)
    
    # Create temporary environment class with custom reward parameters
    class CustomEnv(IndianStockTradingEnv):
        def __init__(self, df, symbol, **kwargs):
            # Extract the additional parameters we need
            self.USE_BOLLINGER_BREAKOUTS = additional_params['USE_BOLLINGER_BREAKOUTS']
            self.USE_RSI_FILTER = additional_params['USE_RSI_FILTER'] 
            self.USE_VOLUME_FILTER = additional_params['USE_VOLUME_FILTER']
            self.USE_PROFIT_TRAILING = additional_params['USE_PROFIT_TRAILING']
            self.SWING_TRADING_MODE = additional_params['SWING_TRADING_MODE']
            
            # Call parent constructor with only the parameters it expects
            super().__init__(df, symbol, **kwargs)
            
        def calculate_enhanced_reward(self, action, exit_type=None):
            # Get the standard reward calculation
            reward = super().calculate_enhanced_reward(action, exit_type)
            
            # Apply trial-specific modifiers
            if action >= 1 and action <= 4:  # Buy actions
                reward -= transaction_cost_penalty  # Penalty for trading
                
                # Penalize for excessive trades
                if self.daily_trades >= self.max_trades_per_day - 1:
                    reward -= excessive_trading_penalty
                
                # Penalize buy signals below threshold
                if self.buy_signal_strength < min_signal_strength:
                    reward -= 3.0
                    
            elif action == 5 and self.shares_held > 0:  # Sell actions
                # Scale reward by holding period to encourage longer holds for profitable trades
                if hasattr(self, 'days_in_trade') and self.days_in_trade > 0:
                    holding_multiplier = min(1 + (self.days_in_trade * holding_reward_multiplier), 3.0)
                    reward *= holding_multiplier
            
            # Penalize drawdowns more heavily
            if self.shares_held > 0:
                current_drawdown = (self.max_net_worth - self.net_worth) / self.max_net_worth
                if current_drawdown > 0.05:  # Lower threshold for drawdown penalty
                    reward -= current_drawdown * drawdown_penalty_multiplier
                    
            return reward
            
        def _check_stop_and_target(self, current_price):
            # Get standard stop/target logic
            hit_exit, exit_type = super()._check_stop_and_target(current_price)
            
            # Add custom early exit for losing trades
            if not hit_exit and self.position_type and 'cost_basis' in self.__dict__:
                profit_pct = 0
                if self.position_type == 'long':
                    profit_pct = (current_price - self.cost_basis) / self.cost_basis
                else:  # short
                    profit_pct = (self.cost_basis - current_price) / self.cost_basis
                    
                # Exit losing trades earlier based on trial parameter
                if profit_pct < early_exit_threshold and self.days_in_trade > 2:
                    return True, 'early_loss_exit'
                    
            return hit_exit, exit_type
    
    # Create environment with custom parameters
    train_env = DummyVecEnv([lambda: CustomEnv(df=train_df, symbol=symbol, **updated_env_kwargs)])
    
    # Create a PPO model with the suggested hyperparameters
    model = PPO(
        'MlpPolicy',
        train_env,
        verbose=0,  # Reduce verbosity for optimization
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

    # Train the model with early pruning capability
    try:
        # Use a relatively short training period for faster optimization
        timesteps = 50_000
        model.learn(total_timesteps=timesteps)
        
        # Evaluate on a validation set (a segment of the training data we haven't trained on)
        # This helps prevent overfitting in the hyperparameter search
        valid_size = min(10_000, len(train_df) // 4)
        valid_df = train_df.iloc[-valid_size:].copy().reset_index(drop=True)
        eval_env = CustomEnv(df=valid_df, symbol=symbol, **updated_env_kwargs)
        
        # Run episodes on validation data
        rewards = []
        n_eval_episodes = 3  # Multiple episodes for more robust evaluation
        
        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = eval_env.step(action)
                episode_reward += reward
                
            rewards.append(episode_reward)
            
        # Calculate metrics that matter most: return, Sharpe, win rate
        mean_reward = np.mean(rewards)
        metrics = eval_env.calculate_performance_metrics()
        
        # Create a composite score that balances different objectives
        # Higher weight on Sharpe ratio and win rate
        total_return = metrics['total_return'] if metrics['total_return'] > -1 else -1  # Cap extreme losses
        sharpe_ratio = metrics['sharpe_ratio'] if metrics['sharpe_ratio'] > -3 else -3  # Cap extreme negative Sharpes
        win_rate = metrics['win_rate']
        max_drawdown = metrics['max_drawdown'] if metrics['max_drawdown'] < 0.5 else 0.5  # Cap extreme drawdowns
        
        # Composite score: prioritize Sharpe and win rate over raw returns
        score = (
            0.3 * total_return + 
            0.4 * sharpe_ratio + 
            0.2 * win_rate - 
            0.1 * max_drawdown
        )
        
        # Print selected params and performance for monitoring
        print(f"\nTrial {trial.number}:")
        print(f"Learning rate: {learning_rate:.6f}")
        print(f"Trailing stop multiplier: {trailing_stop_atr_multiplier:.2f}")
        print(f"Max position size: {max_position_size_pct:.2f}")
        print(f"Max trades per day: {max_trades_per_day}")
        print(f"Performance - Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, Win rate: {win_rate:.2%}")
        print(f"Composite score: {score:.4f}")
        
        return score  # Return composite score instead of just reward
        
    except Exception as e:
        print(f"Error in trial {trial.number}: {str(e)}")
        # Return a very low score for failed trials
        return -10.0

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
            sampler = TPESampler(n_startup_trials=50)
            pruner = optuna.pruners.HyperbandPruner(
                min_resource=10, 
                max_resource=30, 
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

            n_trials = 300
            n_jobs = 3  # Number of processes to run in parallel
            # n_jobs = max(1, os.cpu_count() - 1)  # Use all but one CPU core
            print(f"Running {n_trials} trials with {n_jobs} parallel jobs")

            with tqdm(total=n_trials, desc="Hyperparameter Tuning", unit="trials") as pbar:
                study.optimize(
                    lambda trial: optimize_hyperparameters(trial, train_df, symbol, **env_kwargs),
                    n_trials=n_trials,
                    n_jobs=n_jobs,
                    callbacks=[
                        lambda study, trial: pbar.update(1)
                    ]
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
                verbose=1,
                **best_hyperparameters,
                # tensorboard_log=f"./tensorboard/{symbol}/",
                device=device
            )

            # --- Load last trained steps from DB ---
            c.execute("SELECT last_trained_steps FROM model_training_progress WHERE symbol=?", (symbol,))
            row = c.fetchone()
            last_trained_steps = row[0] if row else 0

            print(f"➡️ Last trained steps for {symbol}: {last_trained_steps}")

            total_timesteps = 100000
            log_interval = 1000
            batch_size = 2048

            with Live(refresh_per_second=4) as live:
                for i in range(0, total_timesteps, 2048):
                    # Train model
                    start_time = time.time()
                    best_model.learn(total_timesteps=min(2048, total_timesteps - i), reset_num_timesteps=False,progress_bar=True,log_interval=log_interval)
                    end_time = time.time()

                    # Evaluate model to get reward
                    mean_reward, _ = evaluate_policy(best_model, create_vec_env(train_df, symbol, **env_kwargs), n_eval_episodes=5, warn=False)

                    # **CREATE A NEW TABLE FOR EACH UPDATE**
                    update_table = Table(title="Training Progress")
                    update_table.add_column("Step", style="cyan")
                    update_table.add_column("Reward", justify="right", style="magenta")
                    update_table.add_column("Time", justify="right", style="green")
                    
                    # **ADD THE NEW ROW TO THE NEW TABLE**
                    update_table.add_row(f"{i+batch_size}", f"{mean_reward:.2f}", f"{end_time - start_time:.2f} sec")
                    
                    # **UPDATE LIVE WITH THE NEW TABLE**
                    live.update(update_table, refresh=True)

                    # --- Save model and update DB after every batch ---
                    best_model.save(model_path)
                    c.execute("INSERT OR REPLACE INTO model_training_progress (symbol, last_trained_steps) VALUES (?, ?)",(symbol, i + batch_size))

                    conn.commit()
                    
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
        
        # --- Close DB connection at the end ---
        conn.close()

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
            trade_headers = ["Episode", "Timestamp", "Action", "Price (₹)", "Shares", "Value (₹)", 
                            "Capital After (₹)", "Stop Loss (₹)", "Target (₹)"]
            
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
            holding_headers = ["Episode", "Entry Time", "Exit Time", "Holding Period", "Entry Price (₹)", 
                            "Exit Price (₹)", "Shares", "P/L (₹)", "P/L %", "Exit Reason"]
            
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