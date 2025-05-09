"""
Indian market constants and configuration settings with improved parameters.
"""

from datetime import datetime, timedelta

class IndianMarketConstants:
    """Constants specific to Indian stock market"""
    MARKET_OPEN_TIME = datetime.strptime("09:15:00", "%H:%M:%S").time()
    MARKET_CLOSE_TIME = datetime.strptime("15:30:00", "%H:%M:%S").time()
    
    # Tax & brokerage constants
    STT_RATE = 0.00025  # Securities Transaction Tax (0.025% on sell)
    EXCHANGE_TXN_CHARGE = 0.0000325  # NSE transaction charge (0.00325%)
    GST_RATE = 0.18  # GST on brokerage and transaction charges (18%)
    SEBI_CHARGES = 0.000001  # SEBI charges (0.0001%)
    STAMP_DUTY = 0.00015  # Stamp duty (0.015% on buy)
    
    # Broker specific (default for Zerodha)
    BROKERAGE_RATE = 0.0003  # 0.03% or Rs. 20 per order whichever is lower
    MAX_BROKERAGE_PER_ORDER = 20  # Maximum Rs. 20 per order
    
    # RL specific
    MAX_STEPS = 400  # Maximum steps per episode
    MAX_EP_DURATION = timedelta(hours=6)  # Maximum episode duration
    
    # Exchange holiday list 2025 (update as needed)
    HOLIDAYS = [
        "2025-01-26",  # Republic Day
        "2025-03-01",  # Mahashivratri
        "2025-03-25",  # Holi
        "2025-04-14",  # Dr. Ambedkar Jayanti
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day
        "2025-08-15",  # Independence Day
        "2025-10-02",  # Gandhi Jayanti
        "2025-10-24",  # Dussehra
        "2025-11-12",  # Diwali
        "2025-12-25",  # Christmas
    ]


class TrainingConfig:
    # Directory settings - ADDED THESE LINES
    TENSORBOARD_LOG = "logs/tensorboard/"  # Fixed the missing attribute
    MODEL_DIR = "models/"
    LOG_DIR = "logs/"
    
    # Original parameters
    WINDOW_SIZE = 10
    INITIAL_BALANCE = 100000.0
    DEFAULT_TIMESTEPS = 1000000
    REWARD_THRESHOLD = 50.0
    N_ENVS = 4
    
    # Original PPO parameters
    LEARNING_RATE = 0.0003
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    
    # Modified parameters for improved performance
    WINDOW_SIZE = 15  # Increase window size for more context
    DEFAULT_TIMESTEPS = 1500000  # More training steps
    REWARD_THRESHOLD = 100.0  # Higher reward threshold for better models
    N_ENVS = 8  # More parallel environments
    LOG_INTERVAL = 500  # More frequent logging
    
    # Modified PPO hyperparameters
    LEARNING_RATE = 3e-5  # Smaller learning rate for more stability
    N_STEPS = 4096  # Larger batch collection for more stable updates
    BATCH_SIZE = 256  # Larger mini-batches
    N_EPOCHS = 15  # More epochs per update
    GAMMA = 0.995  # Higher discount factor for better long-term planning
    GAE_LAMBDA = 0.97  # Better advantage estimation
    CLIP_RANGE = 0.15  # Smaller clip range for more stable updates
    ENT_COEF = 0.005  # Less exploration (more exploitation)
    VF_COEF = 0.7  # More emphasis on value function accuracy
    MAX_GRAD_NORM = 0.3  # Tighter gradient clipping
    
    # New parameters
    EARLY_STOPPING_PATIENCE = 20  # Stop training if no improvement
    USE_LINEAR_SCHEDULE = True  # Gradually decrease learning rate
    VALIDATION_INTERVAL = 10000  # Validate more frequently
    DROPOUT_RATE = 0.1  # Add dropout to policy network
    
    # Feature importance weighting
    REWARD_WEIGHTS = {
        'profit': 1.0,
        'drawdown': -3.0,        # Heavy penalty for drawdowns
        'trade_frequency': 0.2,  # Slight reward for activity
        'success_rate': 1.5,     # Big reward for high success rate
        'holding_time': -0.1     # Slight penalty for holding too long
    }


class TradingConfig:
    """Improved configuration for live trading"""
    DEFAULT_STOP_LOSS_PCT = 0.01  # Tighter stop-loss (1% instead of 2%)
    DEFAULT_TAKE_PROFIT_PCT = 0.02  # New take-profit parameter (2%)
    MAX_DRAWDOWN_PCT = 0.10  # Maximum 10% drawdown allowed
    MAX_POSITION_SIZE = 0.05  # Maximum 5% of capital per position
    DAILY_LOSS_LIMIT_PCT = 0.02  # Maximum 2% daily loss
    
    CHECK_INTERVAL = 5  # seconds
    CANDLE_INTERVAL = "5minute"
    DEFAULT_BROKER = "zerodha"
    DEFAULT_MODEL_PATH = "./models/scalping_model_v2.zip"
    
    # Time-based trading restrictions
    AVOID_FIRST_MINUTES = 15  # Avoid trading in first 15 minutes after market open
    AVOID_LAST_MINUTES = 15  # Avoid trading in last 15 minutes before market close
    
    # Position management
    TRAILING_STOP_ACTIVATION_PCT = 0.01  # Activate trailing stop when profit reaches 1%
    TRAILING_STOP_DISTANCE_PCT = 0.005  # Trailing stop follows price at 0.5% distance