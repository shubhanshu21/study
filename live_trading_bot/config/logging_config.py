"""
Logging configuration for the trading bot.
"""
import os
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import sys
from datetime import datetime

def setup_logging(level='INFO', log_dir='logs', service_name='rl_trading_bot'):
    """
    Set up logging configuration
    
    Args:
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_dir (str): Directory for log files
        service_name (str): Name of the service for log files
        
    Returns:
        logging.Logger: Configured logger
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get numeric log level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Create base logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Daily rotating file handler
    current_date = datetime.now().strftime('%Y-%m-%d')
    daily_file_handler = TimedRotatingFileHandler(
        os.path.join(log_dir, f'{service_name}_{current_date}.log'),
        when='midnight',
        backupCount=30  # Keep 30 days of logs
    )
    daily_file_handler.setLevel(numeric_level)
    daily_file_handler.setFormatter(formatter)
    logger.addHandler(daily_file_handler)
    
    # Error file handler (separate file for errors)
    error_file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{service_name}_error.log'),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(formatter)
    logger.addHandler(error_file_handler)
    
    # Trading file handler (separate file for trading actions)
    trading_file_handler = RotatingFileHandler(
        os.path.join(log_dir, f'{service_name}_trades.log'),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=10
    )
    trading_file_handler.setLevel(logging.INFO)
    trading_file_handler.setFormatter(formatter)
    
    # Add a filter to only include trading-related logs
    class TradeFilter(logging.Filter):
        def filter(self, record):
            return any(keyword in record.message for keyword in 
                      ['trade', 'order', 'buy', 'sell', 'position', 'stop', 'target'])
    
    trading_file_handler.addFilter(TradeFilter())
    logger.addHandler(trading_file_handler)
    
    return logger