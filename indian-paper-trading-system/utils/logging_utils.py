import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from config.trading_config import LOG_LEVEL

# Define log levels for specific events
LOG_LEVELS = {
    'DATA_LOAD': logging.INFO,
    'MODEL_TRAINING': logging.INFO,
    'MODEL_EVALUATION': logging.INFO,
    'PERFORMANCE_METRICS': logging.INFO,
    'EXCEPTION': logging.ERROR,
    'RISK_MANAGEMENT': logging.WARNING,
    'TRADE_EXECUTION': logging.INFO,
    'MARKET_DATA': logging.DEBUG,
    'SCHEDULER': logging.INFO,
    'DATABASE': logging.INFO,
    'DASHBOARD': logging.INFO
}

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',   # Green
        'WARNING': '\033[33m', # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m', # Red background
        'RESET': '\033[0m'    # Reset
    }
    
    def format(self, record):
        """Format log record with colors"""
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging(log_dir="logs", log_file="paper_trading.log", console_level=None, file_level=None):
    """Set up logging with console and file handlers
    
    Args:
        log_dir: Directory for log files
        log_file: Log file name
        console_level: Console logging level (default from LOG_LEVEL in config)
        file_level: File logging level (default from LOG_LEVEL in config)
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Determine logging levels
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    config_level = level_map.get(LOG_LEVEL, logging.INFO)
    console_level = console_level or config_level
    file_level = file_level or config_level
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # Create colored formatter for console
    console_formatter = ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Create file handler (rotating by size, max 5 files of 10MB each)
    log_path = os.path.join(log_dir, log_file)
    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(file_level)
    
    # Create formatter for file
    file_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Create daily rotating file handler for archiving
    daily_log_file = os.path.join(log_dir, "daily", f"paper_trading_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Create daily directory if it doesn't exist
    daily_dir = os.path.join(log_dir, "daily")
    if not os.path.exists(daily_dir):
        os.makedirs(daily_dir)
    
    daily_handler = TimedRotatingFileHandler(
        filename=daily_log_file,
        when='midnight',
        interval=1,
        backupCount=30  # Keep logs for 30 days
    )
    daily_handler.setLevel(file_level)
    daily_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(daily_handler)
    
    # Log setup completed
    root_logger.info("Logging system initialized")
    return root_logger

def get_logger(name):
    """Get a module-specific logger"""
    return logging.getLogger(name)

class TradeLogger:
    """Specialized logger for trade activities"""
    
    def __init__(self, log_dir="logs/trades"):
        """Initialize trade logger"""
        self.log_dir = log_dir
        
        # Create trade logs directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create logger
        self.logger = logging.getLogger("trades")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create daily log file
        daily_log_file = os.path.join(
            self.log_dir, 
            f"trades_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        # Create handler
        handler = logging.FileHandler(daily_log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def log_trade(self, symbol, action, quantity, price, value, transaction_costs=None, pnl=None, pnl_pct=None):
        """Log a trade"""
        msg = f"{action} {quantity} {symbol} @ ₹{price:.2f} (Value: ₹{value:.2f})"
        
        if transaction_costs:
            if isinstance(transaction_costs, dict):
                txn_cost = transaction_costs.get('total', 0)
            else:
                txn_cost = transaction_costs
            msg += f", Costs: ₹{txn_cost:.2f}"
        
        if pnl is not None:
            msg += f", P&L: ₹{pnl:.2f}"
            if pnl_pct is not None:
                msg += f" ({pnl_pct:.2%})"
        
        self.logger.info(msg)
    
    def log_order(self, order_id, symbol, action, quantity, price, status):
        """Log an order"""
        self.logger.info(f"Order {order_id}: {action} {quantity} {symbol} @ ₹{price:.2f} - {status}")

class PerformanceLogger:
    """Specialized logger for performance metrics"""
    
    def __init__(self, log_dir="logs/performance"):
        """Initialize performance logger"""
        self.log_dir = log_dir
        
        # Create performance logs directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create logger
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers if any
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create daily log file
        daily_log_file = os.path.join(
            self.log_dir, 
            f"performance_{datetime.now().strftime('%Y%m%d')}.log"
        )
        
        # Create handler
        handler = logging.FileHandler(daily_log_file)
        handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def log_daily_summary(self, date, open_value, close_value, pnl, pnl_pct, trade_count, win_count, loss_count):
        """Log daily performance summary"""
        self.logger.info(f"Daily Summary {date}: Value: ₹{close_value:.2f}, P&L: ₹{pnl:.2f} ({pnl_pct:.2%}), Trades: {trade_count} (W: {win_count}, L: {loss_count})")
    
    def log_metrics(self, metrics):
        """Log performance metrics"""
        if not metrics:
            self.logger.warning("No metrics to log")
            return
        
        msg = "Performance Metrics: "
        metric_parts = []
        
        for key, value in metrics.items():
            if isinstance(value, float):
                if 'pct' in key or 'rate' in key:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
                
            metric_parts.append(f"{key}={formatted_value}")
        
        msg += ", ".join(metric_parts)
        self.logger.info(msg)

def log_exception(logger, e, context=""):
    """Log an exception with traceback"""
    import traceback
    error_msg = f"{context}: {str(e)}" if context else str(e)
    logger.error(error_msg)
    logger.error(traceback.format_exc())

def measure_execution_time(logger=None):
    """Decorator to measure execution time of functions"""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed = end_time - start_time
            
            log_msg = f"Function {func.__name__} took {elapsed:.4f} seconds to execute"
            if logger:
                logger.debug(log_msg)
            else:
                logging.getLogger().debug(log_msg)
                
            return result
        return wrapper
    return decorator