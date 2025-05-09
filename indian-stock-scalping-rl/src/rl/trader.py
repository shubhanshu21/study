"""
Live trader implementation using the trained RL model.
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from stable_baselines3 import PPO

from .environment import StockScalpingEnv
from ..data.data_fetcher import DataFetcher
from ..trading.order_manager import OrderManager
from ..constants import IndianMarketConstants, TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Trader")


class ScalpingTrader:
    """Class for live trading using the trained RL agent"""
    
    def __init__(self, symbol: str, model_path: str = TradingConfig.DEFAULT_MODEL_PATH, 
                window_size: int = 10, initial_balance: float = 100000.0, 
                broker: str = TradingConfig.DEFAULT_BROKER, is_paper_trading: bool = True):
        """Initialize the trader"""
        self.symbol = symbol
        self.model_path = model_path
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.broker = broker
        self.is_paper_trading = is_paper_trading
        
        # Initialize components
        self.data_fetcher = DataFetcher(broker=broker)
        self.order_manager = OrderManager(broker=broker, is_paper_trading=is_paper_trading)
        
        # Authenticate if needed
        if not is_paper_trading:
            if not self.data_fetcher.authenticate():
                raise ValueError("Authentication failed")
                
            if not self.order_manager.authenticate():
                raise ValueError("Order manager authentication failed")
                
        # Create environment
        self.env = StockScalpingEnv(
            symbol=symbol,
            initial_balance=initial_balance,
            window_size=window_size,
            is_training=False,
            data_fetcher=self.data_fetcher,
            order_manager=self.order_manager
        )
        
        # Load model
        try:
            self.model = PPO.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")
            
        logger.info(f"Trader initialized for {symbol} in {'paper' if is_paper_trading else 'live'} mode")
            
    def start_trading(self, max_steps: int = None, 
                     stop_loss_pct: float = TradingConfig.DEFAULT_STOP_LOSS_PCT):
        """Start live trading"""
        logger.info(f"Starting trading for {self.symbol}...")
        logger.info(f"Mode: {'Paper Trading' if self.is_paper_trading else 'Live Trading'}")
        
        # Ensure data connection
        if not self.data_fetcher.is_connected and not self.data_fetcher.authenticate():
            logger.error("Failed to connect to data source")
            return
            
        # Subscribe to symbol
        self.data_fetcher.subscribe([self.symbol])
        
        # Wait for market data
        logger.info("Waiting for market data...")
        time.sleep(5)
        
        # Check if market is open
        if not self.data_fetcher.is_market_open():
            logger.info("Market is closed. Trading session will start when market opens.")
            
            # Wait for market to open
            while not self.data_fetcher.is_market_open():
                logger.info("Waiting for market to open...")
                time.sleep(60)  # Check every minute
                
        logger.info("Market is open. Starting trading session.")
        
        # Reset environment
        obs = self.env.reset()
        done = False
        step = 0
        
        # Trading loop
        while not done:
            # Get current time and price
            current_time = datetime.now().time()
            current_price = self.env.current_price()
            
            # Check trading hours
            if (current_time < IndianMarketConstants.MARKET_OPEN_TIME or 
                current_time > IndianMarketConstants.MARKET_CLOSE_TIME):
                logger.info("Market closed. Ending trading session.")
                break
                
            # Check max steps
            if max_steps and step >= max_steps:
                logger.info(f"Reached maximum steps ({max_steps}). Ending trading session.")
                break
                
            # Get portfolio info
            portfolio = self.order_manager.get_positions()
            balance = portfolio.get('balance', 0)
            
            if isinstance(portfolio.get('positions'), dict):
                position_info = portfolio.get('positions', {}).get(self.symbol, {})
                position = position_info.get('qty', 0)
                entry_price = position_info.get('avg_price', 0)
                unrealized_pnl = position_info.get('pnl', 0)
            else:
                position = self.env.position
                entry_price = self.env.entry_price
                unrealized_pnl = 0
                
            # Check stop loss
            if position != 0 and entry_price > 0:
                if position > 0:  # Long position
                    loss_pct = (entry_price - current_price) / entry_price
                    if loss_pct > stop_loss_pct:
                        logger.warning(f"Stop loss triggered: Loss {loss_pct*100:.2f}% exceeds threshold {stop_loss_pct*100:.2f}%")
                        # Force sell action
                        action = 2  # Sell
                        stop_loss_triggered = True
                    else:
                        # Normal prediction
                        action, _ = self.model.predict(obs, deterministic=True)
                        stop_loss_triggered = False
                elif position < 0:  # Short position
                    loss_pct = (current_price - entry_price) / entry_price
                    if loss_pct > stop_loss_pct:
                        logger.warning(f"Stop loss triggered: Loss {loss_pct*100:.2f}% exceeds threshold {stop_loss_pct*100:.2f}%")
                        # Force buy action
                        action = 1  # Buy
                        stop_loss_triggered = True
                    else:
                        # Normal prediction
                        action, _ = self.model.predict(obs, deterministic=True)
                        stop_loss_triggered = False
            else:
                # No position, normal prediction
                action, _ = self.model.predict(obs, deterministic=True)
                stop_loss_triggered = False
                
            # Take action
            obs, reward, done, info = self.env.step(action)
            
            # Print info
            action_name = "Hold" if action == 0 else "Buy" if action == 1 else "Sell"
            logger.info(f"Step {step} | Time: {datetime.now().strftime('%H:%M:%S')}")
            logger.info(f"Action: {action_name}" + (" (Stop Loss)" if stop_loss_triggered else ""))
            logger.info(f"Price: ₹{current_price:.2f}")
            logger.info(f"Position: {position}")
            logger.info(f"Balance: ₹{balance:.2f}")
            
            if position != 0:
                logger.info(f"Entry Price: ₹{entry_price:.2f}")
                logger.info(f"Unrealized P&L: ₹{unrealized_pnl:.2f} ({unrealized_pnl/entry_price/abs(position)*100:.2f}%)")
                
            logger.info(f"Total P&L: ₹{self.env.total_pnl:.2f}")
            logger.info("-" * 50)
            
            # Update position info from order manager if paper trading
            if self.is_paper_trading:
                self.order_manager.update_paper_positions()
                
            # Increment step
            step += 1
            
            # Wait for next candle (5 minutes)
            seconds_to_next_candle = self._seconds_to_next_candle(5)
            if seconds_to_next_candle > 0:
                time.sleep(seconds_to_next_candle)
                
        # Trading session ended
        logger.info("\nTrading session summary:")
        logger.info(f"Total steps: {step}")
        logger.info(f"Final balance: ₹{self.order_manager.get_positions().get('balance', 0):.2f}")
        logger.info(f"Total P&L: ₹{self.env.total_pnl:.2f} ({self.env.total_pnl/self.initial_balance*100:.2f}%)")
        logger.info(f"Total trades: {self.env.total_trades}")
        if self.env.total_trades > 0:
            logger.info(f"Success rate: {self.env.successful_trades/self.env.total_trades*100:.2f}%")
        logger.info(f"Max drawdown: {self.env.max_drawdown*100:.2f}%")
            
    def stop_trading(self):
        """Stop trading and clean up resources"""
        # Close all positions
        logger.info("Closing all positions...")
        
        if self.env.position > 0:
            # Close long position
            self.env.step(2)  # Sell
        elif self.env.position < 0:
            # Close short position
            self.env.step(1)  # Buy
            
        # Stop data connection
        self.data_fetcher.stop_websocket()
        
        # Close environment
        self.env.close()
        
        logger.info("Trading session stopped")
        
    def _seconds_to_next_candle(self, minutes: int = 5) -> int:
        """Calculate seconds to next candle"""
        now = datetime.now()
        current_minute = now.minute
        current_second = now.second
        current_microsecond = now.microsecond
        
        # Calculate target minute
        target_minute = (current_minute // minutes + 1) * minutes
        
        # Calculate seconds to next candle
        seconds_to_next = ((target_minute - current_minute) * 60) - current_second - (current_microsecond / 1000000)
        
        return max(0, seconds_to_next)