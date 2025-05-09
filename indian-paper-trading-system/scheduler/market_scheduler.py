import time
import logging
import threading
import schedule
import datetime
import pytz
from config.market_config import MarketConfig

logger = logging.getLogger(__name__)

class MarketScheduler:
    """Scheduler for running trading operations during market hours"""
    
    def __init__(self, trading_env, data_fetchers):
        """Initialize MarketScheduler
        
        Args:
            trading_env: Trading environment instance
            data_fetchers: Dictionary of data fetcher instances
        """
        self.trading_env = trading_env
        self.data_fetchers = data_fetchers
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        
        # Scheduling variables
        self.scheduler_thread = None
        self.stop_scheduler = False
        self.is_running = False
        
        # Callbacks
        self.pre_market_callbacks = []
        self.market_open_callbacks = []
        self.market_update_callbacks = []
        self.market_close_callbacks = []
        self.post_market_callbacks = []
        
        # Trading symbols
        self.trading_symbols = []
        
        logger.info("MarketScheduler initialized")
    
    def add_trading_symbols(self, symbols):
        """Add symbols to trade"""
        self.trading_symbols = symbols
        logger.info(f"Trading symbols set: {symbols}")
    
    def register_pre_market_callback(self, callback):
        """Register callback to run before market opens"""
        self.pre_market_callbacks.append(callback)
    
    def register_market_open_callback(self, callback):
        """Register callback to run when market opens"""
        self.market_open_callbacks.append(callback)
    
    def register_market_update_callback(self, callback, interval_minutes=1):
        """Register callback to run periodically during market hours
        
        Args:
            callback: Function to call
            interval_minutes: Interval in minutes
        """
        self.market_update_callbacks.append((callback, interval_minutes))
    
    def register_market_close_callback(self, callback):
        """Register callback to run when market closes"""
        self.market_close_callbacks.append(callback)
    
    def register_post_market_callback(self, callback):
        """Register callback to run after market closes"""
        self.post_market_callbacks.append(callback)
    
    def _pre_market_routine(self):
        """Run pre-market routine"""
        logger.info("Running pre-market routine")
        
        # Update market data
        self._update_market_data()
        
        # Run registered callbacks
        for callback in self.pre_market_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in pre-market callback: {str(e)}")
    
    def _market_open_routine(self):
        """Run market open routine"""
        logger.info("Running market open routine")
        
        # Update market data
        self._update_market_data()
        
        # Run registered callbacks
        for callback in self.market_open_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in market open callback: {str(e)}")
    
    def _market_update_routine(self):
        """Run market update routine"""
        current_time = datetime.datetime.now(self.timezone).time()
        logger.debug(f"Running market update routine at {current_time}")
        
        # Check if market is still open
        if not self.market_config.is_market_open():
            logger.info("Market is now closed, stopping updates")
            return
        
        # Update market data
        self._update_market_data()
        
        # Update trading environment
        self.trading_env.update()
        
        # Run registered callbacks
        for callback, interval_minutes in self.market_update_callbacks:
            current_minute = datetime.datetime.now().minute
            
            # Run if current minute is divisible by interval
            if current_minute % interval_minutes == 0:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in market update callback: {str(e)}")
    
    def _market_close_routine(self):
        """Run market close routine"""
        logger.info("Running market close routine")
        
        # Final update for the day
        self._update_market_data()
        
        # Run registered callbacks
        for callback in self.market_close_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in market close callback: {str(e)}")
    
    def _post_market_routine(self):
        """Run post-market routine"""
        logger.info("Running post-market routine")
        
        # Run registered callbacks
        for callback in self.post_market_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error in post-market callback: {str(e)}")
    
    def _update_market_data(self):
        """Update market data for all trading symbols"""
        if not self.trading_symbols:
            logger.warning("No trading symbols defined, skipping market data update")
            return
        
        # Use Zerodha fetcher if available
        if 'zerodha' in self.data_fetchers:
            try:
                fetcher = self.data_fetchers['zerodha']
                
                # Fetch live data
                live_data = fetcher.fetch_live_data(self.trading_symbols)
                
                # Update trading environment
                for symbol, data in live_data.items():
                    self.trading_env.update_market_data(symbol, data)
                
                logger.debug(f"Market data updated for {len(live_data)} symbols using Zerodha")
                return
            except Exception as e:
                logger.error(f"Error updating data from Zerodha: {str(e)}")
        
        # Fallback to NSE fetcher
        if 'nse' in self.data_fetchers:
            try:
                fetcher = self.data_fetchers['nse']
                
                # Fetch data for each symbol
                for symbol in self.trading_symbols:
                    data = fetcher.fetch_equity_quote(symbol)
                    
                    # Format data for trading environment
                    if data:
                        formatted_data = {
                            'Date': datetime.datetime.now(self.timezone).date(),
                            'Time': datetime.datetime.now(self.timezone).time(),
                            'Open': data.get('open', 0),
                            'High': data.get('dayHigh', 0),
                            'Low': data.get('dayLow', 0),
                            'Close': data.get('lastPrice', 0),
                            'Volume': data.get('totalTradedVolume', 0),
                            'Symbol': symbol
                        }
                        
                        self.trading_env.update_market_data(symbol, formatted_data)
                
                logger.debug(f"Market data updated for {len(self.trading_symbols)} symbols using NSE")
                return
            except Exception as e:
                logger.error(f"Error updating data from NSE: {str(e)}")
        
        logger.error("No data fetchers available to update market data")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while not self.stop_scheduler:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error in scheduler loop: {str(e)}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        logger.info("Starting market scheduler")
        
        # Clear existing schedules
        schedule.clear()
        
        # Schedule pre-market routine
        pre_market_time = f"{self.market_config.pre_market_start.hour:02d}:{self.market_config.pre_market_start.minute:02d}"
        schedule.every().day.at(pre_market_time).do(self._pre_market_routine)
        logger.info(f"Scheduled pre-market routine at {pre_market_time}")
        
        # Schedule market open routine
        market_open_time = f"{self.market_config.market_open.hour:02d}:{self.market_config.market_open.minute:02d}"
        schedule.every().day.at(market_open_time).do(self._market_open_routine)
        logger.info(f"Scheduled market open routine at {market_open_time}")
        
        # Schedule market update routine (every minute during market hours)
        schedule.every(1).minutes.do(self._market_update_routine)
        logger.info("Scheduled market update routine every minute")
        
        # Schedule market close routine
        market_close_time = f"{self.market_config.market_close.hour:02d}:{self.market_config.market_close.minute:02d}"
        schedule.every().day.at(market_close_time).do(self._market_close_routine)
        logger.info(f"Scheduled market close routine at {market_close_time}")
        
        # Schedule post-market routine
        post_market_time = f"{self.market_config.post_market_end.hour:02d}:{self.market_config.post_market_end.minute:02d}"
        schedule.every().day.at(post_market_time).do(self._post_market_routine)
        logger.info(f"Scheduled post-market routine at {post_market_time}")
        
        # Reset stop flag
        self.stop_scheduler = False
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        self.is_running = True
        logger.info("Market scheduler started")
    
    def stop(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        logger.info("Stopping market scheduler")
        
        # Set stop flag
        self.stop_scheduler = True
        
        # Wait for thread to finish
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        # Clear all schedules
        schedule.clear()
        
        self.is_running = False
        logger.info("Market scheduler stopped")
    
    def update_now(self):
        """Manually trigger an update now"""
        logger.info("Manual update triggered")
        
        # Check if market is open
        if not self.market_config.is_market_open():
            logger.warning("Market is closed, update may not reflect real-time data")
        
        # Run update routine
        self._update_market_data()
        self.trading_env.update()
    
    def run_simulation(self, days=1, update_interval_seconds=5):
        """Run a simulated trading session for testing
        
        Args:
            days: Number of days to simulate
            update_interval_seconds: Interval between updates in seconds
        """
        logger.info(f"Starting trading simulation for {days} days")
        
        try:
            for day in range(days):
                # Pre-market
                logger.info(f"Day {day+1}: Pre-market")
                self._pre_market_routine()
                time.sleep(1)
                
                # Market open
                logger.info(f"Day {day+1}: Market open")
                self._market_open_routine()
                time.sleep(1)
                
                # Market hours updates
                logger.info(f"Day {day+1}: Market hours")
                for i in range(int(6 * 60 / (update_interval_seconds / 60))):  # 6 hours of market
                    self._market_update_routine()
                    time.sleep(update_interval_seconds)
                
                # Market close
                logger.info(f"Day {day+1}: Market close")
                self._market_close_routine()
                time.sleep(1)
                
                # Post-market
                logger.info(f"Day {day+1}: Post-market")
                self._post_market_routine()
                time.sleep(1)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        except Exception as e:
            logger.error(f"Error in simulation: {str(e)}")
        
        logger.info("Simulation complete")