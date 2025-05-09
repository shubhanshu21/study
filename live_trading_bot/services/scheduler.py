"""
Scheduler service for managing trading tasks.
"""
import time
import logging
import threading
import schedule
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TradingScheduler:
    """
    Scheduler service for managing trading tasks.
    Handles scheduling of data updates, trading decisions, and position checks.
    """
    
    def __init__(self, trading_strategies, broker, config, notification_service):
        """
        Initialize the scheduler service
        
        Args:
            trading_strategies (dict): Dictionary of symbol -> strategy instances
            broker: Broker instance
            config: Application configuration
            notification_service: Notification service instance
        """
        self.trading_strategies = trading_strategies
        self.broker = broker
        self.config = config
        self.notification_service = notification_service
        self.running = False
        self.scheduler_thread = None
        
        # Initialize schedule
        self._init_schedule()
        
        logger.info(f"Initialized trading scheduler with {len(trading_strategies)} symbols")
    
    def _init_schedule(self):
        """Initialize the scheduler with tasks"""
        # Clear any existing schedules
        schedule.clear()
        
        # Add regular data update task (every N seconds)
        schedule.every(self.config.DATA_UPDATE_INTERVAL).seconds.do(self._run_data_updates)
        
        # Add trading decision task (every N seconds)
        schedule.every(self.config.TRADING_DECISION_INTERVAL).seconds.do(self._run_trading_decisions)
        
        # Add position check task (every N seconds)
        schedule.every(self.config.POSITION_CHECK_INTERVAL).seconds.do(self._run_position_checks)
        
        # Add daily status update task (every minute)
        schedule.every(1).minutes.do(self._run_daily_status_updates)
        
        # Add daily summary task at market close
        schedule.every().day.at(f"{self.config.MARKET_CLOSE_HOUR}:{self.config.MARKET_CLOSE_MINUTE}").do(
            self._run_daily_summary)
        
        logger.info("Trading schedule initialized")
    
    def _run_data_updates(self):
        """Run data updates for all strategies"""
        if not self._is_market_open():
            return
            
        logger.debug("Running data updates")
        
        for symbol, strategy in self.trading_strategies.items():
            try:
                strategy.update_data()
            except Exception as e:
                logger.exception(f"Error updating data for {symbol}: {str(e)}")
    
    def _run_trading_decisions(self):
        """Run trading decisions for all strategies"""
        if not self._is_market_open():
            return
            
        logger.debug("Running trading decisions")
        
        for symbol, strategy in self.trading_strategies.items():
            try:
                # Skip if too close to market open or close
                now = datetime.now(self.config.TIMEZONE)
                market_open = now.replace(
                    hour=self.config.MARKET_OPEN_HOUR,
                    minute=self.config.MARKET_OPEN_MINUTE,
                    second=0,
                    microsecond=0
                )
                
                market_close = now.replace(
                    hour=self.config.MARKET_CLOSE_HOUR,
                    minute=self.config.MARKET_CLOSE_MINUTE,
                    second=0,
                    microsecond=0
                )
                
                minutes_since_open = (now - market_open).total_seconds() / 60
                minutes_to_close = (market_close - now).total_seconds() / 60
                
                if minutes_since_open < self.config.AVOID_FIRST_MINUTES:
                    logger.debug(f"Skipping decision for {symbol} - too close to market open")
                    continue
                    
                if 0 <= minutes_to_close <= self.config.AVOID_LAST_MINUTES:
                    logger.debug(f"Skipping decision for {symbol} - too close to market close")
                    continue
                
                strategy.make_trading_decision()
            except Exception as e:
                logger.exception(f"Error making trading decision for {symbol}: {str(e)}")
    
    def _run_position_checks(self):
        """Run position checks for all strategies"""
        if not self._is_market_open():
            return
            
        logger.debug("Running position checks")
        
        for symbol, strategy in self.trading_strategies.items():
            try:
                # Get current price
                current_price = strategy.broker.get_ltp(symbol)
                
                if current_price and strategy.position_type:
                    # Update trailing stop
                    strategy._update_trailing_stop(current_price)
                    
                    # Check exit conditions
                    exit_triggered, exit_type = strategy._check_exit_conditions(current_price)
                    
                    if exit_triggered:
                        logger.info(f"Exit triggered for {symbol}: {exit_type}")
                        strategy._execute_sell(exit_type)
            except Exception as e:
                logger.exception(f"Error checking position for {symbol}: {str(e)}")
    
    def _run_daily_status_updates(self):
        """Run daily status updates for all strategies"""
        logger.debug("Running daily status updates")
        
        for symbol, strategy in self.trading_strategies.items():
            try:
                strategy.update_daily_status()
            except Exception as e:
                logger.exception(f"Error updating daily status for {symbol}: {str(e)}")
    
    def _run_daily_summary(self):
        """Run daily summary at end of day"""
        logger.info("Generating daily trading summary")
        
        # Get overall account performance
        funds = self.broker.get_funds()
        
        if not funds:
            logger.error("Could not get account funds for daily summary")
            return
            
        total_balance = funds.get('balance', 0)
        
        # Get positions
        positions = self.broker.get_positions()
        position_value = sum(p.get('quantity', 0) * p.get('last_price', 0) 
                           for p in positions.values() if p.get('quantity', 0) > 0)
        
        # Calculate total account value
        total_value = total_balance + position_value
        
        # Get strategy metrics
        strategy_metrics = {}
        all_trades_today = 0
        total_profit_today = 0
        total_loss_today = 0
        
        for symbol, strategy in self.trading_strategies.items():
            metrics = strategy.get_performance_metrics()
            strategy_metrics[symbol] = metrics
            
            all_trades_today += metrics.get('daily_trades', 0)
            total_profit_today += metrics.get('daily_profit', 0)
            total_loss_today += metrics.get('daily_loss', 0)
        
        # Generate summary message
        net_profit_today = total_profit_today - total_loss_today
        summary = (
            f"Daily Trading Summary\n"
            f"Date: {datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d')}\n"
            f"Total Account Value: ₹{total_value:.2f}\n"
            f"Cash Balance: ₹{total_balance:.2f}\n"
            f"Position Value: ₹{position_value:.2f}\n"
            f"Total Trades Today: {all_trades_today}\n"
            f"Net Profit Today: ₹{net_profit_today:.2f}\n"
            f"\nSymbol Performance:\n"
        )
        
        for symbol, metrics in strategy_metrics.items():
            summary += (
                f"{symbol}:\n"
                f"  Trades: {metrics.get('daily_trades', 0)}\n"
                f"  Profit: ₹{metrics.get('daily_profit', 0):.2f}\n"
                f"  Loss: ₹{metrics.get('daily_loss', 0):.2f}\n"
                f"  Net: ₹{metrics.get('daily_profit', 0) - metrics.get('daily_loss', 0):.2f}\n"
                f"  Position: {metrics.get('position_type', 'None')}\n"
                f"  Position Value: ₹{metrics.get('position_value', 0):.2f}\n"
                f"  Market Regime: {metrics.get('current_market_regime', 'unknown')}\n"
            )
        
        logger.info(f"Daily Summary:\n{summary}")
        
        # Send notification
        self.notification_service.send_alert("Daily Trading Summary", summary)
        
        return True
    
    def _is_market_open(self):
        """Check if market is currently open"""
        try:
            now = datetime.now(self.config.TIMEZONE)
            
            # If weekend, market is closed
            if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                return False
                
            # Check market hours
            market_open = now.replace(
                hour=self.config.MARKET_OPEN_HOUR, 
                minute=self.config.MARKET_OPEN_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            market_close = now.replace(
                hour=self.config.MARKET_CLOSE_HOUR, 
                minute=self.config.MARKET_CLOSE_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.exception(f"Error checking if market is open: {str(e)}")
            return False
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        logger.info("Starting scheduler loop")
        
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logger.exception(f"Error in scheduler loop: {str(e)}")
                time.sleep(5)  # Sleep longer on error
    
    def start(self):
        """Start the scheduler"""
        if self.running:
            logger.warning("Scheduler already running")
            return
            
        logger.info("Starting scheduler")
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        # Send notification
        self.notification_service.send_alert(
            "Trading Bot Started", 
            f"Trading bot started at {datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}"
        )
    
    def stop(self):
        """Stop the scheduler"""
        if not self.running:
            logger.warning("Scheduler not running")
            return
            
        logger.info("Stopping scheduler")
        self.running = False
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
            
        # Send notification
        self.notification_service.send_alert(
            "Trading Bot Stopped", 
            f"Trading bot stopped at {datetime.now(self.config.TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}"
        )