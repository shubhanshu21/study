#!/usr/bin/env python3

import os
import sys
import argparse
import logging
import signal
import json
from datetime import datetime
import pytz
import time
from dotenv import load_dotenv

# Configure logging before imports to catch all logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("PaperTrading")

# Load environment variables
load_dotenv()

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import modules
from config.market_config import MarketConfig
from config.trading_config import (
    DEFAULT_SYMBOLS, DEFAULT_INITIAL_BALANCE, 
    ENABLE_DASHBOARD, DASHBOARD_PORT, REALTIME_UPDATE_INTERVAL
)
from data.fetchers.zerodha_fetcher import ZerodhaDataFetcher
from data.fetchers.nse_fetcher import NSEDataFetcher
from data.processors.indicators import IndicatorProcessor
from engine.environment import TradingEnvironment
from engine.strategy import MultiTechnicalStrategy, SwingTradingStrategy, RLBasedStrategy
from scheduler.market_scheduler import MarketScheduler

# Optional: Import dashboard if enabled
if ENABLE_DASHBOARD:
    try:
        from ui.dashboard import create_dashboard
        dashboard_available = True
    except ImportError:
        logger.warning("Dashboard modules not available, disabling dashboard")
        dashboard_available = False
else:
    dashboard_available = False

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    logger.info("Shutdown signal received")
    if 'scheduler' in globals():
        scheduler.stop()
    if dashboard_available and 'dashboard_thread' in globals():
        dashboard_thread.join(timeout=3)
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class PaperTradingSystem:
    """Main paper trading system class"""
    
    def __init__(self, symbols=None, initial_balance=DEFAULT_INITIAL_BALANCE, 
                 strategy_name="multi_technical", use_dashboard=ENABLE_DASHBOARD):
        """Initialize the paper trading system"""
        logger.info("Initializing paper trading system")
        
        # Set trading symbols
        self.symbols = symbols if symbols else DEFAULT_SYMBOLS
        logger.info(f"Trading symbols: {self.symbols}")
        
        # Initialize market config
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        
        # Initialize data fetchers
        self.data_fetchers = {}
        self._init_data_fetchers()
        
        # Initialize indicator processor
        self.indicator_processor = IndicatorProcessor()
        
        # Initialize trading environment
        self.trading_env = TradingEnvironment(initial_balance=initial_balance)
        
        # Initialize trading strategy
        self.strategy_name = strategy_name
        self.strategies = {}
        self._init_strategies()
        
        # Initialize scheduler
        self.scheduler = MarketScheduler(self.trading_env, self.data_fetchers)
        self.scheduler.add_trading_symbols(self.symbols)
        
        # Register callbacks
        self._register_callbacks()
        
        # Initialize dashboard
        self.use_dashboard = use_dashboard and dashboard_available
        self.dashboard_app = None
        self.dashboard_thread = None
        
        if self.use_dashboard:
            self._init_dashboard()
        
        logger.info("Paper trading system initialized")
    
    def _init_data_fetchers(self):
        """Initialize data fetchers"""
        try:
            # Try to initialize Zerodha fetcher
            zerodha_fetcher = ZerodhaDataFetcher()
            self.data_fetchers['zerodha'] = zerodha_fetcher
            logger.info("Zerodha data fetcher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Zerodha fetcher: {str(e)}")
        
        try:
            # Initialize NSE fetcher as backup
            nse_fetcher = NSEDataFetcher()
            self.data_fetchers['nse'] = nse_fetcher
            logger.info("NSE data fetcher initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NSE fetcher: {str(e)}")
        
        if not self.data_fetchers:
            logger.error("No data fetchers could be initialized, system will not function")
            raise RuntimeError("No data fetchers available")
    
    def _init_strategies(self):
        """Initialize trading strategies"""
        # Initialize multi-technical strategy
        self.strategies['multi_technical'] = MultiTechnicalStrategy()
        
        # Initialize swing trading strategy
        self.strategies['swing'] = SwingTradingStrategy()
        
        # Try to initialize RL-based strategy
        try:
            rl_strategy = RLBasedStrategy()
            
            # Try to load models for each symbol
            models_loaded = 0
            for symbol in self.symbols:
                model_path = f"models/enhanced_{symbol}.zip"
                if os.path.exists(model_path):
                    if rl_strategy.load_model(model_path):
                        models_loaded += 1
            
            if models_loaded > 0:
                self.strategies['rl'] = rl_strategy
                logger.info(f"RL strategy initialized with {models_loaded} models")
            else:
                logger.warning("No models found for RL strategy, not using it")
        except Exception as e:
            logger.warning(f"Failed to initialize RL strategy: {str(e)}")
        
        logger.info(f"Trading strategies initialized: {list(self.strategies.keys())}")
        
        # Set active strategy
        if self.strategy_name in self.strategies:
            self.active_strategy = self.strategies[self.strategy_name]
            logger.info(f"Active strategy set to {self.strategy_name}")
        else:
            # Fallback to multi-technical
            self.strategy_name = 'multi_technical'
            self.active_strategy = self.strategies['multi_technical']
            logger.warning(f"Requested strategy not available, using {self.strategy_name}")
    
    def _register_callbacks(self):
        """Register callbacks with the scheduler"""
        # Pre-market: Load historical data
        self.scheduler.register_pre_market_callback(self._pre_market_routine)
        
        # Market open: Initialize trading day
        self.scheduler.register_market_open_callback(self._market_open_routine)
        
        # Market update: Update data and check signals
        self.scheduler.register_market_update_callback(
            self._market_update_routine, interval_minutes=REALTIME_UPDATE_INTERVAL
        )
        
        # Market close: Close positions if needed
        self.scheduler.register_market_close_callback(self._market_close_routine)
        
        # Post-market: Save results
        self.scheduler.register_post_market_callback(self._post_market_routine)
        
        logger.info("Scheduler callbacks registered")
    
    def _init_dashboard(self):
        """Initialize dashboard"""
        try:
            if dashboard_available:
                self.dashboard_app = create_dashboard(
                    self.trading_env, self.scheduler, self.strategies, self.symbols
                )
                logger.info(f"Dashboard initialized on port {DASHBOARD_PORT}")
            else:
                logger.warning("Dashboard not available")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard: {str(e)}")
            self.use_dashboard = False
    
    def _pre_market_routine(self):
        """Pre-market routine"""
        logger.info("Running pre-market routine")
        
        # Load/update historical data for all symbols
        for symbol in self.symbols:
            try:
                # Try to use Zerodha first
                if 'zerodha' in self.data_fetchers:
                    df = self.data_fetchers['zerodha'].update_historical_data(symbol)
                    if df is not None:
                        # Calculate technical indicators
                        df = self.indicator_processor.calculate_all_indicators(df)
                        # Update trading environment
                        self.trading_env.historical_data[symbol] = df
                        logger.info(f"Historical data updated for {symbol} using Zerodha")
                        continue
                
                # Fallback to NSE
                if 'nse' in self.data_fetchers:
                    df = self.data_fetchers['nse'].fetch_stock_csv_data(symbol, period='1y')
                    if df is not None and not df.empty:
                        # Calculate technical indicators
                        df = self.indicator_processor.calculate_all_indicators(df)
                        # Update trading environment
                        self.trading_env.historical_data[symbol] = df
                        logger.info(f"Historical data updated for {symbol} using NSE")
                        continue
                
                logger.warning(f"Failed to update historical data for {symbol}")
                
            except Exception as e:
                logger.error(f"Error updating historical data for {symbol}: {str(e)}")
        
        # Get market status
        if 'nse' in self.data_fetchers:
            try:
                market_status = self.data_fetchers['nse'].fetch_market_status()
                logger.info(f"Market status: {market_status.get('status', 'unknown')}")
            except Exception as e:
                logger.error(f"Error fetching market status: {str(e)}")
    
    def _market_open_routine(self):
        """Market open routine"""
        logger.info("Running market open routine")
        
        # Check for any overnight gaps
        for symbol in self.symbols:
            if symbol in self.trading_env.historical_data and symbol in self.trading_env.current_data:
                hist_close = self.trading_env.historical_data[symbol]['Close'].iloc[-1]
                current_open = self.trading_env.current_data[symbol]['Open']
                
                gap_pct = (current_open - hist_close) / hist_close
                
                if abs(gap_pct) > 0.02:  # 2% gap
                    logger.info(f"Significant gap detected for {symbol}: {gap_pct:.2%}")
        
        # Save initial state for the day
        state_file = f"data/states/state_{datetime.now().strftime('%Y%m%d')}_open.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.trading_env.save_state(state_file)
    
    def _market_update_routine(self):
        """Market update routine"""
        current_time = datetime.now(self.timezone).strftime("%H:%M:%S")
        
        # Skip high volatility periods if configured to do so
        if self.market_config.should_avoid_high_volatility():
            logger.debug(f"Skipping update at {current_time} - high volatility period")
            return
        
        # Process each symbol
        for symbol in self.symbols:
            # Skip if no current data
            if symbol not in self.trading_env.current_data:
                continue
                
            # Skip if no historical data
            if symbol not in self.trading_env.historical_data:
                continue
            
            try:
                # Get current and historical data
                current_data = self.trading_env.current_data[symbol]
                historical_data = self.trading_env.historical_data[symbol]
                
                # Generate trading signals
                signal = self.active_strategy.generate_signals(symbol, current_data, historical_data)
                
                logger.debug(f"Signal for {symbol}: {signal['signal']} (strength: {signal['strength']:.2f})")
                
                # Process signal
                self._process_signal(symbol, signal)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        # Process pending orders and check stops
        self.trading_env.update()
    
    def _process_signal(self, symbol, signal):
        """Process a trading signal"""
        # Skip weak signals
        if signal['strength'] < 0.3:
            return
        
        # Get current position
        current_position = self.trading_env.positions.get(symbol, None)
        current_qty = current_position['quantity'] if current_position else 0
        
        # Process BUY signal
        if signal['signal'] in ['BUY', 'STRONG_BUY'] and current_qty == 0:
            # Calculate position size
            price = signal['price']
            
            # Get position size from strategy
            if hasattr(self.active_strategy, 'calculate_position_size'):
                qty, position_pct = self.active_strategy.calculate_position_size(
                    symbol, signal['strength'], price, self.trading_env.current_balance
                )
            else:
                # Default position sizing
                max_position_value = self.trading_env.current_balance * 0.1  # 10% of balance
                qty = int(max_position_value // price)
            
            # Skip if quantity is too small
            if qty < 1:
                logger.debug(f"Skipping buy for {symbol} - quantity too small: {qty}")
                return
            
            # Place order
            success, order_id = self.trading_env.place_order(
                symbol=symbol,
                order_type='BUY',
                quantity=qty,
                price=None,  # Market order
                stop_loss=signal.get('stop_loss'),
                target=signal.get('target')
            )
            
            if success:
                logger.info(f"BUY order placed for {symbol}: {qty} @ ₹{price:.2f} (ID: {order_id})")
        
        # Process SELL signal
        elif signal['signal'] in ['SELL', 'STRONG_SELL'] and current_qty > 0:
            # Place sell order for all shares
            success, order_id = self.trading_env.place_order(
                symbol=symbol,
                order_type='SELL',
                quantity=current_qty,
                price=None  # Market order
            )
            
            if success:
                logger.info(f"SELL order placed for {symbol}: {current_qty} @ ₹{signal['price']:.2f} (ID: {order_id})")
    
    def _market_close_routine(self):
        """Market close routine"""
        logger.info("Running market close routine")
        
        # Save state
        state_file = f"data/states/state_{datetime.now().strftime('%Y%m%d')}_close.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.trading_env.save_state(state_file)
        
        # Get current portfolio
        portfolio = self.trading_env.get_portfolio_summary()
        
        # Display summary
        logger.info(f"End of day summary - Portfolio Value: ₹{portfolio['portfolio_value']:.2f}")
        logger.info(f"Cash: ₹{portfolio['current_balance']:.2f} ({portfolio['cash_ratio']:.2%})")
        logger.info(f"Open Positions: {portfolio['position_count']}")
        
        # Save summary to file
        summary_file = f"data/summaries/summary_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(portfolio, f, indent=4, default=str)
        
        logger.info(f"Daily summary saved to {summary_file}")
    
    def _post_market_routine(self):
        """Post-market routine"""
        logger.info("Running post-market routine")
        
        # Calculate performance metrics
        metrics = self.trading_env.get_performance_metrics()
        
        # Display metrics
        logger.info(f"Performance: Total Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}, Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Save metrics to file
        metrics_file = f"data/metrics/metrics_{datetime.now().strftime('%Y%m%d')}.json"
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        logger.info(f"Performance metrics saved to {metrics_file}")
    
    def start(self):
        """Start the paper trading system"""
        logger.info("Starting paper trading system")
        
        # Start dashboard if enabled
        if self.use_dashboard and self.dashboard_app:
            import threading
            self.dashboard_thread = threading.Thread(
                target=self.dashboard_app.server.run,
                kwargs={'debug': False, 'port': DASHBOARD_PORT},
                daemon=True
            )
            self.dashboard_thread.start()
            logger.info(f"Dashboard started on port {DASHBOARD_PORT}")
        
        # Check if market is open
        is_market_open = self.market_config.is_market_open()
        
        if is_market_open:
            logger.info("Market is currently open, starting trading")
            # Run pre-market routine if not already done
            self._pre_market_routine()
            # Run market open routine
            self._market_open_routine()
        else:
            logger.info("Market is currently closed")
            
            # Calculate time to next market open
            seconds_to_open = self.market_config.time_to_next_market_open()
            hours_to_open = seconds_to_open / 3600
            
            logger.info(f"Next market open in {hours_to_open:.2f} hours")
        
        # Start scheduler
        self.scheduler.start()
        
        logger.info("Paper trading system started")
        
        # Update immediately to get initial data
        self.scheduler.update_now()
    
    def stop(self):
        """Stop the paper trading system"""
        logger.info("Stopping paper trading system")
        
        # Stop scheduler
        self.scheduler.stop()
        
        # Save final state
        state_file = f"data/states/state_final.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.trading_env.save_state(state_file)
        
        logger.info("Paper trading system stopped")
    
    def run_simulation(self, days=1):
        """Run a trading simulation"""
        logger.info(f"Running trading simulation for {days} days")
        
        # Run simulation using scheduler
        self.scheduler.run_simulation(days=days)
        
        # Save results
        state_file = f"data/simulation/sim_state_final.json"
        os.makedirs(os.path.dirname(state_file), exist_ok=True)
        self.trading_env.save_state(state_file)
        
        # Calculate performance metrics
        metrics = self.trading_env.get_performance_metrics()
        
        # Display metrics
        logger.info(f"Simulation Performance: Total Return: {metrics['total_return']:.2%}, Sharpe: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Win Rate: {metrics['win_rate']:.2%}, Profit Factor: {metrics['profit_factor']:.2f}")
        
        # Save metrics to file
        metrics_file = f"data/simulation/sim_metrics.json"
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4, default=str)
        
        logger.info(f"Simulation results saved to {metrics_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Indian Stock Market Paper Trading System')
    
    parser.add_argument(
        '--symbols', type=str, default=','.join(DEFAULT_SYMBOLS),
        help=f'Comma-separated list of symbols to trade (default: {",".join(DEFAULT_SYMBOLS)})'
    )
    
    parser.add_argument(
        '--balance', type=float, default=DEFAULT_INITIAL_BALANCE,
        help=f'Initial balance (default: {DEFAULT_INITIAL_BALANCE})'
    )
    
    parser.add_argument(
        '--strategy', type=str, default='multi_technical',
        choices=['multi_technical', 'swing', 'rl'],
        help='Trading strategy to use (default: multi_technical)'
    )
    
    parser.add_argument(
        '--simulate', action='store_true',
        help='Run in simulation mode'
    )
    
    parser.add_argument(
        '--days', type=int, default=1,
        help='Number of days to simulate (default: 1)'
    )
    
    parser.add_argument(
        '--dashboard', action='store_true',
        help='Enable web dashboard'
    )
    
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Debug logging enabled")
    
    # Parse symbols
    symbols = args.symbols.split(',')
    
    # Create paper trading system
    system = PaperTradingSystem(
        symbols=symbols,
        initial_balance=args.balance,
        strategy_name=args.strategy,
        use_dashboard=args.dashboard or ENABLE_DASHBOARD
    )
    
    try:
        if args.simulate:
            # Run simulation mode
            system.run_simulation(days=args.days)
        else:
            # Run live trading mode
            system.start()
            
            # Keep main thread alive
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        system.stop()
        logger.info("Paper Trading System shutdown complete")

if __name__ == "__main__":
    main()


# Live Paper Trading:
# python app.py --symbols HDFCBANK,RELIANCE,INFY --strategy multi_technical --dashboard

# Simulation Mode:
# python app.py --symbols SBIN,TCS --strategy swing --simulate --days 5