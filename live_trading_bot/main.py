#!/usr/bin/env python
"""
Main entry point for the RL Trading Bot.
Handles startup, scheduling, and shutdown of the trading service.
"""
import os
import sys
import argparse
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.app_config import AppConfig
from config.logging_config import setup_logging
from services.scheduler import TradingScheduler
from services.notifications import NotificationService
from trading.zerodha_broker import ZerodhaBroker
from trading.strategy import RLTradingStrategy
from utils.market_regime import MarketRegimeDetector
from data.data_loader import ZerodhaDataLoader
from models.model_factory import ModelFactory

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='RL Trading Bot')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                      default='paper', help='Trading mode')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      default='INFO', help='Logging level')
    parser.add_argument('--config', type=str, default='config/app_config.py',
                      help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main function to start the trading service"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = AppConfig(args.config)
    
    # Setup logging
    logger = setup_logging(level=args.log_level, 
                          log_dir=config.LOG_DIR,
                          service_name="rl_trading_bot")
    
    logger.info(f"Starting RL Trading Bot in {args.mode} mode")
    
    try:
        # Initialize services
        notification_service = NotificationService(config)
        
        # Initialize broker connection
        broker = ZerodhaBroker(
            api_key=config.ZERODHA_API_KEY,
            api_secret=config.ZERODHA_API_SECRET,
            totp_secret=config.ZERODHA_TOTP_SECRET,  # For automated login
            redirect_url=config.ZERODHA_REDIRECT_URL,
            mode=args.mode
        )
        
        # Initialize broker connection
        if not broker.authenticate():
            logger.error("Failed to authenticate with Zerodha")
            notification_service.send_alert("Authentication Failed", 
                                          "Failed to authenticate with Zerodha. Trading service not started.")
            return 1
            
        logger.info("Successfully authenticated with Zerodha")
        
        # Initialize data loader
        data_loader = ZerodhaDataLoader(broker, config.SYMBOLS)
        
        # Initialize market regime detector
        regime_detector = MarketRegimeDetector(lookback=config.REGIME_LOOKBACK)
        
        # Load models for each symbol
        model_factory = ModelFactory(config.MODEL_DIR)
        trading_strategies = {}
        
        for symbol in config.SYMBOLS:
            model = model_factory.load_model(symbol, model_type="ppo")
            trading_strategies[symbol] = RLTradingStrategy(
                symbol=symbol,
                model=model,
                data_loader=data_loader,
                regime_detector=regime_detector,
                broker=broker,
                config=config
            )
            logger.info(f"Loaded model for {symbol}")
        
        # Initialize scheduler
        scheduler = TradingScheduler(
            trading_strategies=trading_strategies,
            broker=broker,
            config=config,
            notification_service=notification_service
        )
        
        # Start the scheduler
        scheduler.start()
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            logger.info("Received shutdown signal")
            
        # Graceful shutdown
        scheduler.stop()
        broker.logout()
        logger.info("Trading service stopped")
        return 0
        
    except Exception as e:
        logger.exception(f"Fatal error: {str(e)}")
        notification_service.send_alert("Service Error", 
                                       f"Trading service encountered an error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())