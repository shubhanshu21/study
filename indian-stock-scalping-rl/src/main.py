"""
Main entry point for the RL trading system.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
from .data.data_fetcher import DataFetcher
from .rl.trainer import ScalpingTrainer
from .rl.trader import ScalpingTrader
from .constants import TrainingConfig, TradingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Main")


def main():
    """Main function to run the application"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="RL-based Stock Scalping Trader for Indian Markets")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "trade"], default="train",
                       help="Operation mode: train, backtest, or trade")
    parser.add_argument("--symbol", type=str, default=os.getenv('DEFAULT_SYMBOL', 'RELIANCE'),
                       help="Stock symbol to trade")
    parser.add_argument("--broker", type=str, choices=["zerodha", "upstox"], default=TradingConfig.DEFAULT_BROKER,
                       help="Broker to use")
    parser.add_argument("--paper", action="store_true",
                       help="Use paper trading mode")
    parser.add_argument("--model", type=str, default=TradingConfig.DEFAULT_MODEL_PATH,
                       help="Path to save/load model")
    parser.add_argument("--balance", type=float, default=TrainingConfig.INITIAL_BALANCE,
                       help="Initial balance for trading/backtesting")
    parser.add_argument("--window", type=int, default=TrainingConfig.WINDOW_SIZE,
                       help="Observation window size")
    parser.add_argument("--steps", type=int, default=TrainingConfig.DEFAULT_TIMESTEPS,
                       help="Number of steps for training")
    parser.add_argument("--envs", type=int, default=TrainingConfig.N_ENVS,
                       help="Number of parallel environments for training")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    logger.info(f"Running in {args.mode} mode for {args.symbol}")
    
    # Initialize data fetcher for historical data
    data_fetcher = DataFetcher(broker=args.broker)
    authenticated = data_fetcher.authenticate()
    
    if not authenticated:
        logger.warning("Authentication failed. Using minimal functionality.")
    
    # Get historical data
    logger.info(f"Fetching historical data for {args.symbol}...")
    # historical_data = data_fetcher.fetch_historical_data(
    #     args.symbol,
    #     interval="5minute",
    #     from_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
    #     to_date=datetime.now().strftime('%Y-%m-%d')
    # )

    # Load CSV into DataFrame
    df = pd.read_csv(f"datasets/{args.symbol}.csv")

    # Columns to keep
    columns_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']

    # Filter DataFrame to keep only those columns (ignore missing)
    df = df[[col for col in columns_to_keep if col in df.columns]]

    # Ensure 'Date' is a column, not index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)

    historical_data = df
    
    if historical_data.empty:
        logger.error(f"Failed to fetch historical data for {args.symbol}")
        return
        
    logger.info(f"Got {len(historical_data)} candles of historical data")
    
    # Execute based on mode
    if args.mode == "train":
        # Training mode
        logger.info(f"Training RL agent for {args.symbol}...")
        
        # Create trainer
        trainer = ScalpingTrainer(
            symbol=args.symbol,
            data=historical_data,
            window_size=args.window,
            initial_balance=args.balance,
            n_envs=args.envs,
            data_fetcher=data_fetcher
        )
        
        # Train the model
        model = trainer.train(total_timesteps=args.steps, log_interval=TrainingConfig.LOG_INTERVAL)
        
        # Save the model
        trainer.save_model(path=args.model)
        
        # Run backtest
        logger.info("\nBacktesting the trained model...")
        backtest_results = trainer.backtest()
        
    elif args.mode == "backtest":
        # Backtest mode
        logger.info(f"Backtesting model on {args.symbol}...")
        
        # Create trainer and load model
        trainer = ScalpingTrainer(
            symbol=args.symbol,
            data=historical_data,
            window_size=args.window,
            initial_balance=args.balance,
            data_fetcher=data_fetcher
        )
        
        trainer.load_model(path=args.model)
        
        # Run backtest
        backtest_results = trainer.backtest()
        
    elif args.mode == "trade":
        # Trading mode
        logger.info(f"Starting trading session for {args.symbol}...")
        
        # Create trader
        trader = ScalpingTrader(
            symbol=args.symbol,
            model_path=args.model,
            window_size=args.window,
            initial_balance=args.balance,
            broker=args.broker,
            is_paper_trading=args.paper
        )
        
        try:
            # Start trading
            trader.start_trading(stop_loss_pct=TradingConfig.DEFAULT_STOP_LOSS_PCT)
        except KeyboardInterrupt:
            logger.info("\nTrading interrupted by user.")
        finally:
            # Clean up
            trader.stop_trading()
            
    # Close data fetcher
    data_fetcher.stop_websocket()
    logger.info("Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Error: {e}")