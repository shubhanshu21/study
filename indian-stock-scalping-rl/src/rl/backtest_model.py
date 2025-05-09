#!/usr/bin/env python
"""
Backtesting script for the improved RL scalping model.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from stable_baselines3 import PPO

from src.rl.environment_v2 import StockScalpingEnv
from src.data.data_fetcher import DataFetcher
from src.constants import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backtest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Backtest")

def plot_backtest_results(equity_file, price_file, trades_file, output_dir="results"):
    """Create visualizations of backtest results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    equity_df = pd.read_csv(equity_file)
    equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
    
    # Load price data if available
    price_data = None
    if os.path.exists(price_file):
        price_data = pd.read_csv(price_file)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    
    # Load trades if available
    trades_data = None
    if os.path.exists(trades_file):
        trades_data = pd.read_csv(trades_file)
        if 'timestamp' in trades_data.columns:
            trades_data['timestamp'] = pd.to_datetime(trades_data['timestamp'])
    
    # 1. Equity Curve
    plt.figure(figsize=(15, 7))
    plt.plot(equity_df['timestamp'], equity_df['portfolio_value'], label='Portfolio Value')
    
    if price_data is not None:
        # Normalize price to start at initial portfolio value for comparison
        initial_value = equity_df['portfolio_value'].iloc[0]
        price_ratio = initial_value / price_data['close'].iloc[0]
        plt.plot(price_data['timestamp'], price_data['close'] * price_ratio, 
                 label='Asset Price (Normalized)', alpha=0.7, linestyle='--')
    
    # Mark buy/sell points if trades data is available
    if trades_data is not None:
        buy_trades = trades_data[trades_data['action'] == 'buy']
        sell_trades = trades_data[trades_data['action'] == 'close_long']
        
        # Plot buy points
        for _, trade in buy_trades.iterrows():
            plt.scatter(trade['timestamp'], trade['price'], 
                        color='green', marker='^', s=100, alpha=0.7)
        
        # Plot sell points
        for _, trade in sell_trades.iterrows():
            plt.scatter(trade['timestamp'], trade['price'], 
                        color='red', marker='v', s=100, alpha=0.7)
    
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{output_dir}/equity_curve.png")
    
    # 2. Drawdown Chart
    plt.figure(figsize=(15, 7))
    
    # Calculate drawdown
    rolling_max = equity_df['portfolio_value'].cummax()
    drawdown = (rolling_max - equity_df['portfolio_value']) / rolling_max
    
    plt.plot(equity_df['timestamp'], drawdown * 100)
    plt.title('Portfolio Drawdown')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (%)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/drawdown.png")
    
    # 3. Daily Returns
    if len(equity_df) > 1:
        # Group by date and get closing value
        equity_df['date'] = equity_df['timestamp'].dt.date
        daily_data = equity_df.groupby('date')['portfolio_value'].last().reset_index()
        
        # Calculate daily returns
        daily_data['prev_value'] = daily_data['portfolio_value'].shift(1)
        daily_data['return'] = (daily_data['portfolio_value'] - daily_data['prev_value']) / daily_data['prev_value']
        daily_data = daily_data.dropna()
        
        plt.figure(figsize=(15, 7))
        plt.bar(daily_data['date'], daily_data['return'] * 100)
        plt.title('Daily Returns (%)')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.savefig(f"{output_dir}/daily_returns.png")
        
        # 4. Return Distribution
        plt.figure(figsize=(15, 7))
        plt.hist(daily_data['return'] * 100, bins=30, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('Return Distribution')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig(f"{output_dir}/return_distribution.png")
    
    # 5. Trade Analysis
    if trades_data is not None and len(trades_data) > 0:
        # Filter for completed trades with PnL
        pnl_trades = trades_data[trades_data['action'].isin(['close_long', 'close_short'])]
        
        if not pnl_trades.empty:
            plt.figure(figsize=(15, 7))
            
            # Sort trades by PnL
            pnl_trades = pnl_trades.sort_values('pnl')
            
            # Create bar chart of PnL
            plt.bar(range(len(pnl_trades)), pnl_trades['pnl'], color=['red' if x < 0 else 'green' for x in pnl_trades['pnl']])
            plt.title('Trade PnL Distribution')
            plt.xlabel('Trade #')
            plt.ylabel('PnL')
            plt.grid(True)
            plt.savefig(f"{output_dir}/trade_pnl.png")
            
            # 6. Win/Loss Pie Chart
            plt.figure(figsize=(10, 10))
            win_count = (pnl_trades['pnl'] > 0).sum()
            loss_count = (pnl_trades['pnl'] <= 0).sum()
            labels = [f'Wins ({win_count})', f'Losses ({loss_count})']
            sizes = [win_count, loss_count]
            colors = ['green', 'red']
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Win/Loss Ratio')
            plt.savefig(f"{output_dir}/win_loss_ratio.png")
    
    logger.info(f"Backtest visualization plots saved to {output_dir}/")
    
def calculate_performance_metrics(equity_curve, trades):
    """Calculate comprehensive performance metrics"""
    if len(equity_curve) < 2:
        return {"error": "Not enough data points"}
    
    initial_value = equity_curve['portfolio_value'].iloc[0]
    final_value = equity_curve['portfolio_value'].iloc[-1]
    
    # Basic metrics
    metrics = {
        'initial_equity': initial_value,
        'final_equity': final_value,
        'absolute_return': final_value - initial_value,
        'pct_return': ((final_value / initial_value) - 1) * 100,
        'trade_count': len(trades) if trades is not None else 0,
    }
    
    # Daily metrics
    equity_curve['date'] = pd.to_datetime(equity_curve['timestamp']).dt.date
    daily_values = equity_curve.groupby('date')['portfolio_value'].last()
    
    if len(daily_values) > 1:
        daily_returns = daily_values.pct_change().dropna()
        
        metrics.update({
            'trading_days': len(daily_values),
            'profitable_days': (daily_returns > 0).sum(),
            'losing_days': (daily_returns <= 0).sum(),
            'avg_daily_return': daily_returns.mean() * 100,
            'daily_return_std': daily_returns.std() * 100,
            'best_day_return': daily_returns.max() * 100,
            'worst_day_return': daily_returns.min() * 100,
        })
        
        # Annualized metrics (assuming 252 trading days)
        trading_days = len(daily_returns)
        if trading_days > 5:  # Need at least a week of data
            ann_factor = 252 / trading_days
            ann_return = ((1 + daily_returns.mean()) ** 252) - 1
            ann_volatility = daily_returns.std() * np.sqrt(252)
            
            metrics.update({
                'annualized_return': ann_return * 100,
                'annualized_volatility': ann_volatility * 100,
                'sharpe_ratio': (ann_return - 0.05) / ann_volatility if ann_volatility > 0 else 0,
                'calmar_ratio': abs(ann_return / metrics.get('max_drawdown', 1)) if metrics.get('max_drawdown', 0) > 0 else 0,
            })
    
    # Drawdown analysis
    rolling_max = equity_curve['portfolio_value'].cummax()
    drawdown = (rolling_max - equity_curve['portfolio_value']) / rolling_max
    max_drawdown = drawdown.max()
    
    metrics['max_drawdown'] = max_drawdown * 100
    
    # Average drawdown duration
    if max_drawdown > 0:
        in_drawdown = False
        drawdown_start = None
        drawdown_periods = []
        
        for i, dd in enumerate(drawdown):
            if not in_drawdown and dd > 0:
                in_drawdown = True
                drawdown_start = i
            elif in_drawdown and dd == 0:
                in_drawdown = False
                if drawdown_start is not None:
                    drawdown_periods.append(i - drawdown_start)
        
        if drawdown_periods:
            metrics['avg_drawdown_duration'] = np.mean(drawdown_periods)
            metrics['max_drawdown_duration'] = max(drawdown_periods)
    
    # Trade analysis
    if trades is not None and len(trades) > 0:
        win_trades = trades[trades['pnl'] > 0] if 'pnl' in trades.columns else pd.DataFrame()
        loss_trades = trades[trades['pnl'] <= 0] if 'pnl' in trades.columns else pd.DataFrame()
        
        win_count = len(win_trades)
        loss_count = len(loss_trades)
        
        metrics.update({
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_count / (win_count + loss_count) * 100 if (win_count + loss_count) > 0 else 0,
        })
        
        if win_count > 0 and loss_count > 0:
            avg_win = win_trades['pnl'].mean()
            avg_loss = abs(loss_trades['pnl'].mean())
            
            metrics.update({
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'win_loss_ratio': avg_win / avg_loss if avg_loss > 0 else float('inf'),
                'profit_factor': abs(win_trades['pnl'].sum() / loss_trades['pnl'].sum()) if loss_trades['pnl'].sum() != 0 else float('inf'),
                'expectancy': (win_count / (win_count + loss_count) * avg_win) - (loss_count / (win_count + loss_count) * avg_loss),
            })
    
    return metrics
    
def backtest_model(model_path, data, symbol, window_size, initial_balance, output_dir):
    """Run backtest on a saved model and generate results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test environment
    env = StockScalpingEnv(
        symbol=symbol,
        data=data,
        initial_balance=initial_balance,
        window_size=window_size,
        is_training=True  # Use training mode for backtesting
    )
    
    # Load the model
    model = PPO.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    
    # Run backtest
    logger.info("Starting backtest...")
    
    # Handle Gymnasium API which returns a tuple from reset()
    obs_tuple = env.reset()
    obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
    
    done = False
    
    # Track data for visualization
    timestamps = []
    prices = []
    portfolio_values = []
    actions = []
    positions = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        
        # Handle different step return formats
        step_result = env.step(action)
        
        # Check if step returns (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result
        
        # Track metrics
        current_datetime = env.get_current_datetime()
        timestamps.append(current_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(current_datetime, datetime) else str(current_datetime))
        prices.append(info['price'])
        portfolio_values.append(info['portfolio_value'])
        actions.append(action)
        positions.append(info['position'])
    
    # Save equity curve
    equity_df = pd.DataFrame({
        'timestamp': timestamps,
        'portfolio_value': portfolio_values,
        'action': actions,
        'position': positions
    })
    equity_file = f"{output_dir}/{symbol}_equity_curve.csv"
    equity_df.to_csv(equity_file, index=False)
    
    # Save price data
    price_df = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    price_file = f"{output_dir}/{symbol}_prices.csv"
    price_df.to_csv(price_file, index=False)
    
    # Save trades
    trades_df = pd.DataFrame(env.trades)
    trades_file = f"{output_dir}/{symbol}_trades.csv"
    if not trades_df.empty:
        trades_df.to_csv(trades_file, index=False)
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(equity_df, trades_df)
    
    # Save metrics
    metrics_file = f"{output_dir}/{symbol}_metrics.json"
    pd.Series(metrics).to_json(metrics_file)
    
    # Generate plots
    plot_backtest_results(equity_file, price_file, trades_file, output_dir)
    
    # Print summary
    logger.info("\nBacktest Results:")
    logger.info(f"Initial Balance: ₹{metrics['initial_equity']:,.2f}")
    logger.info(f"Final Balance: ₹{metrics['final_equity']:,.2f}")
    logger.info(f"Total Return: ₹{metrics['absolute_return']:,.2f} ({metrics['pct_return']:.2f}%)")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
    logger.info(f"Total Trades: {metrics['trade_count']}")
    
    if 'win_rate' in metrics:
        logger.info(f"Win Rate: {metrics['win_rate']:.2f}%")
    
    if 'sharpe_ratio' in metrics:
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    
    if 'profit_factor' in metrics:
        logger.info(f"Profit Factor: {metrics['profit_factor']:.4f}")
    
    logger.info(f"Detailed results saved to {output_dir}/")
    
    return metrics

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Backtest an RL scalping model")
    parser.add_argument("--symbol", type=str, default=os.getenv('DEFAULT_SYMBOL', 'RELIANCE'),
                       help="Stock symbol to backtest on")
    parser.add_argument("--window", type=int, default=TrainingConfig.WINDOW_SIZE,
                       help="Observation window size")
    parser.add_argument("--balance", type=float, default=TrainingConfig.INITIAL_BALANCE,
                       help="Initial balance for backtesting")
    parser.add_argument("--model", type=str, default="models/scalping_model_v2.zip",
                       help="Path to model file")
    parser.add_argument("--broker", type=str, choices=["zerodha", "upstox"], default="zerodha",
                       help="Broker to fetch data from")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of days of historical data to use")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    logger.info(f"Starting backtest for {args.symbol} with {args.days} days of data")
    
    # Create directories
    os.makedirs(args.output, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Initialize data fetcher
    data_fetcher = DataFetcher(broker=args.broker)
    authenticated = data_fetcher.authenticate()
    
    if not authenticated:
        logger.warning("Authentication failed. Will attempt to use cached data.")
    
    # Fetch or load historical data
    data_file = f"data/{args.symbol}_historical_{args.days}days.csv"
    
    if os.path.exists(data_file):
        logger.info(f"Loading data from {data_file}")
        historical_data = pd.read_csv(data_file)
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data.set_index('date', inplace=True)
    else:
        logger.info(f"Fetching {args.days} days of historical data for {args.symbol}...")
        from_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y-%m-%d')
        to_date = datetime.now().strftime('%Y-%m-%d')
        
        historical_data = data_fetcher.fetch_historical_data(
            args.symbol, interval="5minute", 
            from_date=from_date, to_date=to_date
        )
        
        # Save to file
        if not historical_data.empty:
            historical_data.to_csv(data_file)
            logger.info(f"Saved data to {data_file}")
    
    if historical_data.empty:
        logger.error(f"No data available for {args.symbol}")
        return
        
    logger.info(f"Got {len(historical_data)} candles of historical data")
    
    # Run backtest
    metrics = backtest_model(
        args.model,
        historical_data,
        args.symbol,
        args.window,
        args.balance,
        args.output
    )
    
    # Clean up
    data_fetcher.stop_websocket()
    logger.info("Backtest completed!")

if __name__ == "__main__":
    main()
