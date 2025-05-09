"""
Trainer class for the RL scalping agent.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from .environment import StockScalpingEnv
from ..data.data_fetcher import DataFetcher
from ..constants import TrainingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Trainer")


class ScalpingCallback(BaseCallback):
    """Custom callback for training the RL agent"""
    
    def __init__(self, verbose=0, log_interval=100):
        super(ScalpingCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.best_reward = -np.inf
        self.best_model = None
        
    def _on_step(self) -> bool:
        """Called at each step"""
        if self.n_calls % self.log_interval == 0:
            # Log training progress
            ep_rewards = self.training_env.get_attr('rewards_history')[0]
            if len(ep_rewards) > 0:
                mean_reward = np.mean(ep_rewards[-self.log_interval:])
                mean_drawdown = self.training_env.get_attr('max_drawdown')[0]
                trades = self.training_env.get_attr('total_trades')[0]
                success_rate = (self.training_env.get_attr('successful_trades')[0] / trades 
                               if trades > 0 else 0)
                
                logger.info(f"Step: {self.n_calls}")
                logger.info(f"Mean reward: {mean_reward:.4f}")
                logger.info(f"Max drawdown: {mean_drawdown * 100:.2f}%")
                logger.info(f"Trades: {trades}")
                logger.info(f"Success rate: {success_rate * 100:.2f}%")
                
                # Save best model
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    self.best_model = self.model
                    
                    # Save best model
                    model_path = os.path.join(os.getcwd(), "models", "best_scalping_model")
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    self.best_model.save(model_path)
                    
                    logger.info(f"New best model saved! Reward: {mean_reward:.4f}")
                    
        return True


class ScalpingTrainer:
    """Class for training the RL agent"""
    
    def __init__(self, symbol: str, data: pd.DataFrame = None, window_size: int = TrainingConfig.WINDOW_SIZE,
                initial_balance: float = TrainingConfig.INITIAL_BALANCE, n_envs: int = TrainingConfig.N_ENVS, 
                data_fetcher: DataFetcher = None):
        """Initialize the trainer"""
        self.symbol = symbol
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.n_envs = n_envs
        self.data_fetcher = data_fetcher
        
        # Create training environment
        self.create_env()
        
        # Initialize model
        self.model = None
        
        logger.info(f"Initialized trainer for {symbol}")
        
    def create_env(self):
        """Create the training environment"""
        if self.data is None and self.data_fetcher:
            # Fetch historical data for training
            logger.info("Fetching historical data for training...")
            self.data = self.data_fetcher.fetch_historical_data(
                self.symbol, interval='5minute', 
                from_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                to_date=datetime.now().strftime('%Y-%m-%d')
            )
            
        if self.data.empty:
            raise ValueError("No data available for training")
            
        # Create environment
        def make_env():
            env = StockScalpingEnv(
                symbol=self.symbol,
                data=self.data,
                initial_balance=self.initial_balance,
                window_size=self.window_size,
                is_training=True
            )
            return env
            
        # Create vectorized environment
        self.env = DummyVecEnv([make_env for _ in range(self.n_envs)])
        logger.info(f"Created {self.n_envs} vectorized environments")
        
    def train(self, total_timesteps: int = TrainingConfig.DEFAULT_TIMESTEPS, 
              log_interval: int = TrainingConfig.LOG_INTERVAL):
        """Train the RL agent"""
        # Create model if not exists
        if self.model is None:
            self.model = PPO(
                "MlpPolicy",
                self.env,
                verbose=0,
                learning_rate=TrainingConfig.LEARNING_RATE,
                n_steps=TrainingConfig.N_STEPS,
                batch_size=TrainingConfig.BATCH_SIZE,
                n_epochs=TrainingConfig.N_EPOCHS,
                gamma=TrainingConfig.GAMMA,
                gae_lambda=TrainingConfig.GAE_LAMBDA,
                clip_range=TrainingConfig.CLIP_RANGE,
                clip_range_vf=None,
                ent_coef=TrainingConfig.ENT_COEF,
                vf_coef=TrainingConfig.VF_COEF,
                max_grad_norm=TrainingConfig.MAX_GRAD_NORM,
                tensorboard_log=TrainingConfig.TENSORBOARD_LOG
            )
            
        # Create callback
        callback = ScalpingCallback(log_interval=log_interval)
        
        # Create directories
        os.makedirs(os.path.dirname(TrainingConfig.TENSORBOARD_LOG), exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Train the model
        logger.info(f"Starting training for {total_timesteps} steps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        logger.info("Training completed!")
        
        # Return best model from callback
        if callback.best_model is not None:
            self.model = callback.best_model
            
        return self.model
        
    def save_model(self, path: str = "models/scalping_model"):
        """Save the model"""
        if self.model:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info(f"Model saved to {path}")
        else:
            logger.warning("No model to save. Train first.")
            
    def load_model(self, path: str = "models/scalping_model"):
        """Load a trained model"""
        try:
            self.model = PPO.load(path, self.env)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            
    def backtest(self, test_data: pd.DataFrame = None):
        """
        Backtest the model on test data.
        
        Args:
            test_data: Optional test data for backtesting. If None, uses the last 30% of available data.
            
        Returns:
            Dict containing backtest results and performance metrics
        """
        logger.info("Starting backtest...")
        
        if self.model is None:
            logger.warning("No model available. Train or load first.")
            return {'status': 'failed', 'message': 'No model available'}
            
        try:
            if test_data is None:
                # Use last 30% of available data for testing
                train_size = int(len(self.data) * 0.7)
                test_data = self.data.iloc[train_size:]
                logger.info(f"Using {len(test_data)} data points ({train_size}:{len(self.data)}) for backtest")
                
            # Create test environment
            test_env = StockScalpingEnv(
                symbol=self.symbol,
                data=test_data,
                initial_balance=self.initial_balance,
                window_size=self.window_size,
                is_training=True  # Still use training mode for backtesting
            )
            
            # Run backtest
            # Handle the Gymnasium API which returns a tuple from reset()
            obs_tuple = test_env.reset()
            # Extract just the observation (first element) if it's a tuple
            if isinstance(obs_tuple, tuple):
                obs = obs_tuple[0]  # This fixes the API compatibility issue
            else:
                obs = obs_tuple
                
            done = False
            total_steps = 0
            
            # Track for visualization
            prices = []
            portfolio_values = []
            actions = []
            timestamps = []
            positions = []
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                
                # Handle Gymnasium API which might return (obs, reward, terminated, truncated, info)
                # or the older (obs, reward, done, info) format
                step_result = test_env.step(action)
                
                if len(step_result) == 5:  # New Gymnasium API
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Old Gym API
                    obs, reward, done, info = step_result
                
                # Track metrics
                prices.append(info['price'])
                portfolio_values.append(info['portfolio_value'])
                actions.append(action)
                positions.append(info['position'])
                timestamps.append(test_env.get_current_datetime())
                
                total_steps += 1
                
            # Calculate metrics
            returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else []
            sharpe_ratio = self._calculate_sharpe_ratio(portfolio_values)
            max_drawdown = test_env.max_drawdown
            
            # Calculate daily returns
            if len(timestamps) > 1:
                daily_returns = self._calculate_daily_returns(timestamps, portfolio_values)
            else:
                daily_returns = []
            
            # Save backtest results
            backtest_dir = "data/backtest"
            os.makedirs(backtest_dir, exist_ok=True)
            
            # Save trades to CSV
            trades_file = f"{backtest_dir}/{self.symbol}_backtest_trades.csv"
            trades_df = pd.DataFrame(test_env.trades)
            if not trades_df.empty:
                trades_df.to_csv(trades_file, index=False)
            
            # Create equity curve data
            equity_data = {
                'timestamp': [t.strftime('%Y-%m-%d %H:%M:%S') if isinstance(t, datetime) else t for t in timestamps],
                'price': prices,
                'portfolio_value': portfolio_values,
                'action': actions,
                'position': positions
            }
            
            equity_df = pd.DataFrame(equity_data)
            equity_file = f"{backtest_dir}/{self.symbol}_backtest_equity.csv"
            equity_df.to_csv(equity_file, index=False)
            
            # Print backtest results
            logger.info("\nBacktest Results:")
            logger.info(f"Initial Balance: ₹{self.initial_balance:,.2f}")
            logger.info(f"Final Balance: ₹{test_env.balance:,.2f}")
            logger.info(f"Total P&L: ₹{test_env.total_pnl:,.2f} ({test_env.total_pnl / self.initial_balance * 100:.2f}%)")
            logger.info(f"Max Drawdown: {max_drawdown * 100:.2f}%")
            logger.info(f"Total Trades: {test_env.total_trades}")
            if test_env.total_trades > 0:
                logger.info(f"Success Rate: {test_env.successful_trades / test_env.total_trades * 100:.2f}%")
            logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
            
            # Prepare and return results
            return {
                'status': 'success',
                'balance': test_env.balance,
                'pnl': test_env.total_pnl,
                'pnl_pct': test_env.total_pnl / self.initial_balance * 100,
                'max_drawdown': max_drawdown,
                'trades': test_env.total_trades,
                'success_rate': test_env.successful_trades / test_env.total_trades if test_env.total_trades > 0 else 0,
                'sharpe': sharpe_ratio,
                'equity_curve': portfolio_values,
                'trade_history': test_env.trades,
                'equity_file': equity_file,
                'trades_file': trades_file,
                'daily_returns': daily_returns
            }
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'failed', 'message': str(e)}
        
    def _calculate_sharpe_ratio(self, equity_curve: List[float], risk_free_rate: float = 0.05, trading_days: int = 252) -> float:
        """Calculate Sharpe ratio from equity curve"""
        if len(equity_curve) < 2:
            return 0
            
        # Calculate daily returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Annualize
        mean_return = np.mean(returns) * trading_days
        std_return = np.std(returns) * np.sqrt(trading_days)
        
        if std_return == 0:
            return 0
            
        # Sharpe ratio
        sharpe = (mean_return - risk_free_rate) / std_return
        return sharpe
        
    def _calculate_daily_returns(self, timestamps, portfolio_values):
        """Calculate daily returns from equity curve"""
        daily_data = {}
        
        for i, ts in enumerate(timestamps):
            date_str = ts.strftime('%Y-%m-%d') if isinstance(ts, datetime) else str(ts).split(' ')[0]
            if date_str not in daily_data:
                daily_data[date_str] = []
            daily_data[date_str].append(portfolio_values[i])
        
        daily_returns = []
        prev_value = None
        
        for date in sorted(daily_data.keys()):
            # Use closing value of the day
            day_closing = daily_data[date][-1]
            
            if prev_value is not None:
                daily_return = (day_closing - prev_value) / prev_value
                daily_returns.append({
                    'date': date,
                    'return': daily_return,
                    'value': day_closing
                })
            
            prev_value = day_closing
        
        return daily_returns