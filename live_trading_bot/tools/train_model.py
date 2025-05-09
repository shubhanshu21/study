#!/usr/bin/env python
"""
Train RL model for trading.
This tool is used to train or retrain models for specific trading symbols.
"""
import os
import sys
import argparse
import logging
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from tqdm import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from gymnasium_environment import GymTradingEnvironment

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import project modules
from config.app_config import AppConfig
from config.logging_config import setup_logging
from data.data_loader import ZerodhaDataLoader
from data.features import FeatureEngineering
from data.preprocessor import DataPreprocessor
from trading.environment import TradingEnvironment
from utils.indicators import calculate_indicators

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train RL trading model')
    parser.add_argument('--symbol', type=str, required=True, help='Trading symbol')
    parser.add_argument('--days', type=int, default=365, help='Number of days of data to use')
    parser.add_argument('--trials', type=int, default=100, help='Number of hyperparameter optimization trials')
    parser.add_argument('--steps', type=int, default=300000, help='Number of training steps')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level')
    parser.add_argument('--no-optimize', action='store_true', help='Skip hyperparameter optimization')
    parser.add_argument('--retrain', action='store_true', help='Retrain existing model')
    return parser.parse_args()

def load_data(symbol, days, config, logger):
    """
    Load and prepare data for training
    
    Args:
        symbol (str): Trading symbol
        days (int): Number of days of data to load
        config: Configuration object
        logger: Logger instance
        
    Returns:
        tuple: (train_df, test_df)
    """
    try:
        logger.info(f"Loading data for {symbol}")
        
        # Check if data file exists
        data_path = os.path.join(config.DATA_DIR, f"{symbol}.csv")
        
        if os.path.exists(data_path):
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # Ensure date column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
            # Filter to requested number of days
            if days > 0 and 'date' in df.columns:
                end_date = df['date'].max()
                start_date = end_date - timedelta(days=days)
                df = df[df['date'] >= start_date]
                
            logger.info(f"Loaded {len(df)} rows of data for {symbol}")
        else:
            logger.warning(f"Data file not found for {symbol}. Using sample data.")
            
            # Generate sample data
            df = generate_sample_data(symbol, days)
            
            # Save sample data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
            
            logger.info(f"Generated and saved sample data with {len(df)} rows for {symbol}")
        
        # Calculate technical indicators
        feature_engineering = FeatureEngineering(config)
        df = feature_engineering.calculate_features(df)
        
        # Calculate additional indicators
        df = calculate_indicators(df)
        
        # Preprocess data
        preprocessor = DataPreprocessor(config)
        df = preprocessor.preprocess(df)
        
        # Split data into training and testing sets
        split_ratio = 0.8
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy().reset_index(drop=True)
        test_df = df.iloc[split_idx:].copy().reset_index(drop=True)
        
        logger.info(f"Data preparation complete. Train: {len(train_df)} rows, Test: {len(test_df)} rows")
        
        return train_df, test_df
        
    except Exception as e:
        logger.exception(f"Error loading data: {str(e)}")
        return None, None

def generate_sample_data(symbol, days):
    """
    Generate sample data for testing
    
    Args:
        symbol (str): Symbol name
        days (int): Number of days
        
    Returns:
        pandas.DataFrame: Sample data
    """
    # Generate dates
    end_date = datetime.now().date()
    dates = [end_date - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    # Generate prices with random walk
    start_price = 1000.0
    price = start_price
    prices = []
    for _ in range(len(dates)):
        # Random daily return between -2% and 2%
        daily_return = np.random.normal(0.0005, 0.015)
        price *= (1 + daily_return)
        prices.append(price)
    
    # Generate OHLC data
    data = []
    for i, date in enumerate(dates):
        # Generate intraday variation
        close_price = prices[i]
        open_price = close_price * (1 + np.random.normal(0, 0.005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))
        
        # Generate volume
        volume = int(np.random.normal(100000, 30000))
        if volume < 10000:
            volume = 10000
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume,
            'symbol': symbol
        })
    
    return pd.DataFrame(data)

def create_environment(df, symbol, config, logger):
    """
    Create trading environment
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLCV and feature data
        symbol (str): Trading symbol
        config: Configuration object
        logger: Logger instance
        
    Returns:
        gym.Env: Trading environment
    """
    try:
        # Create environment using the new GymTradingEnvironment class
        env = GymTradingEnvironment(
            symbol=symbol,
            data=df,
            config=config,
            initial_balance=float(getattr(config, 'DEFAULT_INITIAL_BALANCE', 100000)),
            window_size=int(getattr(config, 'WINDOW_SIZE', 20))
        )
        
        # Wrap environment with Monitor
        log_dir = os.path.join(config.LOG_DIR, 'sb3_logs')
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        
        # Create vectorized environment
        env = DummyVecEnv([lambda: env])
        
        return env
        
    except Exception as e:
        logger.exception(f"Error creating environment: {str(e)}")
        return None

def objective(trial, train_df, test_df, symbol, config, logger):
    """
    Objective function for hyperparameter optimization
    
    Args:
        trial: Optuna trial
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Testing data
        symbol (str): Trading symbol
        config: Configuration object
        logger: Logger instance
        
    Returns:
        float: Negative mean reward (to be minimized)
    """
    try:
        # Define hyperparameters to optimize
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
        n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096, 8192])
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        gae_lambda = trial.suggest_uniform('gae_lambda', 0.8, 0.99)
        gamma = trial.suggest_uniform('gamma', 0.9, 0.999)
        n_epochs = trial.suggest_categorical('n_epochs', [5, 10, 15, 20])
        ent_coef = trial.suggest_loguniform('ent_coef', 0.001, 0.1)
        clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
        
        # Define policy network architecture
        net_arch = [
            dict(
                pi=[trial.suggest_categorical('pi_l1', [64, 128, 256]), 
                    trial.suggest_categorical('pi_l2', [32, 64, 128])], 
                vf=[trial.suggest_categorical('vf_l1', [64, 128, 256]), 
                    trial.suggest_categorical('vf_l2', [32, 64, 128])]
            )
        ]
        
        policy_kwargs = dict(net_arch=net_arch)
        
        # Create environment
        env = create_environment(train_df, symbol, config, logger)
        
        if env is None:
            return float('inf')
        
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_epochs=n_epochs,
            ent_coef=ent_coef,
            clip_range=clip_range,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device
        )
        
        # Train model (with reduced steps for optimization)
        model.learn(total_timesteps=50000)
        
        # Evaluate model
        eval_env = create_environment(test_df, symbol, config, logger)
        
        if eval_env is None:
            return float('inf')
            
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
        
        # Log results
        logger.info(f"Trial {trial.number}: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        # Return negative mean reward (to be minimized)
        return -mean_reward
        
    except Exception as e:
        logger.exception(f"Error in optimization trial: {str(e)}")
        return float('inf')

def optimize_hyperparameters(train_df, test_df, symbol, config, n_trials, logger):
    """
    Optimize hyperparameters using Optuna
    
    Args:
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Testing data
        symbol (str): Trading symbol
        config: Configuration object
        n_trials (int): Number of optimization trials
        logger: Logger instance
        
    Returns:
        dict: Best hyperparameters
    """
    try:
        logger.info(f"Starting hyperparameter optimization for {symbol} with {n_trials} trials")
        
        # Create study
        study_name = f"ppo_{symbol}"
        storage_name = f"sqlite:///{os.path.join(config.DB_DIR, 'optuna.db')}"
        
        sampler = TPESampler(n_startup_trials=10)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=10000)
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
            direction='minimize'
        )
        
        # Run optimization
        study.optimize(
            lambda trial: objective(trial, train_df, test_df, symbol, config, logger),
            n_trials=n_trials
        )
        
        # Get best parameters
        best_params = study.best_params
        best_value = -study.best_value  # Convert back to positive reward
        
        logger.info(f"Optimization complete. Best reward: {best_value:.2f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Save best parameters
        best_params_path = os.path.join(config.MODEL_DIR, f"{symbol}_best_params.json")
        os.makedirs(os.path.dirname(best_params_path), exist_ok=True)
        
        with open(best_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
            
        logger.info(f"Best parameters saved to {best_params_path}")
        
        return best_params
        
    except Exception as e:
        logger.exception(f"Error optimizing hyperparameters: {str(e)}")
        return {}

def train_model(train_df, test_df, symbol, config, params, steps, logger, retrain=False):
    """
    Train model with the given parameters
    
    Args:
        train_df (pandas.DataFrame): Training data
        test_df (pandas.DataFrame): Testing data
        symbol (str): Trading symbol
        config: Configuration object
        params (dict): Model hyperparameters
        steps (int): Number of training steps
        logger: Logger instance
        retrain (bool): Whether to retrain an existing model
        
    Returns:
        tuple: (model, model_path)
    """
    try:
        logger.info(f"Training model for {symbol} with {steps} steps")
        
        # Create environment
        env = create_environment(train_df, symbol, config, logger)
        
        if env is None:
            return None, None
            
        # Define model path
        model_path = os.path.join(config.MODEL_DIR, f"{symbol}.zip")
        
        # Check if model exists and whether to retrain
        if os.path.exists(model_path) and retrain:
            logger.info(f"Loading existing model from {model_path} for retraining")
            model = PPO.load(model_path, env=env)
        else:
            # Create model with optimized parameters
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Prepare policy kwargs
            policy_kwargs = dict(
                net_arch=[
                    dict(
                        pi=[params.get('pi_l1', 128), params.get('pi_l2', 64)],
                        vf=[params.get('vf_l1', 128), params.get('vf_l2', 64)]
                    )
                ]
            )
            
            # Create model
            model = PPO(
                'MlpPolicy',
                env,
                learning_rate=params.get('learning_rate', 3e-4),
                n_steps=params.get('n_steps', 2048),
                batch_size=params.get('batch_size', 64),
                gae_lambda=params.get('gae_lambda', 0.95),
                gamma=params.get('gamma', 0.99),
                n_epochs=params.get('n_epochs', 10),
                ent_coef=params.get('ent_coef', 0.01),
                clip_range=params.get('clip_range', 0.2),
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=device
            )
        
        # Train model with progress bar
        progress_bar = tqdm(total=steps, desc=f"Training {symbol}")
        
        # Define callback to update progress bar
        def progress_callback(locals, globals):
            progress_bar.update(1)
            return True
            
        # Train model in smaller chunks to show progress
        chunk_size = 5000
        remaining_steps = steps
        
        while remaining_steps > 0:
            current_chunk = min(chunk_size, remaining_steps)
            model.learn(total_timesteps=current_chunk, callback=progress_callback, reset_num_timesteps=False)
            remaining_steps -= current_chunk
            
            # Save checkpoint
            model.save(model_path)
            logger.info(f"Saved checkpoint at {steps - remaining_steps} steps")
        
        progress_bar.close()
        
        # Save final model
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Evaluate model
        eval_env = create_environment(test_df, symbol, config, logger)
        
        if eval_env is None:
            return model, model_path
            
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        
        logger.info(f"Evaluation complete. Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
        
        return model, model_path
        
    except Exception as e:
        logger.exception(f"Error training model: {str(e)}")
        return None, None

def main():
    """Main function"""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = AppConfig(args.config)
    
    # Setup logging
    logger = setup_logging(
        level=args.log_level,
        log_dir=config.LOG_DIR,
        service_name=f"train_{args.symbol}"
    )
    
    logger.info(f"Starting model training for {args.symbol}")
    
    # Load data
    train_df, test_df = load_data(args.symbol, args.days, config, logger)
    
    if train_df is None or test_df is None:
        logger.error("Data loading failed. Exiting.")
        return 1
    
    # Optimize hyperparameters if requested
    if not args.no_optimize:
        best_params = optimize_hyperparameters(
            train_df=train_df,
            test_df=test_df,
            symbol=args.symbol,
            config=config,
            n_trials=args.trials,
            logger=logger
        )
    else:
        # Load best parameters if they exist
        best_params_path = os.path.join(config.MODEL_DIR, f"{args.symbol}_best_params.json")
        if os.path.exists(best_params_path):
            logger.info(f"Loading best parameters from {best_params_path}")
            with open(best_params_path, 'r') as f:
                best_params = json.load(f)
        else:
            logger.warning("No optimized parameters found. Using defaults.")
            best_params = {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'gae_lambda': 0.95,
                'gamma': 0.99,
                'n_epochs': 10,
                'ent_coef': 0.01,
                'clip_range': 0.2,
                'pi_l1': 128,
                'pi_l2': 64,
                'vf_l1': 128,
                'vf_l2': 64
            }
    
    # Train model
    model, model_path = train_model(
        train_df=train_df,
        test_df=test_df,
        symbol=args.symbol,
        config=config,
        params=best_params,
        steps=args.steps,
        logger=logger,
        retrain=args.retrain
    )
    
    if model is None:
        logger.error("Model training failed. Exiting.")
        return 1
        
    logger.info(f"Model training complete for {args.symbol}")
    logger.info(f"Model saved to {model_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())