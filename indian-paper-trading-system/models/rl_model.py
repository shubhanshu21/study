import os
import numpy as np
import pandas as pd
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback
from config.trading_config import (
    RL_TRAINING_TIMESTEPS, BATCH_SIZE, LEARNING_RATE, MODEL_SAVE_PATH
)

logger = logging.getLogger(__name__)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log scalar value
        if hasattr(self.model, 'env') and hasattr(self.model.env, 'envs'):
            env = self.model.env.envs[0]
            if hasattr(env, 'net_worth'):
                self.logger.record('metrics/net_worth', env.net_worth)
            if hasattr(env, 'rewards') and len(env.rewards) > 0:
                self.logger.record('metrics/last_reward', env.rewards[-1])
            if hasattr(env, 'max_drawdown'):
                self.logger.record('metrics/max_drawdown', env.max_drawdown)
        return True

class RLModel:
    """Reinforcement Learning model for trading"""
    
    def __init__(self, env=None, custom_policy=None, verbose=1):
        """Initialize RL model"""
        self.env = env
        self.model = None
        self.custom_policy = custom_policy
        self.verbose = verbose
        self.callbacks = [TensorboardCallback()]
        
        logger.info("RL model initialized")
    
    def train(self, timesteps=RL_TRAINING_TIMESTEPS, save_path=None):
        """Train the model"""
        if self.env is None:
            logger.error("Environment not set, cannot train model")
            return False
        
        logger.info(f"Starting model training for {timesteps} timesteps")
        
        try:
            # Create or load model
            if self.model is None:
                policy_kwargs = dict(
                    net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])]
                )
                
                self.model = PPO(
                    policy="MlpPolicy" if self.custom_policy is None else self.custom_policy,
                    env=self.env,
                    learning_rate=LEARNING_RATE,
                    n_steps=2048,
                    batch_size=BATCH_SIZE,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.01,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    use_sde=False,
                    sde_sample_freq=-1,
                    target_kl=None,
                    tensorboard_log="./tensorboard_logs/",
                    policy_kwargs=policy_kwargs,
                    verbose=self.verbose,
                    device='auto'
                )
            
            # Train model
            self.model.learn(
                total_timesteps=timesteps,
                callback=self.callbacks,
                reset_num_timesteps=False
            )
            
            # Save model if path provided
            if save_path:
                self.model.save(save_path)
                logger.info(f"Model saved to {save_path}")
            
            logger.info("Model training complete")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return False
    
    def load(self, model_path):
        """Load a trained model"""
        try:
            self.model = PPO.load(model_path, env=self.env)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def save(self, save_path):
        """Save the model"""
        if self.model is None:
            logger.error("No model to save")
            return False
        
        try:
            self.model.save(save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def predict(self, observation):
        """Make a prediction based on an observation"""
        if self.model is None:
            logger.error("No model loaded, cannot make prediction")
            return 0, None
        
        try:
            action, _states = self.model.predict(observation, deterministic=True)
            return action, _states
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return 0, None
    
    def evaluate(self, n_eval_episodes=10):
        """Evaluate the model performance"""
        if self.model is None or self.env is None:
            logger.error("Model or environment not set, cannot evaluate")
            return None
        
        try:
            mean_reward, std_reward = evaluate_policy(
                self.model, self.env, n_eval_episodes=n_eval_episodes
            )
            
            logger.info(f"Evaluation results: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
            return mean_reward, std_reward
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return None
    
    def optimize_hyperparameters(self, env, n_trials=100, n_timesteps=10000):
        """Optimize hyperparameters using Optuna"""
        try:
            import optuna
            from optuna.pruners import MedianPruner
            from optuna.samplers import TPESampler
            
            logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
            
            def objective(trial):
                # Define the hyperparameters to tune
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
                n_steps = trial.suggest_categorical('n_steps', [1024, 2048, 4096])
                batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
                gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.99, 0.999])
                gae_lambda = trial.suggest_categorical('gae_lambda', [0.9, 0.95, 0.99])
                ent_coef = trial.suggest_loguniform('ent_coef', 0.0001, 0.1)
                
                # Create model with trial hyperparameters
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    ent_coef=ent_coef,
                    verbose=0
                )
                
                # Train for a short period
                model.learn(total_timesteps=n_timesteps)
                
                # Evaluate
                mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
                
                return mean_reward
            
            # Create study
            sampler = TPESampler(n_startup_trials=10)
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            
            study = optuna.create_study(
                study_name="ppo_optimization",
                direction="maximize",
                sampler=sampler,
                pruner=pruner
            )
            
            # Run optimization
            study.optimize(objective, n_trials=n_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_reward = study.best_value
            
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best reward: {best_reward:.2f}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {str(e)}")
            return None
    
    def set_environment(self, env):
        """Set or update the environment"""
        self.env = env
        
        # Update model environment if already created
        if self.model is not None:
            self.model.set_env(env)
    
    def get_model_summary(self):
        """Get a summary of the model architecture"""
        if self.model is None:
            return "No model loaded"
        
        try:
            summary = []
            policy = self.model.policy
            
            # Basic info
            summary.append(f"Model type: {type(self.model).__name__}")
            summary.append(f"Policy type: {type(policy).__name__}")
            
            # Network architecture
            if hasattr(policy, 'net_arch'):
                summary.append(f"Network architecture: {policy.net_arch}")
            
            # Parameters
            if hasattr(self.model, 'learning_rate'):
                summary.append(f"Learning rate: {self.model.learning_rate}")
            if hasattr(self.model, 'gamma'):
                summary.append(f"Gamma (discount): {self.model.gamma}")
            if hasattr(self.model, 'n_steps'):
                summary.append(f"N steps: {self.model.n_steps}")
            
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error getting model summary: {str(e)}")
            return "Error generating model summary"


class CustomTradingPolicy(ActorCriticPolicy):
    """Custom policy for trading specific features"""
    
    def __init__(self, *args, **kwargs):
        super(CustomTradingPolicy, self).__init__(*args, **kwargs)
        
    def forward(self, obs, deterministic=False):
        """Forward pass in network"""
        return super().forward(obs, deterministic)
    
    @staticmethod
    def _predict(self, observation, deterministic=False):
        """Predict action and value from observation"""
        return super()._predict(observation, deterministic)


def create_and_train_model(env, symbol, timesteps=RL_TRAINING_TIMESTEPS):
    """Helper function to create and train a model for a specific symbol"""
    
    # Ensure model save path exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Create model
    rl_model = RLModel(env=env)
    
    # Train model
    model_path = os.path.join(MODEL_SAVE_PATH, f"{symbol}.zip")
    success = rl_model.train(timesteps=timesteps, save_path=model_path)
    
    return success, model_path, rl_model