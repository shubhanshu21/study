"""
Factory for loading and creating RL models.
"""
import os
import logging
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory for loading and creating RL models.
    """
    
    def __init__(self, model_dir):
        """
        Initialize the model factory
        
        Args:
            model_dir (str): Directory for saved models
        """
        self.model_dir = model_dir
        self.models = {}  # Cache for loaded models
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Model factory initialized with directory: {model_dir}")
    
    def load_model(self, symbol, model_type='ppo'):
        """
        Load a model for a symbol
        
        Args:
            symbol (str): Trading symbol
            model_type (str): Model type to load ('ppo', 'a2c', etc.)
            
        Returns:
            Model instance if successful, None otherwise
        """
        try:
            # Check if model is already loaded
            model_key = f"{symbol}_{model_type}"
            if model_key in self.models:
                logger.debug(f"Using cached model for {symbol}")
                return self.models[model_key]
                
            # Check if model file exists
            model_path = os.path.join(self.model_dir, f"{model_type}_{symbol}.zip")
            enhanced_model_path = os.path.join(self.model_dir, f"enhanced_{symbol}.zip")
            
            if os.path.exists(enhanced_model_path):
                model_path = enhanced_model_path
                logger.info(f"Using enhanced model for {symbol}")
            elif not os.path.exists(model_path):
                logger.error(f"Model file not found for {symbol}: {model_path}")
                return None
                
            # Load the model
            logger.info(f"Loading model for {symbol} from {model_path}")
            
            if model_type.lower() == 'ppo':
                # Use GPU if available
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = PPO.load(model_path, device=device)
                
                logger.info(f"Model loaded for {symbol} on device: {device}")
                
                # Cache the model
                self.models[model_key] = model
                
                return model
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            logger.exception(f"Error loading model for {symbol}: {str(e)}")
            return None