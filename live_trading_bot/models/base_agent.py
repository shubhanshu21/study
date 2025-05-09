"""
Base agent class for reinforcement learning.
"""
import os
import logging
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base agent class for reinforcement learning.
    All RL agents should inherit from this class.
    """
    
    def __init__(self, state_dim, action_dim, config, name="base_agent"):
        """
        Initialize the base agent
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config: Application configuration
            name (str): Name of the agent
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.name = name
        
        # Create model directory if it doesn't exist
        self.model_dir = os.path.join(config.MODEL_DIR, name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Training parameters
        self.learning_rate = 3e-4
        self.gamma = 0.99  # Discount factor
        self.training_steps = 0
        
        logger.info(f"Base agent '{name}' initialized")
    
    @abstractmethod
    def act(self, state, deterministic=False):
        """
        Take an action based on the current state
        
        Args:
            state: Current state
            deterministic (bool): Whether to take deterministic actions
            
        Returns:
            Action to take
        """
        pass
    
    @abstractmethod
    def train(self, replay_buffer, batch_size):
        """
        Train the agent
        
        Args:
            replay_buffer: Buffer of experiences
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path=None):
        """
        Save the agent
        
        Args:
            path (str, optional): Path to save the agent
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, path):
        """
        Load the agent
        
        Args:
            path (str): Path to load the agent from
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for a state
        
        Args:
            state: Current state
            
        Returns:
            numpy.ndarray: Action probabilities
        """
        # Default implementation returns uniform distribution
        return np.ones(self.action_dim) / self.action_dim
    
    def update_parameters(self, parameters):
        """
        Update agent parameters
        
        Args:
            parameters (dict): New parameters
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            for key, value in parameters.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    logger.debug(f"Updated parameter '{key}' to {value}")
                else:
                    logger.warning(f"Parameter '{key}' not found in agent")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating parameters: {str(e)}")
            return False
    
    def get_parameters(self):
        """
        Get agent parameters
        
        Returns:
            dict: Agent parameters
        """
        return {
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "training_steps": self.training_steps
        }