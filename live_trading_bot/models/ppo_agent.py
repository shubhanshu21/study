"""
Proximal Policy Optimization (PPO) agent.
"""
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from models.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Actor-Critic network
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dim (int): Dimension of the hidden layer
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy) network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic (value) network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        Forward pass through the network
        
        Args:
            state: Current state
            
        Returns:
            tuple: (action_probs, state_value)
        """
        features = self.feature_extractor(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value
    
    def act(self, state, deterministic=False):
        """
        Take an action based on the current state
        
        Args:
            state: Current state
            deterministic (bool): Whether to take deterministic actions
            
        Returns:
            tuple: (action, action_log_prob, state_value)
        """
        action_probs, state_value = self.forward(state)
        
        if deterministic:
            action = torch.argmax(action_probs).item()
            action_log_prob = torch.log(action_probs[action])
        else:
            dist = Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
            action = action.item()
        
        return action, action_log_prob, state_value
    
    def evaluate(self, state, action):
        """
        Evaluate an action in the given state
        
        Args:
            state: Current state
            action: Action to evaluate
            
        Returns:
            tuple: (action_log_probs, state_values, entropy)
        """
        action_probs, state_value = self.forward(state)
        
        dist = Categorical(action_probs)
        action_log_probs = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action_log_probs, state_value, entropy

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """
    
    def __init__(self, state_dim, action_dim, config, name="ppo_agent"):
        """
        Initialize the PPO agent
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            config: Application configuration
            name (str): Name of the agent
        """
        super(PPOAgent, self).__init__(state_dim, action_dim, config, name)
        
        # PPO parameters
        self.learning_rate = float(config.get('PPO_LEARNING_RATE', 3e-4))
        self.gamma = float(config.get('PPO_GAMMA', 0.99))
        self.clip_eps = float(config.get('PPO_CLIP_EPS', 0.2))
        self.epochs = int(config.get('PPO_EPOCHS', 10))
        self.gae_lambda = float(config.get('PPO_GAE_LAMBDA', 0.95))
        self.entropy_coef = float(config.get('PPO_ENTROPY_COEF', 0.01))
        self.value_coef = float(config.get('PPO_VALUE_COEF', 0.5))
        self.batch_size = int(config.get('PPO_BATCH_SIZE', 64))
        
        # Create device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create actor-critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim=256).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        logger.info(f"PPO agent '{name}' initialized on device: {self.device}")
    
    def act(self, state, deterministic=False):
        """
        Take an action based on the current state
        
        Args:
            state: Current state
            deterministic (bool): Whether to take deterministic actions
            
        Returns:
            Action to take
        """
        try:
            # Convert state to tensor
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                
            # Get action from policy
            with torch.no_grad():
                action, _, _ = self.policy.act(state, deterministic)
                
            return action
            
        except Exception as e:
            logger.exception(f"Error getting action: {str(e)}")
            return np.random.randint(self.action_dim)
    
    def train(self, replay_buffer, batch_size=None):
        """
        Train the agent
        
        Args:
            replay_buffer: Buffer of experiences
            batch_size (int, optional): Batch size for training
            
        Returns:
            dict: Training metrics
        """
        try:
            # Use provided batch size or default
            batch_size = batch_size or self.batch_size
            
            # Get batch from replay buffer
            states, actions, rewards, next_states, dones, log_probs, values = replay_buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.LongTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            next_states = torch.FloatTensor(next_states).to(self.device)
            dones = torch.FloatTensor(dones).to(self.device)
            old_log_probs = torch.FloatTensor(log_probs).to(self.device)
            old_values = torch.FloatTensor(values).to(self.device)
            
            # Compute returns and advantages
            returns, advantages = self._compute_gae(rewards, values, next_states, dones)
            
            # Optimize policy for multiple epochs
            total_actor_loss = 0
            total_critic_loss = 0
            total_entropy = 0
            
            for _ in range(self.epochs):
                # Get new log probs and values
                new_log_probs, new_values, entropy = self.policy.evaluate(states, actions)
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Compute surrogate loss
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                
                # Compute actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Compute critic loss
                critic_loss = 0.5 * (returns - new_values).pow(2).mean()
                
                # Compute total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy.mean()
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Track metrics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
            
            # Update training steps
            self.training_steps += 1
            
            # Return metrics
            metrics = {
                "actor_loss": total_actor_loss / self.epochs,
                "critic_loss": total_critic_loss / self.epochs,
                "entropy": total_entropy / self.epochs
            }
            
            logger.debug(f"Training metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error training agent: {str(e)}")
            return {"error": str(e)}
    
    def _compute_gae(self, rewards, values, next_states, dones):
        """
        Compute returns and advantages using Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: Batch of rewards
            values: Batch of state values
            next_states: Batch of next states
            dones: Batch of done flags
            
        Returns:
            tuple: (returns, advantages)
        """
        # Get values for next states
        with torch.no_grad():
            next_values = self.policy(next_states)[1].squeeze()
            
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            # Compute delta
            if dones[i]:
                delta = rewards[i] - values[i]
                gae = delta
            else:
                delta = rewards[i] + self.gamma * next_values[i] - values[i]
                gae = delta + self.gamma * self.gae_lambda * gae
                
            # Compute returns and advantages
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
            
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        
        return returns, advantages
    
    def save(self, path=None):
        """
        Save the agent
        
        Args:
            path (str, optional): Path to save the agent
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Use provided path or default
            path = path or os.path.join(self.model_dir, f"{self.name}.pt")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            torch.save({
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "parameters": self.get_parameters()
            }, path)
            
            logger.info(f"Agent saved to {path}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error saving agent: {str(e)}")
            return False
    
    def load(self, path):
        """
        Load the agent
        
        Args:
            path (str): Path to load the agent from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.error(f"File not found: {path}")
                return False
                
            # Load model
            checkpoint = torch.load(path, map_location=self.device)
            
            # Load model parameters
            self.policy.load_state_dict(checkpoint["policy_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Load agent parameters
            if "parameters" in checkpoint:
                self.update_parameters(checkpoint["parameters"])
                
            logger.info(f"Agent loaded from {path}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error loading agent: {str(e)}")
            return False
    
    def get_action_probabilities(self, state):
        """
        Get action probabilities for a state
        
        Args:
            state: Current state
            
        Returns:
            numpy.ndarray: Action probabilities
        """
        try:
            # Convert state to tensor
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
                
            # Get action probabilities from policy
            with torch.no_grad():
                action_probs, _ = self.policy(state)
                
            return action_probs.cpu().numpy()[0]
            
        except Exception as e:
            logger.exception(f"Error getting action probabilities: {str(e)}")
            return super().get_action_probabilities(state)
    
    def get_parameters(self):
        """
        Get agent parameters
        
        Returns:
            dict: Agent parameters
        """
        return {
            **super().get_parameters(),
            "clip_eps": self.clip_eps,
            "epochs": self.epochs,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "batch_size": self.batch_size
        }