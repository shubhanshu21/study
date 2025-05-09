"""
Custom policy network for the RL model.
Includes attention mechanism and separate feature processing paths.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces
import numpy as np

class AttentionBlock(nn.Module):
    """Self-attention block for time series data"""
    def __init__(self, feature_dim, num_heads=4):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.ReLU(),
            nn.Linear(feature_dim * 4, feature_dim)
        )
        
    def forward(self, x):
        # x shape: (seq_len, batch_size, feature_dim)
        attended, _ = self.attention(x, x, x)
        x = x + attended  # Residual connection
        x = self.layer_norm1(x)
        
        ff_output = self.feed_forward(x)
        x = x + ff_output  # Residual connection
        x = self.layer_norm2(x)
        
        return x

class CustomFeatureExtractor(nn.Module):
    """Custom feature extractor for stock trading data"""
    def __init__(self, observation_space, window_size=15, feature_columns=10):
        super(CustomFeatureExtractor, self).__init__()
        
        # Calculate dimensions
        self.window_size = window_size
        self.feature_columns = feature_columns
        self.time_series_dims = window_size * feature_columns
        self.position_dims = 6  # Position indicators
        
        # Feature extractor for time series data
        self.price_extractor = nn.Sequential(
            nn.Linear(5 * window_size, 128),  # OHLCV data
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        self.indicator_extractor = nn.Sequential(
            nn.Linear((feature_columns - 5) * window_size, 128),  # Technical indicators
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU()
        )
        
        # Position features
        self.position_extractor = nn.Sequential(
            nn.Linear(self.position_dims, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU()
        )
        
        # Attention mechanism for time series data
        self.attention = AttentionBlock(feature_dim=64, num_heads=4)
        
        # Final feature combination
        self.combiner = nn.Sequential(
            nn.Linear(64 + 64 + 16, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        
    def forward(self, observations):
        batch_size = observations.shape[0]
        
        # Split features
        price_features = observations[:, :5*self.window_size]
        indicator_features = observations[:, 5*self.window_size:self.time_series_dims]
        position_features = observations[:, self.time_series_dims:]
        
        # Extract features
        price_features = self.price_extractor(price_features)
        indicator_features = self.indicator_extractor(indicator_features)
        
        # Reshape for attention (seq_len, batch_size, feature_dim)
        price_features_reshaped = price_features.view(batch_size, 1, -1).transpose(0, 1)
        indicator_features_reshaped = indicator_features.view(batch_size, 1, -1).transpose(0, 1)
        
        # Apply attention
        price_features_attended = self.attention(price_features_reshaped).transpose(0, 1).squeeze(1)
        
        # Process position features
        position_features = self.position_extractor(position_features)
        
        # Combine all features
        combined = torch.cat([price_features_attended, indicator_features, position_features], dim=1)
        features = self.combiner(combined)
        
        return features

class CustomActorCriticPolicy(ActorCriticPolicy):
    """Custom actor-critic policy for stock trading"""
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        window_size=15,
        feature_columns=10,
        *args,
        **kwargs
    ):
        # Adjust network architecture
        policy_kwargs = kwargs.get("policy_kwargs", {})
        policy_kwargs.update({
            "net_arch": [dict(pi=[128, 64], vf=[128, 64])],
            "activation_fn": nn.LeakyReLU
        })
        kwargs["policy_kwargs"] = policy_kwargs
        
        super(CustomActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs
        )
        
        # Override feature extractor
        self.features_extractor = CustomFeatureExtractor(
            observation_space,
            window_size=window_size,
            feature_columns=feature_columns
        )
        
        # Recreate the network with our custom feature extractor
        self._build(lr_schedule)
        
    def _build(self, lr_schedule):
        """Build networks and optimizers"""
        super(CustomActorCriticPolicy, self)._build(lr_schedule)
        
        # Add additional layer before final output for better trading decisions
        self.action_net = nn.Sequential(
            self.action_net,
            nn.Dropout(0.1),
            nn.Linear(self.action_space.n, self.action_space.n),
            nn.Softmax(dim=1)
        )
        
        # Value function gets a boost too
        self.value_net = nn.Sequential(
            self.value_net,
            nn.Dropout(0.1),
            nn.Linear(1, 1)
        )
        
        # Update optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs
        )
