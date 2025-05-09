"""
Data preprocessing for trading.
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing for trading data"""
    
    def __init__(self, config):
        """
        Initialize data preprocessor
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.window_size = config.WINDOW_SIZE
        
        logger.info("Data preprocessor initialized")
    
    def preprocess(self, df):
        """
        Preprocess data for model input
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV and feature data
            
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure columns are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Basic preprocessing
            df = self._handle_missing_values(df)
            df = self._handle_outliers(df)
            df = self._normalize_features(df)
            
            logger.debug("Data preprocessing completed")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error preprocessing data: {str(e)}")
            return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        try:
            # Fill NaN values in price data
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns and df[col].isnull().any():
                    df[col].fillna(method='ffill', inplace=True)
                    df[col].fillna(method='bfill', inplace=True)
            
            # Fill missing volume with 0
            if 'volume' in df.columns:
                df['volume'].fillna(0, inplace=True)
            
            # Fill other NaN values
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            
            logger.debug("Handled missing values")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error handling missing values: {str(e)}")
            return df
    
    def _handle_outliers(self, df):
        """Handle outliers in the dataframe"""
        try:
            # List of columns to check for outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude some columns
            exclude_cols = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Handle outliers using winsorization
            for col in numeric_cols:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                
                # Winsorize the column
                df.loc[df[col] < q1, col] = q1
                df.loc[df[col] > q3, col] = q3
            
            logger.debug("Handled outliers")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error handling outliers: {str(e)}")
            return df
    
    def _normalize_features(self, df):
        """Normalize features in the dataframe"""
        try:
            # List of columns to normalize
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Exclude some columns
            exclude_cols = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            # Normalize features to [-1, 1] range
            for col in numeric_cols:
                # Check if column has variation
                if df[col].max() - df[col].min() > 0:
                    # Normalize using min-max scaling
                    df[col + '_norm'] = 2 * (df[col] - df[col].min()) / (df[col].max() - df[col].min()) - 1
                    
                    # Replace original with normalized version
                    df[col] = df[col + '_norm']
                    df.drop(col + '_norm', axis=1, inplace=True)
            
            logger.debug("Normalized features")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error normalizing features: {str(e)}")
            return df
    
    def prepare_observation(self, df, current_step, additional_features=None):
        """
        Prepare observation vector for RL model
        
        Args:
            df (pandas.DataFrame): DataFrame with preprocessed data
            current_step (int): Current step in the dataframe
            additional_features (dict, optional): Additional features to include
            
        Returns:
            numpy.ndarray: Observation vector
        """
        try:
            # Check if we have enough data
            if current_step < self.window_size:
                logger.warning(f"Not enough data to prepare observation at step {current_step}")
                return None
                
            # Get current window
            window_data = df.iloc[current_step-self.window_size+1:current_step+1]
            
            # Basic price features
            price_features = [
                window_data['close'].iloc[-1] / window_data['close'].mean(),  # Normalized close
                window_data['open'].iloc[-1] / window_data['close'].iloc[-1],  # Open relative to close
                window_data['high'].iloc[-1] / window_data['close'].iloc[-1],  # High relative to close
                window_data['low'].iloc[-1] / window_data['close'].iloc[-1],   # Low relative to close
            ]
            
            # Include volume if available
            if 'volume' in window_data.columns:
                volume_features = [
                    window_data['volume'].iloc[-1] / window_data['volume'].mean() if window_data['volume'].mean() > 0 else 0,
                ]
            else:
                volume_features = [0]
            
            # Technical indicators if available
            technical_features = []
            
            for indicator in ['RSI', 'MACD', 'ATR', 'BB_Upper', 'BB_Lower', 'STOCH_K', 'STOCH_D', 'ADX']:
                if indicator in window_data.columns:
                    technical_features.append(window_data[indicator].iloc[-1])
                    
            # Moving averages
            ma_features = []
            for ma in ['SMA20', 'SMA50', 'EMA20', 'EMA50']:
                if ma in window_data.columns:
                    ma_features.append(window_data['close'].iloc[-1] / window_data[ma].iloc[-1])
            
            # Include additional features if provided
            extra_features = []
            if additional_features:
                for key, value in additional_features.items():
                    extra_features.append(value)
            
            # Combine all features
            observation = np.array(price_features + volume_features + technical_features + ma_features + extra_features, dtype=np.float32)
            
            # Ensure observation has the right shape
            expected_length = 60  # The expected length of the observation vector
            if len(observation) > expected_length:
                observation = observation[:expected_length]
            elif len(observation) < expected_length:
                observation = np.append(observation, np.zeros(expected_length - len(observation)))
            
            return observation
            
        except Exception as e:
            logger.exception(f"Error preparing observation: {str(e)}")
            return None