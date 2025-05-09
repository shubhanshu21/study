"""
Feature engineering for trading data.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering for trading data"""
    
    def __init__(self, config):
        """
        Initialize feature engineering
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.window_size = config.WINDOW_SIZE
        
        logger.info("Feature engineering initialized")
    
    def calculate_features(self, df):
        """
        Calculate features for a dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            pandas.DataFrame: DataFrame with added features
        """
        try:
            # Create a copy to avoid modifying the original
            df = df.copy()
            
            # Ensure columns are lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Calculate basic features
            self._calculate_price_features(df)
            self._calculate_volume_features(df)
            self._calculate_volatility_features(df)
            self._calculate_trend_features(df)
            
            # Calculate advanced features
            self._calculate_crossovers(df)
            self._calculate_support_resistance(df)
            self._calculate_divergences(df)
            
            # Fill NaN values
            df = self._handle_missing_values(df)
            
            logger.debug(f"Calculated features: {set(df.columns) - set(['open', 'high', 'low', 'close', 'volume', 'timestamp', 'date'])}")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error calculating features: {str(e)}")
            return df
    
    def _calculate_price_features(self, df):
        """Calculate price-based features"""
        try:
            # Price change and returns
            df['price_change'] = df['close'].diff()
            df['pct_change'] = df['close'].pct_change()
            
            # Rolling returns
            for period in [3, 5, 10, 20]:
                df[f'return_{period}d'] = df['close'].pct_change(periods=period)
            
            # Price relative to moving averages
            for period in [10, 20, 50, 200]:
                if f'SMA{period}' in df.columns:
                    df[f'price_to_sma{period}'] = df['close'] / df[f'SMA{period}']
                else:
                    df[f'SMA{period}'] = df['close'].rolling(window=period).mean()
                    df[f'price_to_sma{period}'] = df['close'] / df[f'SMA{period}']
            
            # Normalized price (Z-score)
            for period in [20, 50]:
                mean = df['close'].rolling(window=period).mean()
                std = df['close'].rolling(window=period).std()
                df[f'price_zscore_{period}'] = (df['close'] - mean) / std
            
            # Candle features
            df['body_size'] = abs(df['close'] - df['open'])
            df['body_ratio'] = df['body_size'] / (df['high'] - df['low'])
            df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
            df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
            df['body_to_range'] = df['body_size'] / (df['high'] - df['low'])
            
            # Price momentum
            df['momentum_1d'] = df['close'] - df['close'].shift(1)
            df['momentum_5d'] = df['close'] - df['close'].shift(5)
            
            logger.debug("Calculated price features")
            
        except Exception as e:
            logger.exception(f"Error calculating price features: {str(e)}")
    
    def _calculate_volume_features(self, df):
        """Calculate volume-based features"""
        try:
            # Only proceed if volume column exists
            if 'volume' not in df.columns:
                logger.warning("Volume column not found, skipping volume features")
                return
                
            # Volume change
            df['volume_change'] = df['volume'].pct_change()
            
            # Relative volume
            for period in [5, 10, 20]:
                df[f'rel_volume_{period}'] = df['volume'] / df['volume'].rolling(window=period).mean()
            
            # Volume trend
            df['volume_sma5'] = df['volume'].rolling(window=5).mean()
            df['volume_sma20'] = df['volume'].rolling(window=20).mean()
            df['volume_trend'] = df['volume_sma5'] / df['volume_sma20']
            
            # Price-volume relationship
            df['price_volume_trend'] = (df['pct_change'] * df['volume']).cumsum()
            
            # Volume weighted price
            df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
            
            # Volume oscillator
            df['volume_oscillator'] = df['volume_sma5'] / df['volume_sma20'] - 1
            
            logger.debug("Calculated volume features")
            
        except Exception as e:
            logger.exception(f"Error calculating volume features: {str(e)}")
    
    def _calculate_volatility_features(self, df):
        """Calculate volatility-based features"""
        try:
            # True range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            
            # Average true range
            if 'ATR' not in df.columns:
                df['ATR'] = df['true_range'].rolling(window=14).mean()
            
            # Normalized ATR
            df['norm_atr'] = df['ATR'] / df['close']
            
            # Volatility
            for period in [5, 10, 20, 50]:
                df[f'volatility_{period}'] = df['pct_change'].rolling(window=period).std() * np.sqrt(252)
            
            # Historical volatility
            if len(df) > 20:
                df['hist_volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
            
            # Volatility ratio
            df['volatility_ratio'] = df['volatility_5'] / df['volatility_20'] if 'volatility_5' in df.columns and 'volatility_20' in df.columns else np.nan
            
            logger.debug("Calculated volatility features")
            
        except Exception as e:
            logger.exception(f"Error calculating volatility features: {str(e)}")
    
    def _calculate_trend_features(self, df):
        """Calculate trend-based features"""
        try:
            # Linear regression slope
            def calc_slope(series):
                """Calculate the slope of a linear regression line"""
                x = np.arange(len(series))
                if len(x) < 2:
                    return 0
                return np.polyfit(x, series, 1)[0]
            
            # Calculate slopes over different windows
            for period in [5, 10, 20]:
                df[f'slope_{period}'] = df['close'].rolling(window=period).apply(calc_slope, raw=False)
            
            # Trend strength
            if 'ADX' in df.columns:
                df['trend_strength'] = df['ADX'] / 100  # Normalize to 0-1
            
            # Trend direction (based on EMA crossover)
            if 'EMA20' in df.columns and 'EMA50' in df.columns:
                df['trend_direction'] = np.where(df['EMA20'] > df['EMA50'], 1, -1)
            
            # RSI-based trend
            if 'RSI' in df.columns:
                df['rsi_trend'] = np.where(df['RSI'] > 50, 1, -1)
            
            # Price relative to Bollinger Bands
            if all(x in df.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                df['bb_position'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Ichimoku Cloud position
            if all(x in df.columns for x in ['ICHIMOKU_SENKOU_A', 'ICHIMOKU_SENKOU_B']):
                df['cloud_position'] = np.where(
                    df['close'] > np.maximum(df['ICHIMOKU_SENKOU_A'], df['ICHIMOKU_SENKOU_B']),
                    1,  # Above cloud
                    np.where(
                        df['close'] < np.minimum(df['ICHIMOKU_SENKOU_A'], df['ICHIMOKU_SENKOU_B']),
                        -1,  # Below cloud
                        0   # Inside cloud
                    )
                )
            
            logger.debug("Calculated trend features")
            
        except Exception as e:
            logger.exception(f"Error calculating trend features: {str(e)}")
    
    def _calculate_crossovers(self, df):
        """Calculate crossover signals"""
        try:
            # EMA crossovers
            if 'EMA20' in df.columns and 'EMA50' in df.columns:
                df['ema_crossover'] = np.where(
                    (df['EMA20'].shift(1) <= df['EMA50'].shift(1)) &
                    (df['EMA20'] > df['EMA50']),
                    1,  # Bullish crossover
                    np.where(
                        (df['EMA20'].shift(1) >= df['EMA50'].shift(1)) &
                        (df['EMA20'] < df['EMA50']),
                        -1,  # Bearish crossover
                        0    # No crossover
                    )
                )
            
            # MACD crossovers
            if 'MACD' in df.columns and 'Signal' in df.columns:
                df['macd_crossover'] = np.where(
                    (df['MACD'].shift(1) <= df['Signal'].shift(1)) &
                    (df['MACD'] > df['Signal']),
                    1,  # Bullish crossover
                    np.where(
                        (df['MACD'].shift(1) >= df['Signal'].shift(1)) &
                        (df['MACD'] < df['Signal']),
                        -1,  # Bearish crossover
                        0    # No crossover
                    )
                )
            
            # Stochastic crossovers
            if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
                df['stoch_crossover'] = np.where(
                    (df['STOCH_K'].shift(1) <= df['STOCH_D'].shift(1)) &
                    (df['STOCH_K'] > df['STOCH_D']),
                    1,  # Bullish crossover
                    np.where(
                        (df['STOCH_K'].shift(1) >= df['STOCH_D'].shift(1)) &
                        (df['STOCH_K'] < df['STOCH_D']),
                        -1,  # Bearish crossover
                        0    # No crossover
                    )
                )
            
            # Price crossing SMA
            if 'SMA50' in df.columns:
                df['price_sma_crossover'] = np.where(
                    (df['close'].shift(1) <= df['SMA50'].shift(1)) &
                    (df['close'] > df['SMA50']),
                    1,  # Bullish crossover
                    np.where(
                        (df['close'].shift(1) >= df['SMA50'].shift(1)) &
                        (df['close'] < df['SMA50']),
                        -1,  # Bearish crossover
                        0    # No crossover
                    )
                )
            
            logger.debug("Calculated crossover features")
            
        except Exception as e:
            logger.exception(f"Error calculating crossover features: {str(e)}")
    
    def _calculate_support_resistance(self, df):
        """Calculate support and resistance levels"""
        try:
            window = 20  # Look back window
            threshold = 0.02  # 2% threshold
            
            df['Support'] = np.nan
            df['Resistance'] = np.nan
            
            for i in range(window, len(df)):
                # Get window data
                window_data = df.iloc[i-window:i]
                
                # Find local minima and maxima
                local_min = window_data['low'].min()
                local_max = window_data['high'].max()
                
                # Current price
                current_price = df.iloc[i]['close']
                
                # Check if price is near support/resistance
                if abs(current_price - local_min) / local_min < threshold:
                    df.loc[df.index[i], 'Support'] = local_min
                
                if abs(local_max - current_price) / current_price < threshold:
                    df.loc[df.index[i], 'Resistance'] = local_max
            
            # Forward fill support/resistance levels
            df['Support'] = df['Support'].ffill()
            df['Resistance'] = df['Resistance'].ffill()
            
            # Calculate distance to support/resistance
            df['support_distance'] = (df['close'] - df['Support']) / df['close'] if 'Support' in df.columns else np.nan
            df['resistance_distance'] = (df['Resistance'] - df['close']) / df['close'] if 'Resistance' in df.columns else np.nan
            
            logger.debug("Calculated support/resistance features")
            
        except Exception as e:
            logger.exception(f"Error calculating support/resistance features: {str(e)}")
    
    def _calculate_divergences(self, df):
        """Calculate divergence signals"""
        try:
            # Regular bullish divergence
            if 'close' in df.columns and 'RSI' in df.columns:
                # Price making lower lows
                df['price_lower_low'] = (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))
                
                # RSI making higher lows
                df['rsi_higher_low'] = (df['RSI'] > df['RSI'].shift(1)) & (df['RSI'].shift(1) < df['RSI'].shift(2))
                
                # Bullish divergence
                df['bullish_divergence'] = df['price_lower_low'] & df['rsi_higher_low']
                
                # Clean up
                df.drop(['price_lower_low', 'rsi_higher_low'], axis=1, inplace=True)
            
            # Regular bearish divergence
            if 'close' in df.columns and 'RSI' in df.columns:
                # Price making higher highs
                df['price_higher_high'] = (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2))
                
                # RSI making lower highs
                df['rsi_lower_high'] = (df['RSI'] < df['RSI'].shift(1)) & (df['RSI'].shift(1) > df['RSI'].shift(2))
                
                # Bearish divergence
                df['bearish_divergence'] = df['price_higher_high'] & df['rsi_lower_high']
                
                # Clean up
                df.drop(['price_higher_high', 'rsi_lower_high'], axis=1, inplace=True)
            
            logger.debug("Calculated divergence features")
            
        except Exception as e:
            logger.exception(f"Error calculating divergence features: {str(e)}")
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataframe"""
        try:
            # First, forward fill
            df.fillna(method='ffill', inplace=True)
            
            # Then, backward fill for any remaining NaNs at the beginning
            df.fillna(method='bfill', inplace=True)
            
            # Finally, replace any remaining NaNs with 0
            df.fillna(0, inplace=True)
            
            logger.debug("Handled missing values")
            
            return df
            
        except Exception as e:
            logger.exception(f"Error handling missing values: {str(e)}")
            return df