import pandas as pd
import numpy as np
import logging
from datetime import datetime, time, timedelta
import pytz
from abc import ABC, abstractmethod
from config.trading_config import MAX_POSITION_SIZE_PCT

logger = logging.getLogger(__name__)

class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name="BaseStrategy"):
        """Initialize trading strategy"""
        self.name = name
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    def generate_signals(self, symbol, current_data, historical_data):
        """Generate trading signals
        
        Args:
            symbol: Stock symbol
            current_data: Dictionary with current market data
            historical_data: DataFrame with historical data
            
        Returns:
            Dictionary with signal information
        """
        pass
    
    def calculate_position_size(self, symbol, signal_strength, price, available_capital, existing_position_qty=0):
        """Calculate appropriate position size based on signal strength and available capital"""
        # Base position size as percentage of available capital
        base_position_pct = MAX_POSITION_SIZE_PCT
        
        # Adjust based on signal strength (0.0 to 1.0)
        adjusted_position_pct = base_position_pct * min(max(signal_strength, 0.1), 1.0)
        
        # Calculate position value
        position_value = available_capital * adjusted_position_pct
        
        # Calculate number of shares
        num_shares = int(position_value // price)
        
        # Adjust for existing position
        if existing_position_qty > 0:
            num_shares = max(num_shares - existing_position_qty, 0)
        
        return num_shares, adjusted_position_pct

class RLBasedStrategy(TradingStrategy):
    """Strategy that uses a pre-trained Reinforcement Learning model"""
    
    def __init__(self, model=None, name="RLStrategy"):
        """Initialize RL-based strategy"""
        super().__init__(name=name)
        self.model = model
        
        # Observation preprocessing
        self.window_size = 20
        
        logger.info(f"RL-based strategy '{name}' initialized")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def _prepare_observation(self, symbol, current_data, historical_data):
        """Prepare observation for the RL model"""
        # Get current price data
        current_price = current_data["Close"]
        
        # Get window of historical data
        if len(historical_data) > self.window_size:
            window_data = historical_data.iloc[-self.window_size:]
        else:
            window_data = historical_data
        
        # Basic account state features (normalized)
        obs = np.array([
            1.0,  # Placeholder for balance
            0.0,  # Placeholder for shares held
            0.0,  # Placeholder for cost basis
            current_price / window_data["Close"].mean(),  # Normalized price
            0.0,  # Is in long position
            0.0,  # Is in short position
            0.0,  # Normalized trailing stop
            0.0,  # Normalized target
            0.0,  # Days in trade
            0.0,  # Trading stopped
            0.0,  # Position ratio
            0.0,  # Daily trades ratio
            0.0,  # Drawdown
            0.0,  # Consecutive losses
            0.0,  # Consecutive wins
            0.0,  # Buy signal strength
            0.0,  # Sell signal strength
            # Market regime (one-hot encoding)
            1 if current_data.get('MARKET_REGIME_TEXT') == "trending_up" else 0,
            1 if current_data.get('MARKET_REGIME_TEXT') == "trending_down" else 0,
            1 if current_data.get('MARKET_REGIME_TEXT') == "ranging" else 0,
            1 if current_data.get('MARKET_REGIME_TEXT') == "volatile" else 0
        ])
        
        # Add normalized OHLCV
        close_mean = window_data["Close"].mean()
        obs = np.append(obs, [
            current_data["Open"] / close_mean,
            current_data["High"] / close_mean,
            current_data["Low"] / close_mean,
            current_data["Close"] / close_mean,
            current_data["Volume"] / window_data["Volume"].mean(),
        ])
        
        # Add normalized SMA 5, 10, 20, 50 days
        for period in [5, 10, 20, 50]:
            if f'SMA{period}' in current_data:
                sma = current_data[f'SMA{period}'] / close_mean
            else:
                sma = 1.0  # Default if not available
            obs = np.append(obs, [sma])
        
        # Add EMA 5, 10, 20, 50 days
        for period in [5, 10, 20, 50]:
            if f'EMA{period}' in current_data:
                ema = current_data[f'EMA{period}'] / close_mean
            else:
                ema = 1.0  # Default if not available
            obs = np.append(obs, [ema])
        
        # Add RSI
        if 'RSI' in current_data:
            rsi = current_data['RSI'] / 100  # Normalize RSI
        else:
            rsi = 0.5  # Default middle value if not available
        obs = np.append(obs, [rsi])
        
        # Add MACD
        if 'MACD' in current_data and 'Signal' in current_data:
            macd = current_data['MACD'] / close_mean
            signal = current_data['Signal'] / close_mean
        else:
            macd = 0
            signal = 0
        obs = np.append(obs, [macd, signal])
        
        # Add Bollinger Bands
        if all(x in current_data for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            upper_band = current_data['BB_Upper'] / close_mean
            middle_band = current_data['BB_Middle'] / close_mean
            lower_band = current_data['BB_Lower'] / close_mean
        else:
            upper_band = 1.1
            middle_band = 1.0
            lower_band = 0.9
        obs = np.append(obs, [upper_band, middle_band, lower_band])
        
        # Add price momentum indicators
        if len(window_data) > 5:
            ret_1d = window_data['Close'].pct_change(1).iloc[-1]
            ret_5d = window_data['Close'].pct_change(5).iloc[-1] if len(window_data) > 5 else 0
            ret_20d = window_data['Close'].pct_change(20).iloc[-1] if len(window_data) > 20 else 0
        else:
            ret_1d = 0
            ret_5d = 0
            ret_20d = 0
        obs = np.append(obs, [ret_1d, ret_5d, ret_20d])
        
        # Volume-Based Indicators
        if 'OBV' in current_data:
            max_obv = window_data['OBV'].max() if 'OBV' in window_data.columns else 1
            obv = current_data['OBV'] / max_obv if max_obv != 0 else 0.5
        else:
            obv = 0.5
        obs = np.append(obs, [obv])
        
        # Chaikin Money Flow
        if 'CMF' in current_data:
            cmf = current_data['CMF']  # Already normalized between -1 and 1
            cmf = (cmf + 1) / 2  # Normalize to 0-1 range
        else:
            cmf = 0.5
        obs = np.append(obs, [cmf])
        
        # ADX (Average Directional Index)
        if 'ADX' in current_data:
            adx = current_data['ADX'] / 100  # Normalize ADX to 0-1
        else:
            adx = 0.5
        obs = np.append(obs, [adx])
        
        # Ensure we have exactly 60 elements in the observation
        if len(obs) > 60:
            # Truncate if necessary
            obs = obs[:60]
        elif len(obs) < 60:
            # Pad with zeros if necessary
            obs = np.append(obs, np.zeros(60 - len(obs)))
        
        return obs.astype(np.float32)
    
    def generate_signals(self, symbol, current_data, historical_data):
        """Generate trading signals using the RL model"""
        if self.model is None:
            logger.warning("No model loaded, cannot generate signals")
            return {"signal": "HOLD", "strength": 0.0, "stop_loss": None, "target": None}
        
        # Prepare observation
        obs = self._prepare_observation(symbol, current_data, historical_data)
        
        # Get action from model
        action, _states = self.model.predict(obs, deterministic=True)
        
        # Map action to signal
        # Action space: 0 = Hold, 1-4 = Buy (different sizes), 5 = Sell
        if action == 0:
            signal = "HOLD"
            strength = 0.0
        elif action >= 1 and action <= 4:
            signal = "BUY"
            strength = 0.25 * action  # 0.25, 0.5, 0.75, or 1.0
        elif action == 5:
            signal = "SELL"
            strength = 1.0
        else:
            signal = "HOLD"
            strength = 0.0
        
        # Calculate stop loss and target if buying
        stop_loss = None
        target = None
        
        if signal == "BUY" and 'ATR' in current_data:
            atr = current_data['ATR']
            current_price = current_data['Close']
            
            # Dynamic multipliers based on market regime
            if current_data.get('MARKET_REGIME_TEXT') == "trending_up":
                stop_multiplier = 2.0
                target_multiplier = 6.0
            elif current_data.get('MARKET_REGIME_TEXT') == "trending_down":
                stop_multiplier = 1.8
                target_multiplier = 4.5
            elif current_data.get('MARKET_REGIME_TEXT') == "ranging":
                stop_multiplier = 1.0
                target_multiplier = 3.0
            elif current_data.get('MARKET_REGIME_TEXT') == "volatile":
                stop_multiplier = 2.5
                target_multiplier = 7.0
            else:
                stop_multiplier = 2.0
                target_multiplier = 4.0
            
            stop_loss = current_price - (atr * stop_multiplier)
            target = current_price + (atr * target_multiplier)
        
        return {
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "price": current_data['Close'],
            "stop_loss": stop_loss,
            "target": target,
            "market_regime": current_data.get('MARKET_REGIME_TEXT', 'unknown'),
            "atr": current_data.get('ATR', 0),
            "rsi": current_data.get('RSI', 50),
            "action": int(action),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

class MultiTechnicalStrategy(TradingStrategy):
    """Strategy that combines multiple technical indicators"""
    
    def __init__(self, name="MultiTechnicalStrategy"):
        """Initialize multi-technical strategy"""
        super().__init__(name=name)
        
        # Buy/sell signal thresholds
        self.strong_buy_threshold = 8
        self.buy_threshold = 5
        self.sell_threshold = -5
        self.strong_sell_threshold = -8
        
        # Indicator weights
        self.indicator_weights = {
            'trend': 2.0,  # Higher weight for trend
            'momentum': 1.5,
            'volatility': 1.0,
            'volume': 1.0,
            'pattern': 1.0
        }
        
        logger.info(f"Multi-technical strategy '{name}' initialized")
    
    def _calculate_buy_signal_strength(self, data):
        """Calculate buy signal strength based on technical indicators"""
        buy_signals = 0
        sell_signals = 0
        
        # 1. Trend signals
        if 'EMA20' in data and 'EMA50' in data:
            # Bullish trend
            if data['EMA20'] > data['EMA50']:
                buy_signals += 1 * self.indicator_weights['trend']
                
                # Check for crossover (more recent data needed)
                if 'EMA20_prev' in data and 'EMA50_prev' in data:
                    if data['EMA20_prev'] <= data['EMA50_prev']:
                        buy_signals += 2 * self.indicator_weights['trend']  # Bullish crossover
            else:
                sell_signals += 1 * self.indicator_weights['trend']
                
                # Check for crossover
                if 'EMA20_prev' in data and 'EMA50_prev' in data:
                    if data['EMA20_prev'] >= data['EMA50_prev']:
                        sell_signals += 2 * self.indicator_weights['trend']  # Bearish crossover
        
        # 2. RSI signals
        if 'RSI' in data:
            rsi = data['RSI']
            
            # Oversold zone
            if rsi < 30:
                buy_signals += 2 * self.indicator_weights['momentum']
            # Overbought zone
            elif rsi > 70:
                sell_signals += 2 * self.indicator_weights['momentum']
            # Middle zone buy/sell signals
            elif rsi < 45:
                buy_signals += 0.5 * self.indicator_weights['momentum']
            elif rsi > 55:
                sell_signals += 0.5 * self.indicator_weights['momentum']
        
        # 3. MACD signals
        if 'MACD' in data and 'Signal' in data:
            # MACD above signal line - bullish
            if data['MACD'] > data['Signal']:
                buy_signals += 1 * self.indicator_weights['momentum']
                
                # Check for crossover
                if 'MACD_prev' in data and 'Signal_prev' in data:
                    if data['MACD_prev'] <= data['Signal_prev']:
                        buy_signals += 2 * self.indicator_weights['momentum']  # Bullish crossover
            else:
                sell_signals += 1 * self.indicator_weights['momentum']
                
                # Check for crossover
                if 'MACD_prev' in data and 'Signal_prev' in data:
                    if data['MACD_prev'] >= data['Signal_prev']:
                        sell_signals += 2 * self.indicator_weights['momentum']  # Bearish crossover
        
        # 4. Bollinger Bands signals
        if all(x in data for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            # Price near lower band - potential buy
            if data['Close'] <= data['BB_Lower'] * 1.01:
                buy_signals += 2 * self.indicator_weights['volatility']
            
            # Price near upper band - potential sell
            elif data['Close'] >= data['BB_Upper'] * 0.99:
                sell_signals += 2 * self.indicator_weights['volatility']
            
            # Price crossed over middle band upwards
            if 'Close_prev' in data and data['Close_prev'] < data['BB_Middle'] and data['Close'] > data['BB_Middle']:
                buy_signals += 1 * self.indicator_weights['volatility']
            
            # Price crossed over middle band downwards
            elif 'Close_prev' in data and data['Close_prev'] > data['BB_Middle'] and data['Close'] < data['BB_Middle']:
                sell_signals += 1 * self.indicator_weights['volatility']
        
        # 5. Volume signals
        if 'Volume' in data and 'Volume_SMA20' in data:
            # Volume spike
            if data['Volume'] > data['Volume_SMA20'] * 1.5:
                # Combined with price increase - bullish
                if 'Close_prev' in data and data['Close'] > data['Close_prev']:
                    buy_signals += 2 * self.indicator_weights['volume']
                # Combined with price decrease - bearish
                elif 'Close_prev' in data and data['Close'] < data['Close_prev']:
                    sell_signals += 2 * self.indicator_weights['volume']
        
        # 6. ADX (trend strength)
        if 'ADX' in data:
            adx = data['ADX']
            
            # Strong trend
            if adx > 25:
                # Direction based on other indicators
                if buy_signals > sell_signals:
                    buy_signals += 1 * self.indicator_weights['trend']
                else:
                    sell_signals += 1 * self.indicator_weights['trend']
        
        # 7. Ichimoku Cloud signals
        if all(x in data for x in ['ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'ICHIMOKU_SENKOU_A', 'ICHIMOKU_SENKOU_B']):
            # TK Cross (Tenkan crosses above Kijun)
            if 'ICHIMOKU_TENKAN_prev' in data and 'ICHIMOKU_KIJUN_prev' in data:
                if data['ICHIMOKU_TENKAN_prev'] <= data['ICHIMOKU_KIJUN_prev'] and data['ICHIMOKU_TENKAN'] > data['ICHIMOKU_KIJUN']:
                    buy_signals += 2 * self.indicator_weights['trend']
                elif data['ICHIMOKU_TENKAN_prev'] >= data['ICHIMOKU_KIJUN_prev'] and data['ICHIMOKU_TENKAN'] < data['ICHIMOKU_KIJUN']:
                    sell_signals += 2 * self.indicator_weights['trend']
            
            # Price above cloud - bullish
            if data['Close'] > max(data['ICHIMOKU_SENKOU_A'], data['ICHIMOKU_SENKOU_B']):
                buy_signals += 1 * self.indicator_weights['trend']
            
            # Price below cloud - bearish
            elif data['Close'] < min(data['ICHIMOKU_SENKOU_A'], data['ICHIMOKU_SENKOU_B']):
                sell_signals += 1 * self.indicator_weights['trend']
        
        # 8. Support/Resistance levels
        if 'Support' in data and 'Resistance' in data:
            # Price near support - potential buy
            if data['Support'] > 0 and data['Close'] <= data['Support'] * 1.02:
                buy_signals += 2 * self.indicator_weights['pattern']
            
            # Price near resistance - potential sell
            if data['Resistance'] > 0 and data['Close'] >= data['Resistance'] * 0.98:
                sell_signals += 2 * self.indicator_weights['pattern']
        
        # 9. Stochastic signals
        if 'STOCH_K' in data and 'STOCH_D' in data:
            # Oversold
            if data['STOCH_K'] < 20 and data['STOCH_D'] < 20:
                buy_signals += 1 * self.indicator_weights['momentum']
                
                # Bullish crossover
                if 'STOCH_K_prev' in data and 'STOCH_D_prev' in data:
                    if data['STOCH_K_prev'] <= data['STOCH_D_prev'] and data['STOCH_K'] > data['STOCH_D']:
                        buy_signals += 1 * self.indicator_weights['momentum']
            
            # Overbought
            elif data['STOCH_K'] > 80 and data['STOCH_D'] > 80:
                sell_signals += 1 * self.indicator_weights['momentum']
                
                # Bearish crossover
                if 'STOCH_K_prev' in data and 'STOCH_D_prev' in data:
                    if data['STOCH_K_prev'] >= data['STOCH_D_prev'] and data['STOCH_K'] < data['STOCH_D']:
                        sell_signals += 1 * self.indicator_weights['momentum']
        
        # 10. Supertrend signal
        if 'ST_Trend' in data:
            if data['ST_Trend'] == 1:  # Uptrend
                buy_signals += 2 * self.indicator_weights['trend']
            elif data['ST_Trend'] == -1:  # Downtrend
                sell_signals += 2 * self.indicator_weights['trend']
        
        # Calculate net signal
        net_signal = buy_signals - sell_signals
        
        # Normalize to a -1 to 1 range
        max_possible_signal = 15 * max(self.indicator_weights.values())
        normalized_signal = np.clip(net_signal / max_possible_signal, -1, 1)
        
        return normalized_signal
    
    def _prepare_indicator_data(self, current_data, historical_data):
        """Prepare indicator data including previous values for crossovers"""
        data = current_data.copy()
        
        # Add previous values for crossover detection
        if len(historical_data) >= 2:
            prev_data = historical_data.iloc[-2]
            
            for indicator in ['EMA20', 'EMA50', 'MACD', 'Signal', 'Close', 'ICHIMOKU_TENKAN', 'ICHIMOKU_KIJUN', 'STOCH_K', 'STOCH_D']:
                if indicator in prev_data:
                    data[f'{indicator}_prev'] = prev_data[indicator]
            
            # Add volume SMA for volume analysis
            if 'Volume' in historical_data.columns:
                data['Volume_SMA20'] = historical_data['Volume'].rolling(20).mean().iloc[-1]
        
        return data
    
    def generate_signals(self, symbol, current_data, historical_data):
        """Generate trading signals based on technical indicators"""
        # Prepare data with previous values
        data = self._prepare_indicator_data(current_data, historical_data)
        
        # Calculate signal strength (-1 to 1)
        signal_strength = self._calculate_buy_signal_strength(data)
        
        # Determine signal type based on strength and thresholds
        if signal_strength >= self.strong_buy_threshold / 10:
            signal = "STRONG_BUY"
            strength = 1.0
        elif signal_strength >= self.buy_threshold / 10:
            signal = "BUY"
            strength = 0.7
        elif signal_strength <= self.strong_sell_threshold / 10:
            signal = "STRONG_SELL"
            strength = 1.0
        elif signal_strength <= self.sell_threshold / 10:
            signal = "SELL"
            strength = 0.7
        else:
            signal = "HOLD"
            strength = 0.0
        
        # Calculate stop loss and target if buying
        stop_loss = None
        target = None
        
        if signal in ["BUY", "STRONG_BUY"] and 'ATR' in current_data:
            atr = current_data['ATR']
            current_price = current_data['Close']
            
            # Dynamic multipliers based on market regime
            if 'MARKET_REGIME_TEXT' in current_data:
                regime = current_data['MARKET_REGIME_TEXT']
                
                if regime == "trending_up":
                    stop_multiplier = 2.0
                    target_multiplier = 6.0
                elif regime == "trending_down":
                    stop_multiplier = 1.8
                    target_multiplier = 4.5
                elif regime == "ranging":
                    stop_multiplier = 1.0
                    target_multiplier = 3.0
                elif regime == "volatile":
                    stop_multiplier = 2.5
                    target_multiplier = 7.0
                else:
                    stop_multiplier = 2.0
                    target_multiplier = 4.0
            else:
                # Default multipliers
                stop_multiplier = 2.0
                target_multiplier = 4.0
            
            stop_loss = current_price - (atr * stop_multiplier)
            target = current_price + (atr * target_multiplier)
        
        # Additional trade information
        additional_info = {
            'rsi': data.get('RSI', 50),
            'macd': data.get('MACD', 0),
            'signal_line': data.get('Signal', 0),
            'bb_width': (data.get('BB_Upper', 0) - data.get('BB_Lower', 0)) / data.get('BB_Middle', 1) if 'BB_Middle' in data and data['BB_Middle'] != 0 else 0,
            'adx': data.get('ADX', 0),
            'volume_ratio': data.get('Volume', 0) / data.get('Volume_SMA20', 1) if 'Volume_SMA20' in data and data['Volume_SMA20'] != 0 else 1
        }
        
        return {
            "symbol": symbol,
            "signal": signal,
            "strength": abs(signal_strength),
            "price": current_data['Close'],
            "stop_loss": stop_loss,
            "target": target,
            "market_regime": current_data.get('MARKET_REGIME_TEXT', 'unknown'),
            "atr": current_data.get('ATR', 0),
            "indicators": additional_info,
            "raw_signal_strength": signal_strength,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

class SwingTradingStrategy(TradingStrategy):
    """Strategy focused on swing trading with longer holding periods"""
    
    def __init__(self, name="SwingTradingStrategy"):
        """Initialize swing trading strategy"""
        super().__init__(name=name)
        
        # Configuration
        self.trend_lookback = 50  # Longer trend lookback for swing trading
        self.support_resistance_lookback = 60  # Longer S/R lookback
        
        # Signal thresholds
        self.buy_threshold = 0.6
        self.sell_threshold = -0.6
        
        logger.info(f"Swing trading strategy '{name}' initialized")
    
    def _identify_trend(self, df):
        """Identify the longer-term trend for swing trading"""
        # Use EMAs for trend detection
        if 'EMA20' in df.columns and 'EMA50' in df.columns and 'EMA100' in df.columns:
            # Bullish alignment (fast above slow)
            if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] > df['EMA100'].iloc[-1]:
                # Strong uptrend
                if df['Close'].iloc[-1] > df['EMA20'].iloc[-1]:
                    return 2  # Strong uptrend
                else:
                    return 1  # Moderate uptrend
                    
            # Bearish alignment (fast below slow)
            elif df['EMA20'].iloc[-1] < df['EMA50'].iloc[-1] < df['EMA100'].iloc[-1]:
                # Strong downtrend
                if df['Close'].iloc[-1] < df['EMA20'].iloc[-1]:
                    return -2  # Strong downtrend
                else:
                    return -1  # Moderate downtrend
            
            # Mixed signals - possible range
            else:
                # Check if price is above/below medium-term EMA
                if df['Close'].iloc[-1] > df['EMA50'].iloc[-1]:
                    return 0.5  # Slight bullish bias
                else:
                    return -0.5  # Slight bearish bias
        else:
            # Simplified trend detection using just price action
            window = min(self.trend_lookback, len(df))
            price_start = df['Close'].iloc[-window]
            price_end = df['Close'].iloc[-1]
            
            # Calculate trend based on start vs. end price
            if price_end > price_start * 1.1:  # 10% higher
                return 2  # Strong uptrend
            elif price_end > price_start * 1.03:  # 3% higher
                return 1  # Moderate uptrend
            elif price_end < price_start * 0.9:  # 10% lower
                return -2  # Strong downtrend
            elif price_end < price_start * 0.97:  # 3% lower
                return -1  # Moderate downtrend
            else:
                return 0  # Ranging market
    
    def _identify_key_levels(self, df):
        """Identify key support and resistance levels for swing trading"""
        # Need sufficient data
        if len(df) < self.support_resistance_lookback:
            return None, None
        
        # Subset of data for analysis
        window_data = df.iloc[-self.support_resistance_lookback:]
        
        # Find recent highs and lows
        highs = window_data['High'].values
        lows = window_data['Low'].values
        current_price = df['Close'].iloc[-1]
        
        # Find clusters of highs (resistance) and lows (support)
        resistance_levels = []
        support_levels = []
        
        # Identify clusters using a percentage-based approach
        for i, high in enumerate(highs):
            # Skip if already part of a cluster
            if any(abs(high - r) / r < 0.02 for r in resistance_levels):
                continue
                
            # Count nearby highs
            cluster_count = sum(1 for h in highs if abs(h - high) / high < 0.02)
            
            if cluster_count >= 3:  # At least 3 touches to be significant
                resistance_levels.append(high)
        
        for i, low in enumerate(lows):
            # Skip if already part of a cluster
            if any(abs(low - s) / s < 0.02 for s in support_levels):
                continue
                
            # Count nearby lows
            cluster_count = sum(1 for l in lows if abs(l - low) / low < 0.02)
            
            if cluster_count >= 3:  # At least 3 touches to be significant
                support_levels.append(low)
        
        # Find closest levels to current price
        nearest_resistance = None
        resistance_distance = float('inf')
        
        for level in resistance_levels:
            if level > current_price:
                distance = (level - current_price) / current_price
                if distance < resistance_distance:
                    resistance_distance = distance
                    nearest_resistance = level
        
        nearest_support = None
        support_distance = float('inf')
        
        for level in support_levels:
            if level < current_price:
                distance = (current_price - level) / current_price
                if distance < support_distance:
                    support_distance = distance
                    nearest_support = level
        
        return nearest_support, nearest_resistance
    
    def _check_momentum(self, df):
        """Check momentum indicators for swing trading setup"""
        momentum_score = 0
        
        # RSI
        if 'RSI' in df.columns:
            rsi = df['RSI'].iloc[-1]
            
            # Bullish conditions
            if rsi < 30:  # Oversold
                momentum_score += 2
            elif rsi < 40:
                momentum_score += 1
            # Bearish conditions
            elif rsi > 70:  # Overbought
                momentum_score -= 2
            elif rsi > 60:
                momentum_score -= 1
        
        # MACD
        if 'MACD' in df.columns and 'Signal' in df.columns:
            macd = df['MACD'].iloc[-1]
            signal = df['Signal'].iloc[-1]
            
            # Check for crossovers (need at least 2 data points)
            if len(df) >= 2:
                prev_macd = df['MACD'].iloc[-2]
                prev_signal = df['Signal'].iloc[-2]
                
                # Bullish crossover
                if prev_macd <= prev_signal and macd > signal:
                    momentum_score += 2
                # Bearish crossover
                elif prev_macd >= prev_signal and macd < signal:
                    momentum_score -= 2
            
            # MACD histogram direction
            hist = macd - signal
            
            if len(df) >= 2:
                prev_hist = prev_macd - prev_signal
                
                # Increasing histogram (bullish)
                if hist > prev_hist:
                    momentum_score += 1
                # Decreasing histogram (bearish)
                elif hist < prev_hist:
                    momentum_score -= 1
        
        # Stochastic
        if 'STOCH_K' in df.columns and 'STOCH_D' in df.columns:
            k = df['STOCH_K'].iloc[-1]
            d = df['STOCH_D'].iloc[-1]
            
            # Oversold conditions
            if k < 20 and d < 20:
                momentum_score += 1
                
                # Check for bullish crossover
                if len(df) >= 2:
                    prev_k = df['STOCH_K'].iloc[-2]
                    prev_d = df['STOCH_D'].iloc[-2]
                    
                    if prev_k <= prev_d and k > d:
                        momentum_score += 1
            
            # Overbought conditions
            elif k > 80 and d > 80:
                momentum_score -= 1
                
                # Check for bearish crossover
                if len(df) >= 2:
                    prev_k = df['STOCH_K'].iloc[-2]
                    prev_d = df['STOCH_D'].iloc[-2]
                    
                    if prev_k >= prev_d and k < d:
                        momentum_score -= 1
        
        return momentum_score
    
    def _check_volume_patterns(self, df):
        """Check for swing trading volume patterns"""
        volume_score = 0
        
        if 'Volume' in df.columns and len(df) >= 20:
            # Calculate average volume
            avg_volume = df['Volume'].iloc[-20:].mean()
            current_volume = df['Volume'].iloc[-1]
            
            # Volume vs price relationship
            current_price = df['Close'].iloc[-1]
            prev_price = df['Close'].iloc[-2]
            price_change = current_price - prev_price
            
            # Volume spike (1.5x average)
            if current_volume > avg_volume * 1.5:
                # Volume spike with price increase (bullish)
                if price_change > 0:
                    volume_score += 2
                # Volume spike with price decrease (bearish)
                else:
                    volume_score -= 2
            
            # Diminishing volume on price moves (potential reversal)
            if len(df) >= 3:
                prev_volume = df['Volume'].iloc[-2]
                prev_prev_volume = df['Volume'].iloc[-3]
                
                # Falling volume on up days (bearish divergence)
                if price_change > 0 and prev_price > df['Close'].iloc[-3] and current_volume < prev_volume < prev_prev_volume:
                    volume_score -= 1
                
                # Falling volume on down days (bullish divergence)
                elif price_change < 0 and prev_price < df['Close'].iloc[-3] and current_volume < prev_volume < prev_prev_volume:
                    volume_score += 1
        
        return volume_score
    
    def generate_signals(self, symbol, current_data, historical_data):
        """Generate swing trading signals"""
        # Need sufficient historical data
        if len(historical_data) < max(self.trend_lookback, self.support_resistance_lookback):
            return {
                "symbol": symbol,
                "signal": "HOLD",
                "strength": 0.0,
                "price": current_data['Close'],
                "stop_loss": None,
                "target": None,
                "market_regime": "unknown",
                "reason": "Insufficient historical data for swing analysis"
            }
        
        # 1. Identify trend
        trend_score = self._identify_trend(historical_data)
        
        # 2. Identify key levels
        support, resistance = self._identify_key_levels(historical_data)
        
        # 3. Check momentum
        momentum_score = self._check_momentum(historical_data)
        
        # 4. Check volume patterns
        volume_score = self._check_volume_patterns(historical_data)
        
        # 5. Calculate overall score and determine signal
        current_price = current_data['Close']
        
        # Base score on trend (weight: 2.0)
        overall_score = trend_score * 2.0
        
        # Add momentum signals (weight: 1.5)
        overall_score += momentum_score * 1.5
        
        # Add volume signals (weight: 1.0)
        overall_score += volume_score * 1.0
        
        # Support/resistance proximity analysis (weight: 2.0)
        key_level_score = 0
        
        # Near support (potential buy)
        if support is not None and (current_price - support) / current_price < 0.05:  # Within 5%
            key_level_score += 2
            
            # Very close to support (stronger buy)
            if (current_price - support) / current_price < 0.02:  # Within 2%
                key_level_score += 1
        
        # Near resistance (potential sell)
        if resistance is not None and (resistance - current_price) / current_price < 0.05:  # Within 5%
            key_level_score -= 2
            
            # Very close to resistance (stronger sell)
            if (resistance - current_price) / current_price < 0.02:  # Within 2%
                key_level_score -= 1
        
        # Add key level score to overall score
        overall_score += key_level_score * 2.0
        
        # Normalize score to determine signal strength (-1 to 1 range)
        max_possible_score = 12  # Maximum possible score based on weights
        signal_strength = np.clip(overall_score / max_possible_score, -1, 1)
        
        # Determine signal based on strength and thresholds
        if signal_strength >= self.buy_threshold:
            signal = "BUY"
            strength = abs(signal_strength)
        elif signal_strength <= self.sell_threshold:
            signal = "SELL"
            strength = abs(signal_strength)
        else:
            signal = "HOLD"
            strength = 0.0
        
        # Calculate stop loss and target for buys
        stop_loss = None
        target = None
        
        if signal == "BUY":
            # Default to support level for stop loss if available
            if support is not None:
                stop_loss = support * 0.99  # Slightly below support
            elif 'ATR' in current_data:
                # Or use ATR-based stop if support not available
                atr = current_data['ATR']
                stop_loss = current_price - (atr * 2.5)
            
            # Default to resistance for target if available
            if resistance is not None:
                target = resistance * 0.98  # Slightly below resistance
            elif 'ATR' in current_data:
                # Or use ATR-based target
                atr = current_data['ATR']
                target = current_price + (atr * 5.0)
        
        # Determine market regime
        if 'MARKET_REGIME_TEXT' in current_data:
            market_regime = current_data['MARKET_REGIME_TEXT']
        else:
            # Derive from trend analysis
            if trend_score >= 1.5:
                market_regime = "trending_up"
            elif trend_score <= -1.5:
                market_regime = "trending_down"
            elif abs(trend_score) < 0.8:
                market_regime = "ranging"
            else:
                market_regime = "unknown"
        
        # Build signal information
        signal_info = {
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "price": current_price,
            "stop_loss": stop_loss,
            "target": target,
            "market_regime": market_regime,
            "support": support,
            "resistance": resistance,
            "trend_score": trend_score,
            "momentum_score": momentum_score,
            "volume_score": volume_score,
            "key_level_score": key_level_score,
            "overall_score": overall_score,
            "atr": current_data.get('ATR', 0),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return signal_info