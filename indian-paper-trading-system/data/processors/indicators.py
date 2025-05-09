import numpy as np
import pandas as pd
import logging
from finta import TA

logger = logging.getLogger(__name__)

class IndicatorProcessor:
    """Calculate and add technical indicators to market data"""
    
    def __init__(self):
        """Initialize IndicatorProcessor"""
        logger.info("Initializing IndicatorProcessor")
    
    def calculate_all_indicators(self, df):
        """Calculate all technical indicators for a dataframe"""
        logger.info("Calculating all technical indicators...")
        
        # First, ensure we have the required OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
        
        try:
            # 1. Basic indicators
            # Moving averages
            df = self.add_moving_averages(df)
            
            # 2. Momentum indicators
            df = self.add_momentum_indicators(df)
            
            # 3. Volatility indicators
            df = self.add_volatility_indicators(df)
            
            # 4. Volume indicators
            df = self.add_volume_indicators(df)
            
            # 5. Trend indicators
            df = self.add_trend_indicators(df)
            
            # 6. Oscillator indicators
            df = self.add_oscillator_indicators(df)
            
            # 7. Advanced indicators (Ichimoku, etc.)
            df = self.add_advanced_indicators(df)
            
            # 8. Support/Resistance levels
            df = self.add_support_resistance(df)
            
            # 9. Market regime detection
            df = self.add_market_regime(df)
            
            # Fill any remaining NaN values
            df.bfill(inplace=True)
            df.ffill(inplace=True)
            
            logger.info("All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            raise
    
    def add_moving_averages(self, df):
        """Add moving average indicators to dataframe"""
        try:
            # Simple Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA{period}'] = TA.SMA(df, period=period)
            
            # Exponential Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'EMA{period}'] = TA.EMA(df, period=period)
            
            # Weighted Moving Average
            df['WMA20'] = TA.WMA(df, period=20)
            
            # Hull Moving Average
            df['HMA20'] = TA.HMA(df, period=20)
            
            # Volume-Weighted Moving Average
            df['VWMA20'] = TA.VWMA(df, period=20)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding moving averages: {str(e)}")
            raise
    
    def add_momentum_indicators(self, df):
        """Add momentum indicators to dataframe"""
        try:
            # Relative Strength Index
            df['RSI'] = TA.RSI(df, period=14)
            
            # Moving Average Convergence Divergence
            macd = TA.MACD(df)
            df['MACD'] = macd['MACD']
            df['Signal'] = macd['SIGNAL']
            df['MACD_Hist'] = df['MACD'] - df['Signal']
            
            # Rate of Change
            df['ROC5'] = TA.ROC(df, period=5)
            df['ROC10'] = TA.ROC(df, period=10)
            df['ROC20'] = TA.ROC(df, period=20)
            
            # Stochastic Oscillator
            stoch = TA.STOCH(df)
            df['STOCH_K'] = stoch
            df['STOCH_D'] = stoch.rolling(3).mean()
            
            # Williams %R
            df['WILLIAMS'] = TA.WILLIAMS(df)
            
            # Commodity Channel Index
            df['CCI'] = TA.CCI(df)
            
            # Ultimate Oscillator
            df['UO'] = TA.UO(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding momentum indicators: {str(e)}")
            raise
    
    def add_volatility_indicators(self, df):
        """Add volatility indicators to dataframe"""
        try:
            # Bollinger Bands
            bollinger = TA.BBANDS(df)
            df['BB_Upper'] = bollinger['BB_UPPER']
            df['BB_Middle'] = bollinger['BB_MIDDLE']
            df['BB_Lower'] = bollinger['BB_LOWER']
            
            # Calculate BB width
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Average True Range
            df['ATR'] = TA.ATR(df, period=14)
            
            # Relative ATR (ATR as % of price)
            df['RATR'] = df['ATR'] / df['Close'] * 100
            
            # Keltner Channels
            keltner = TA.KELTNER(df)
            df['KC_Upper'] = keltner['KC_UPPER']
            df['KC_Middle'] = keltner['KC_MIDDLE']
            df['KC_Lower'] = keltner['KC_LOWER']
            
            # Volatility ratio
            df['Volatility_Ratio'] = df['BB_Width'] / (df['BB_Width'].rolling(20).mean()) if 'BB_Width' in df.columns else np.nan
            
            # Standard Deviation
            df['STD20'] = df['Close'].rolling(20).std()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volatility indicators: {str(e)}")
            raise
    
    def add_volume_indicators(self, df):
        """Add volume indicators to dataframe"""
        try:
            # On-Balance Volume
            df['OBV'] = TA.OBV(df)
            
            # Chaikin Money Flow
            df['CMF'] = TA.CMF(df)
            
            # Money Flow Index
            df['MFI'] = TA.MFI(df)
            
            # Volume Weighted Average Price
            df['VWAP'] = TA.VWAP(df)
            
            # Price-Volume Trend
            df['PVT'] = TA.PVT(df)
            
            # Negative Volume Index
            # (Custom implementation as Finta doesn't have NVI)
            df['NVI'] = self._calculate_nvi(df)
            
            # Relative Volume
            df['Rel_Volume'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding volume indicators: {str(e)}")
            raise
    
    def add_trend_indicators(self, df):
        """Add trend indicators to dataframe"""
        try:
            # Average Directional Index
            adx = TA.ADX(df)
            df['ADX'] = adx
            
            # Parabolic SAR
            df['SAR'] = TA.SAR(df)
            
            # Aroon Oscillator
            df['AROON_OSC'] = TA.AROON_OSC(df)
            
            # Directional Movement Index
            dmi = TA.DMI(df)
            df['DI+'] = dmi['DI+']
            df['DI-'] = dmi['DI-']
            
            # Trend Intensity Index (custom)
            df['TII'] = self._calculate_tii(df)
            
            # Vortex Indicator
            vortex = TA.VORTEX(df)
            df['VI+'] = vortex['VI+']
            df['VI-'] = vortex['VI-']
            
            # SuperTrend (custom calculation)
            df = self._calculate_supertrend(df, atr_multiplier=3.0, atr_period=10)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding trend indicators: {str(e)}")
            raise
    
    def add_oscillator_indicators(self, df):
        """Add oscillator indicators to dataframe"""
        try:
            # Awesome Oscillator
            df['AO'] = TA.AO(df)
            
            # True Strength Index
            df['TSI'] = TA.TSI(df)
            
            # Fisher Transform
            df['FISHER'] = TA.FISHER(df)
            
            # Stochastic RSI
            df['STOCH_RSI'] = TA.STOCHRSI(df)
            
            # McGinley Dynamic
            df['MCGD'] = self._calculate_mcginley_dynamic(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding oscillator indicators: {str(e)}")
            raise
    
    def add_advanced_indicators(self, df):
        """Add advanced indicators to dataframe"""
        try:
            # Ichimoku Cloud components
            ichimoku = TA.ICHIMOKU(df)
            
            df['ICHIMOKU_TENKAN'] = ichimoku['TENKAN']
            df['ICHIMOKU_KIJUN'] = ichimoku['KIJUN']
            df['ICHIMOKU_SENKOU_A'] = ichimoku['senkou_span_a']  # lowercase in Finta
            df['ICHIMOKU_SENKOU_B'] = ichimoku['SENKOU']
            df['ICHIMOKU_CHIKOU'] = ichimoku['CHIKOU']
            
            # Calculate Ichimoku cloud status (above/below)
            df['CLOUD_STATUS'] = np.where(df['Close'] > df['ICHIMOKU_SENKOU_A'], 
                                      np.where(df['Close'] > df['ICHIMOKU_SENKOU_B'], 1, 0),
                                      np.where(df['Close'] < df['ICHIMOKU_SENKOU_B'], -1, 0))
            
            # KST (Know Sure Thing) Oscillator
            df['KST'] = TA.KST(df)
            
            # Elder Ray Index
            df['BULL_POWER'] = df['High'] - TA.EMA(df, period=13)
            df['BEAR_POWER'] = df['Low'] - TA.EMA(df, period=13)
            
            # Add TK Cross detection
            df['TK_CROSS'] = np.where((df['ICHIMOKU_TENKAN'].shift(1) < df['ICHIMOKU_KIJUN'].shift(1)) &
                                 (df['ICHIMOKU_TENKAN'] > df['ICHIMOKU_KIJUN']), 1,
                                 np.where((df['ICHIMOKU_TENKAN'].shift(1) > df['ICHIMOKU_KIJUN'].shift(1)) &
                                      (df['ICHIMOKU_TENKAN'] < df['ICHIMOKU_KIJUN']), -1, 0))
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding advanced indicators: {str(e)}")
            raise
    
    def add_support_resistance(self, df, window=20, threshold=0.02):
        """Detect support and resistance levels"""
        try:
            # Initialize support and resistance columns
            df['Support'] = np.nan
            df['Resistance'] = np.nan
            
            for i in range(window, len(df)):
                # Get current window
                window_data = df.iloc[i-window:i]
                
                # Find local minima and maxima
                local_min = window_data['Low'].min()
                local_max = window_data['High'].max()
                
                # Check if current price is near support/resistance
                current_price = df.iloc[i]['Close']
                
                # Support
                if abs(current_price - local_min) / local_min < threshold:
                    df.loc[df.index[i], 'Support'] = local_min
                
                # Resistance
                if abs(local_max - current_price) / current_price < threshold:
                    df.loc[df.index[i], 'Resistance'] = local_max
            
            # Forward fill support/resistance levels
            df['Support'] = df['Support'].ffill()
            df['Resistance'] = df['Resistance'].ffill()
            
            # Calculate distance to nearest levels (as % of price)
            df['Support_Distance'] = (df['Close'] - df['Support']) / df['Close'] * 100
            df['Resistance_Distance'] = (df['Resistance'] - df['Close']) / df['Close'] * 100
            
            return df
            
        except Exception as e:
            logger.error(f"Error detecting support and resistance: {str(e)}")
            raise
    
    def add_market_regime(self, df, lookback=20):
        """Detect market regime and add regime indicator"""
        try:
            # Ensure we have ADX for trend strength
            if 'ADX' not in df.columns:
                adx = TA.ADX(df)
                df['ADX'] = adx
                
            # Ensure we have ATR for volatility
            if 'ATR' not in df.columns:
                df['ATR'] = TA.ATR(df, period=14)
                
            # Ensure we have RSI
            if 'RSI' not in df.columns:
                df['RSI'] = TA.RSI(df, period=14)
                
            # Calculate ATR as % of price (volatility)
            df['ATR_PCT'] = df['ATR'] / df['Close']
            
            # Initialize regime column
            df['MARKET_REGIME'] = np.nan
            
            # Define regimes numerically (for easier ML usage)
            # 1: Trending Up, 2: Trending Down, 3: Ranging, 4: Volatile
            for i in range(lookback, len(df)):
                window = df.iloc[i-lookback:i+1]
                
                # Trend direction using EMAs
                if 'EMA20' in window.columns and 'EMA50' in window.columns:
                    trend_direction = 1 if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1] else -1
                else:
                    price_change = window['Close'].iloc[-1] / window['Close'].iloc[0] - 1
                    trend_direction = 1 if price_change > 0 else -1
                
                # Get key metrics
                adx = window['ADX'].iloc[-1]
                atr_pct = window['ATR_PCT'].iloc[-1]
                rsi = window['RSI'].iloc[-1]
                
                # Regime classification logic
                if adx > 25 and abs(rsi - 50) > 15:
                    # Trending regime
                    if trend_direction > 0:
                        df.loc[df.index[i], 'MARKET_REGIME'] = 1  # Trending Up
                    else:
                        df.loc[df.index[i], 'MARKET_REGIME'] = 2  # Trending Down
                elif atr_pct > 0.025:  # High volatility threshold (2.5%)
                    df.loc[df.index[i], 'MARKET_REGIME'] = 4  # Volatile
                else:
                    df.loc[df.index[i], 'MARKET_REGIME'] = 3  # Ranging
            
            # Forward fill regime
            df['MARKET_REGIME'] = df['MARKET_REGIME'].ffill().fillna(3)  # Default to ranging
            
            # Add a text version of the regime for readability
            regime_map = {
                1: 'trending_up',
                2: 'trending_down',
                3: 'ranging',
                4: 'volatile'
            }
            df['MARKET_REGIME_TEXT'] = df['MARKET_REGIME'].map(regime_map)
            
            return df
        
        except Exception as e:
            logger.error(f"Error adding market regime: {str(e)}")
            raise