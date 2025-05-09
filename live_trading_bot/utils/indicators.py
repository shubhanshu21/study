"""
Technical indicators for trading.
"""
import numpy as np
import pandas as pd

def calculate_indicators(df):
    """
    Calculate technical indicators for a dataframe
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        
    Returns:
        pandas.DataFrame: DataFrame with added indicators
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure columns are lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # SMA - Simple Moving Average
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'SMA{period}'] = df['close'].rolling(window=period).mean()
    
    # EMA - Exponential Moving Average
    for period in [5, 10, 20, 50, 100, 200]:
        df[f'EMA{period}'] = df['close'].ewm(span=period, adjust=False).mean()
    
    # RSI - Relative Strength Index
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    df['RSI'] = calculate_rsi(df['close'], period=14)
    
    # MACD - Moving Average Convergence Divergence
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    
    # Bollinger Bands
    def calculate_bollinger_bands(series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['close'])
    
    # ATR - Average True Range
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    df['ATR'] = calculate_atr(df['high'], df['low'], df['close'])
    
    # Stochastic Oscillator
    def calculate_stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        return k, d
    
    df['STOCH_K'], df['STOCH_D'] = calculate_stochastic(df['high'], df['low'], df['close'])
    
    # ADX - Average Directional Index
    def calculate_adx(high, low, close, period=14):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(window=period).mean()
        
        return adx, plus_di, minus_di
    
    df['ADX'], df['PLUS_DI'], df['MINUS_DI'] = calculate_adx(df['high'], df['low'], df['close'])
    
    # OBV - On-Balance Volume
    def calculate_obv(close, volume):
        obv = volume.copy()
        obv[close < close.shift()] = -volume
        obv[close == close.shift()] = 0
        return obv.cumsum()
    
    if 'volume' in df.columns:
        df['OBV'] = calculate_obv(df['close'], df['volume'])
    
    # Ichimoku Cloud
    def calculate_ichimoku(high, low, close):
        tenkan_period = 9
        kijun_period = 26
        senkou_b_period = 52
        
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2
        
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2
        
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2).shift(kijun_period)
        
        chikou = close.shift(-kijun_period)
        
        return tenkan, kijun, senkou_a, senkou_b, chikou
    
    (df['ICHIMOKU_TENKAN'], df['ICHIMOKU_KIJUN'], 
     df['ICHIMOKU_SENKOU_A'], df['ICHIMOKU_SENKOU_B'], 
     df['ICHIMOKU_CHIKOU']) = calculate_ichimoku(df['high'], df['low'], df['close'])
    
    # ROC - Rate of Change
    df['ROC5'] = df['close'].pct_change(periods=5) * 100
    df['ROC10'] = df['close'].pct_change(periods=10) * 100
    df['ROC20'] = df['close'].pct_change(periods=20) * 100
    
    # Fill NaN values with forward fill then backward fill
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    
    return df