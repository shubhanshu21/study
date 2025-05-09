"""
Market regime detection for trading strategy adaptation.
"""
import numpy as np
import pandas as pd

class MarketRegimeDetector:
    """
    Detects the current market regime (trending, ranging, volatile).
    Adapts trading parameters based on the detected regime.
    """
    
    def __init__(self, lookback=20):
        """
        Initialize the market regime detector
        
        Args:
            lookback (int): Lookback period for regime detection
        """
        self.lookback = lookback
    
    def detect_regime(self, df, current_idx):
        """
        Detect the current market regime
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLC and indicator data
            current_idx (int): Current index in the dataframe
            
        Returns:
            str: Detected market regime
        """
        if df is None or df.empty or current_idx < self.lookback:
            return "unknown"
            
        # Get window of data
        window = df.iloc[max(0, current_idx-self.lookback):current_idx+1]
        
        # Calculate key metrics for regime detection
        adx = window['ADX'].iloc[-1] if 'ADX' in window.columns else 25  # Default to neutral
            
        if 'ATR' in window.columns and 'close' in window.columns:
            atr_pct = window['ATR'].iloc[-1] / window['close'].iloc[-1]
        else:
            atr_pct = 0.015  # Default to 1.5% volatility
            
        rsi = window['RSI'].iloc[-1] if 'RSI' in window.columns else 50  # Default to neutral
        
        # Detect trend direction using multiple indicators
        if 'EMA20' in window.columns and 'EMA50' in window.columns:
            trend_direction = 1 if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1] else -1
        else:
            price_change = window['close'].iloc[-1] / window['close'].iloc[0] - 1
            trend_direction = 1 if price_change > 0 else -1
        
        # Regime classification logic
        if adx > 25 and abs(rsi - 50) > 15:
            regime = "trending"
            
            if trend_direction > 0:
                return "trending_up"
            else:
                return "trending_down"
                
        elif atr_pct > 0.025:  # High volatility threshold (2.5%)
            return "volatile"
            
        else:
            return "ranging"
    
    def get_regime_parameters(self, regime):
        """
        Get optimal parameters for the current market regime
        
        Args:
            regime (str): Detected market regime
            
        Returns:
            dict: Dictionary of optimized parameters for the regime
        """
        params = {
            "trending_up": {
                "trailing_stop_atr_multiplier": 2.0,
                "target_atr_multiplier": 6.0,
                "position_size_pct": 0.35,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            },
            "trending_down": {
                "trailing_stop_atr_multiplier": 1.8,
                "target_atr_multiplier": 4.5,
                "position_size_pct": 0.20,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.8
            },
            "ranging": {
                "trailing_stop_atr_multiplier": 1.0,
                "target_atr_multiplier": 3.0,
                "position_size_pct": 0.25,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            },
            "volatile": {
                "trailing_stop_atr_multiplier": 2.5,
                "target_atr_multiplier": 7.0,
                "position_size_pct": 0.18,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.8
            },
            "unknown": {
                "trailing_stop_atr_multiplier": 1.8,
                "target_atr_multiplier": 4.0,
                "position_size_pct": 0.20,
                "use_trailing_stop": True,
                "min_rr_ratio": 1.5
            }
        }
        
        return params.get(regime, params["unknown"])