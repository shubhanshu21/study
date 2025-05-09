"""
Data loading and processing for trading.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ZerodhaDataLoader:
    """
    Data loader for Zerodha trading platform.
    Handles loading historical data and latest candles.
    """
    
    def __init__(self, broker, symbols):
        """
        Initialize the data loader
        
        Args:
            broker: Broker instance
            symbols (list): List of trading symbols
        """
        self.broker = broker
        self.symbols = symbols
        self.data_cache = {}  # Cache for historical data
        self.latest_candle_time = {}  # Track latest candle time for each symbol
        
        logger.info(f"Initialized data loader for {len(symbols)} symbols")
    
    def get_historical_data(self, symbol, from_date, to_date, interval='5minute'):
        """
        Get historical OHLC data for a symbol
        
        Args:
            symbol (str): Trading symbol
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            interval (str): Candle interval (minute, day, 5minute, etc.)
            
        Returns:
            pandas.DataFrame: DataFrame with OHLC data
        """
        try:
            # Convert dates to datetime objects if they are strings
            if isinstance(from_date, str):
                from_date = datetime.strptime(from_date, '%Y-%m-%d')
            if isinstance(to_date, str):
                to_date = datetime.strptime(to_date, '%Y-%m-%d')
                
            # Get historical data from broker
            data = self.broker.get_historical_data(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if not data:
                logger.warning(f"No historical data found for {symbol}")
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have datetime column
            if 'date' in df.columns and not isinstance(df['date'].iloc[0], datetime):
                df['date'] = pd.to_datetime(df['date'])
                
            # Set date as index
            if 'date' in df.columns:
                df.set_index('date', inplace=True)
            
            # Cache the data
            self.data_cache[symbol] = df
            
            # Update latest candle time
            if not df.empty:
                self.latest_candle_time[symbol] = df.index[-1] if isinstance(df.index[-1], datetime) else df.index[-1].to_pydatetime()
            
            logger.info(f"Loaded historical data for {symbol}: {len(df)} candles")
            return df
            
        except Exception as e:
            logger.exception(f"Error loading historical data for {symbol}: {str(e)}")
            return None
    
    def get_latest_candle(self, symbol, interval='5minute'):
        """
        Get the latest candle for a symbol
        
        Args:
            symbol (str): Trading symbol
            interval (str): Candle interval
            
        Returns:
            dict: Latest candle data
        """
        try:
            # Get current time
            now = datetime.now()
            
            # Calculate from_date based on interval to get a few candles
            if interval == '1minute' or interval == 'minute':
                from_date = now - timedelta(minutes=10)
            elif interval == '5minute':
                from_date = now - timedelta(minutes=30)
            elif interval == '15minute':
                from_date = now - timedelta(minutes=90)
            elif interval == '30minute':
                from_date = now - timedelta(minutes=180)
            elif interval == '60minute' or interval == 'hour':
                from_date = now - timedelta(hours=6)
            else:
                from_date = now - timedelta(days=5)
            
            # Get recent data
            data = self.broker.get_historical_data(
                symbol=symbol,
                from_date=from_date,
                to_date=now,
                interval=interval
            )
            
            if not data or len(data) == 0:
                logger.warning(f"No recent data found for {symbol}")
                return None
                
            # Get the latest candle
            latest_candle = data[-1]
            
            # Check if this is a new candle
            latest_time = latest_candle.get('date', None)
            if latest_time:
                if symbol in self.latest_candle_time and latest_time <= self.latest_candle_time[symbol]:
                    # Not a new candle, just update the existing one
                    if symbol in self.data_cache and not self.data_cache[symbol].empty:
                        self.data_cache[symbol].iloc[-1] = latest_candle
                        logger.debug(f"Updated existing candle for {symbol}")
                        return latest_candle
                
                # New candle, update cache
                self.latest_candle_time[symbol] = latest_time
                
                if symbol in self.data_cache and not self.data_cache[symbol].empty:
                    # Append to existing cache
                    new_row = pd.DataFrame([latest_candle])
                    if 'date' in new_row.columns:
                        new_row.set_index('date', inplace=True)
                    self.data_cache[symbol] = pd.concat([self.data_cache[symbol], new_row])
                    
                    # Keep cache size reasonable
                    if len(self.data_cache[symbol]) > 500:
                        self.data_cache[symbol] = self.data_cache[symbol].iloc[-500:]
                
                logger.debug(f"Got new candle for {symbol}")
            
            return latest_candle
            
        except Exception as e:
            logger.exception(f"Error getting latest candle for {symbol}: {str(e)}")
            return None
    
    def get_last_n_candles(self, symbol, n=100, interval='5minute'):
        """
        Get the last N candles for a symbol
        
        Args:
            symbol (str): Trading symbol
            n (int): Number of candles to get
            interval (str): Candle interval
            
        Returns:
            pandas.DataFrame: DataFrame with the last N candles
        """
        try:
            # Check if we have cached data
            if symbol in self.data_cache and len(self.data_cache[symbol]) >= n:
                return self.data_cache[symbol].iloc[-n:]
            
            # Calculate how far back to go based on interval
            now = datetime.now()
            
            if interval == '1minute' or interval == 'minute':
                from_date = now - timedelta(minutes=n * 2)
            elif interval == '5minute':
                from_date = now - timedelta(minutes=n * 10)
            elif interval == '15minute':
                from_date = now - timedelta(minutes=n * 30)
            elif interval == '30minute':
                from_date = now - timedelta(minutes=n * 60)
            elif interval == '60minute' or interval == 'hour':
                from_date = now - timedelta(hours=n * 2)
            else:
                from_date = now - timedelta(days=n * 2)
            
            # Get historical data
            return self.get_historical_data(
                symbol=symbol,
                from_date=from_date,
                to_date=now,
                interval=interval
            )
            
        except Exception as e:
            logger.exception(f"Error getting last {n} candles for {symbol}: {str(e)}")
            return None
    
    def get_minute_data(self, symbol, days=5):
        """
        Get 1-minute data for a symbol for the last N days
        
        Args:
            symbol (str): Trading symbol
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame with 1-minute data
        """
        now = datetime.now()
        from_date = now - timedelta(days=days)
        
        return self.get_historical_data(
            symbol=symbol,
            from_date=from_date,
            to_date=now,
            interval='minute'
        )
    
    def get_daily_data(self, symbol, days=365):
        """
        Get daily data for a symbol for the last N days
        
        Args:
            symbol (str): Trading symbol
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame with daily data
        """
        now = datetime.now()
        from_date = now - timedelta(days=days)
        
        return self.get_historical_data(
            symbol=symbol,
            from_date=from_date,
            to_date=now,
            interval='day'
        )