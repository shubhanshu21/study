import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import pytz
from kiteconnect import KiteConnect
import time
import json
from config.trading_config import (
    API_KEY, API_SECRET, ACCESS_TOKEN, HISTORICAL_LOOKBACK_DAYS, 
    TICKS_LOOKBACK_MINUTES, CANDLE_TIMEFRAME
)
from config.market_config import MarketConfig

logger = logging.getLogger(__name__)

class ZerodhaDataFetcher:
    """Fetches market data from Zerodha's Kite Connect API"""
    
    def __init__(self, api_key=API_KEY, api_secret=API_SECRET, access_token=ACCESS_TOKEN):
        """Initialize ZerodhaDataFetcher with API credentials"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        self.today = datetime.now(self.timezone).date()
        self.token_mapping = {}  # Map symbol to instrument token
        self.initialized = False
        
        # Connect to Zerodha
        self._connect()
        
    def _connect(self):
        """Establish connection to Kite Connect"""
        try:
            logger.info("Connecting to Zerodha Kite Connect...")
            self.kite = KiteConnect(api_key=self.api_key)
            
            # Set access token if provided
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                logger.info("Kite Connect session established with provided access token")
            else:
                # Generate session (this would typically be done through a separate flow)
                logger.warning("No access token provided. Authentication flow required.")
                # In production, you'd need to implement the authentication flow
            
            # Initialize instrument tokens
            self._initialize_instruments()
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Failed to connect to Zerodha Kite Connect: {str(e)}")
            raise
    
    def _initialize_instruments(self):
        """Initialize instrument tokens mapping for symbols"""
        try:
            # Get all NSE instruments
            all_instruments = self.kite.instruments("NSE")
            
            # Create mapping of trading symbol to instrument token
            for instrument in all_instruments:
                # Store both exact symbol and symbol without NSE suffix
                self.token_mapping[instrument['tradingsymbol']] = instrument['instrument_token']
                
                # Also store without exchange info for easier lookup
                if 'NSE' in instrument['tradingsymbol']:
                    clean_symbol = instrument['tradingsymbol'].replace('-NSE', '')
                    self.token_mapping[clean_symbol] = instrument['instrument_token']
            
            logger.info(f"Initialized {len(self.token_mapping)} instrument tokens")
            
        except Exception as e:
            logger.error(f"Error initializing instruments: {str(e)}")
            raise
    
    def _get_instrument_token(self, symbol):
        """Get instrument token for a symbol"""
        # Try direct mapping first
        token = self.token_mapping.get(symbol)
        
        # If not found, try with NSE suffix
        if not token:
            token = self.token_mapping.get(f"{symbol}-NSE")
            
        # If still not found, try without any suffix
        if not token:
            # Get base symbol without any indices
            base_symbol = symbol.split('-')[0]
            token = self.token_mapping.get(base_symbol)
        
        if not token:
            logger.error(f"Could not find instrument token for symbol: {symbol}")
            raise ValueError(f"Unknown symbol: {symbol}")
            
        return token
    
    def fetch_historical_data(self, symbol, interval=CANDLE_TIMEFRAME, days=HISTORICAL_LOOKBACK_DAYS):
        """Fetch historical OHLCV data for a symbol"""
        if not self.initialized:
            self._connect()
            
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol)
            
            # Calculate date range
            to_date = datetime.now(self.timezone)
            from_date = to_date - timedelta(days=days)
            
            # Fetch data from Kite
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval
            )
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Add symbol column
            df['Symbol'] = symbol
            
            # Ensure Date is in correct timezone
            if isinstance(df['Date'].iloc[0], str):
                df['Date'] = pd.to_datetime(df['Date'])
                
            # Extract time from date
            df['Time'] = df['Date'].dt.time
                
            logger.info(f"Fetched {len(df)} historical records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    def fetch_live_data(self, symbols, mode="full"):
        """Fetch live market data for a list of symbols
        
        Args:
            symbols: List of symbol strings
            mode: 'full' for OHLCV data, 'quote' for just the latest quotes
        """
        if not self.initialized:
            self._connect()
            
        try:
            # Get instrument tokens for all symbols
            instrument_tokens = [self._get_instrument_token(symbol) for symbol in symbols]
            
            if mode == "full":
                # Fetch OHLCV data for the current day
                data = {}
                for symbol, token in zip(symbols, instrument_tokens):
                    # Get data for today
                    from_date = datetime.now(self.timezone).replace(hour=9, minute=0, second=0)
                    to_date = datetime.now(self.timezone)
                    
                    # Fetch intraday data
                    symbol_data = self.kite.historical_data(
                        instrument_token=token,
                        from_date=from_date.strftime("%Y-%m-%d %H:%M:%S"),
                        to_date=to_date.strftime("%Y-%m-%d %H:%M:%S"),
                        interval=CANDLE_TIMEFRAME
                    )
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(symbol_data)
                    
                    # Rename columns to standard format
                    if not df.empty:
                        df.rename(columns={
                            'date': 'Date',
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume'
                        }, inplace=True)
                        
                        # Add symbol column
                        df['Symbol'] = symbol
                        
                        # Ensure Date is in correct timezone
                        if isinstance(df['Date'].iloc[0], str):
                            df['Date'] = pd.to_datetime(df['Date'])
                            
                        # Extract time from date
                        df['Time'] = df['Date'].dt.time
                        
                        data[symbol] = df
                    
                return data
            
            elif mode == "quote":
                # Fetch just the latest quotes
                quotes = self.kite.quote(instrument_tokens)
                
                # Process quotes into a clean format
                processed_quotes = {}
                for token, symbol in zip(instrument_tokens, symbols):
                    if str(token) in quotes:
                        quote = quotes[str(token)]
                        processed_quotes[symbol] = {
                            'Symbol': symbol,
                            'LTP': quote['last_price'],
                            'Change': quote['change'],
                            'Bid': quote['depth']['buy'][0]['price'] if quote['depth']['buy'] else None,
                            'Ask': quote['depth']['sell'][0]['price'] if quote['depth']['sell'] else None,
                            'Volume': quote['volume'],
                            'OHLC': {
                                'Open': quote['ohlc']['open'],
                                'High': quote['ohlc']['high'],
                                'Low': quote['ohlc']['low'],
                                'Close': quote['ohlc']['close']
                            },
                            'Timestamp': quote['timestamp']
                        }
                
                return processed_quotes
            
            else:
                raise ValueError(f"Invalid mode: {mode}. Choose 'full' or 'quote'.")
                
        except Exception as e:
            logger.error(f"Error fetching live data: {str(e)}")
            raise
    
    def fetch_order_book(self, symbol):
        """Fetch order book (market depth) for a symbol"""
        if not self.initialized:
            self._connect()
            
        try:
            # Get instrument token
            instrument_token = self._get_instrument_token(symbol)
            
            # Fetch market depth
            depth = self.kite.market_depth(instrument_token)
            return depth
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            raise
    
    def get_instrument_margins(self, symbols):
        """Get margin requirements for instruments"""
        if not self.initialized:
            self._connect()
            
        try:
            # Get instrument tokens
            instrument_tokens = [self._get_instrument_token(symbol) for symbol in symbols]
            
            # Create the list of instruments for margin calculation
            instruments = []
            for symbol, token in zip(symbols, instrument_tokens):
                instruments.append({
                    "exchange": "NSE",
                    "tradingsymbol": symbol,
                    "quantity": 1  # We'll get margin per unit
                })
            
            # Get margin requirements
            margins = self.kite.margin_required(
                tradingsymbol=instruments
            )
            
            return margins
            
        except Exception as e:
            logger.error(f"Error fetching margin requirements: {str(e)}")
            raise
    
    def save_historical_data(self, symbol, data, directory="data/historical"):
        """Save historical data to CSV file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(directory, exist_ok=True)
            
            # Save to CSV
            file_path = os.path.join(directory, f"{symbol}.csv")
            data.to_csv(file_path, index=False)
            logger.info(f"Saved historical data for {symbol} to {file_path}")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving historical data for {symbol}: {str(e)}")
            raise
    
    def load_historical_data(self, symbol, directory="data/historical"):
        """Load historical data from CSV file"""
        try:
            # Create file path
            file_path = os.path.join(directory, f"{symbol}.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"Historical data file not found for {symbol} at {file_path}")
                return None
            
            # Load from CSV
            df = pd.read_csv(file_path)
            
            # Convert Date column to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                
            # Extract time from date if not present
            if 'Time' not in df.columns and 'Date' in df.columns:
                df['Time'] = df['Date'].dt.time
            
            logger.info(f"Loaded historical data for {symbol} from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {str(e)}")
            raise
    
    def update_historical_data(self, symbol, directory="data/historical"):
        """Update historical data file with latest data"""
        try:
            # Load existing data
            existing_data = self.load_historical_data(symbol, directory)
            
            if existing_data is not None:
                # Get the latest date in existing data
                latest_date = existing_data['Date'].max()
                
                # Calculate days to fetch
                days_diff = (datetime.now(self.timezone).date() - latest_date.date()).days + 1
                
                # Fetch only missing data
                if days_diff > 0:
                    new_data = self.fetch_historical_data(symbol, days=days_diff)
                    
                    # Combine data
                    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                    
                    # Remove duplicates
                    combined_data.drop_duplicates(subset=['Date'], keep='last', inplace=True)
                    
                    # Sort by date
                    combined_data.sort_values('Date', inplace=True)
                    
                    # Save updated data
                    self.save_historical_data(symbol, combined_data, directory)
                    logger.info(f"Updated historical data for {symbol} with {len(new_data)} new records")
                    
                    return combined_data
                else:
                    logger.info(f"Historical data for {symbol} is already up to date")
                    return existing_data
            else:
                # No existing data, fetch all
                new_data = self.fetch_historical_data(symbol)
                self.save_historical_data(symbol, new_data, directory)
                logger.info(f"Created new historical data file for {symbol} with {len(new_data)} records")
                return new_data
                
        except Exception as e:
            logger.error(f"Error updating historical data for {symbol}: {str(e)}")
            raise