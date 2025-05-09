import pandas as pd
import numpy as np
import logging
import os
import json
import requests
from datetime import datetime, timedelta
import pytz
import time
from config.market_config import MarketConfig
from io import StringIO

logger = logging.getLogger(__name__)

class NSEDataFetcher:
    """Fetches market data directly from NSE website and APIs"""
    
    def __init__(self):
        """Initialize NSEDataFetcher"""
        self.market_config = MarketConfig()
        self.timezone = self.market_config.timezone
        self.today = datetime.now(self.timezone).date()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.nseindia.com/'
        }
        self.session = requests.Session()
        self.cookie_expiry = None
        self.refresh_session()
        
    def refresh_session(self):
        """Refresh NSE session and cookies"""
        try:
            logger.info("Refreshing NSE session...")
            
            # Clear existing cookies
            self.session.cookies.clear()
            
            # Visit homepage to get cookies
            response = self.session.get('https://www.nseindia.com/', headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Set cookie expiry (NSE cookies expire in 15 minutes)
            self.cookie_expiry = datetime.now() + timedelta(minutes=14)
            
            logger.info("NSE session refreshed successfully")
            
        except Exception as e:
            logger.error(f"Failed to refresh NSE session: {str(e)}")
            raise
    
    def _check_session(self):
        """Check if session needs to be refreshed"""
        if self.cookie_expiry is None or datetime.now() > self.cookie_expiry:
            self.refresh_session()
    
    def fetch_nifty_indices(self):
        """Fetch Nifty indices data"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/allIndices"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract indices data
            indices = pd.DataFrame(data['data'])
            
            # Format date and time
            indices['timestamp'] = pd.to_datetime(indices['timestamp'])
            indices['Date'] = indices['timestamp'].dt.date
            indices['Time'] = indices['timestamp'].dt.time
            
            return indices
            
        except Exception as e:
            logger.error(f"Error fetching Nifty indices: {str(e)}")
            # Try to refresh session and retry once
            self.refresh_session()
            try:
                url = "https://www.nseindia.com/api/allIndices"
                response = self.session.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                # Extract indices data
                indices = pd.DataFrame(data['data'])
                
                # Format date and time
                indices['timestamp'] = pd.to_datetime(indices['timestamp'])
                indices['Date'] = indices['timestamp'].dt.date
                indices['Time'] = indices['timestamp'].dt.time
                
                return indices
            except Exception as retry_e:
                logger.error(f"Error retrying Nifty indices fetch: {str(retry_e)}")
                raise
    
    def fetch_market_status(self):
        """Fetch current market status"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/marketStatus"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching market status: {str(e)}")
            raise
    
    def fetch_equity_quote(self, symbol):
        """Fetch detailed quote for an equity"""
        self._check_session()
        
        try:
            url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching equity quote for {symbol}: {str(e)}")
            # Try to refresh session and retry once
            self.refresh_session()
            try:
                url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
                response = self.session.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                return data
            except Exception as retry_e:
                logger.error(f"Error retrying equity quote fetch for {symbol}: {str(retry_e)}")
                raise
    
    def fetch_option_chain(self, symbol, index=False):
        """Fetch option chain for a symbol"""
        self._check_session()
        
        try:
            if index:
                url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
            else:
                url = f"https://www.nseindia.com/api/option-chain-equities?symbol={symbol}"
                
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {str(e)}")
            raise
    
    def fetch_historical_data(self, symbol, from_date, to_date=None, series="EQ"):
        """Fetch historical OHLCV data for a symbol from NSE"""
        self._check_session()
        
        if to_date is None:
            to_date = datetime.now(self.timezone).date()
            
        if isinstance(from_date, datetime):
            from_date = from_date.date()
        if isinstance(to_date, datetime):
            to_date = to_date.date()
            
        try:
            from_date_str = from_date.strftime('%d-%m-%Y')
            to_date_str = to_date.strftime('%d-%m-%Y')
            
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=[%22{series}%22]&from={from_date_str}&to={to_date_str}"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract data
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                
                # Rename columns to standard format
                column_mapping = {
                    'CH_TIMESTAMP': 'Date',
                    'CH_OPENING_PRICE': 'Open',
                    'CH_TRADE_HIGH_PRICE': 'High',
                    'CH_TRADE_LOW_PRICE': 'Low',
                    'CH_CLOSING_PRICE': 'Close',
                    'CH_TOT_TRADED_QTY': 'Volume'
                }
                
                # Rename columns that exist
                for old_name, new_name in column_mapping.items():
                    if old_name in df.columns:
                        df.rename(columns={old_name: new_name}, inplace=True)
                
                # Convert Date to datetime
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df['Time'] = df['Date'].dt.time
                
                return df
            else:
                logger.warning(f"No data found for {symbol} from {from_date_str} to {to_date_str}")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            raise
    
    def fetch_market_watch(self):
        """Fetch current market watch data"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/market-data-pre-open?key=ALL"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract data
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
            else:
                logger.warning("No data found for market watch")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching market watch: {str(e)}")
            raise
    
    def fetch_nifty_200_list(self):
        """Fetch the list of Nifty 200 stocks"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20200"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract stock data
            if 'data' in data:
                df = pd.DataFrame(data['data'])
                return df
            else:
                logger.warning("No data found for Nifty 200 list")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching Nifty 200 list: {str(e)}")
            raise
    
    def fetch_stock_csv_data(self, symbol, period='1y'):
        """Fetch historical data in CSV format for a symbol
        
        Args:
            symbol: Stock symbol
            period: Time period (1d, 1w, 1m, 3m, 6m, 1y, 2y, 5y)
        """
        self._check_session()
        
        try:
            # Map period to NSE parameter
            period_map = {
                '1d': 'day',
                '1w': 'week',
                '1m': 'month',
                '3m': 'quarter',
                '6m': 'half-year',
                '1y': 'year',
                '2y': '2years',
                '5y': '5years'
            }
            
            period_param = period_map.get(period, 'year')
            
            url = f"https://www.nseindia.com/api/historical/cm/equity?symbol={symbol}&series=EQ&period={period_param}&format=csv"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            # Parse CSV data
            data = StringIO(response.text)
            df = pd.read_csv(data)
            
            # Rename columns to standard format
            column_mapping = {
                'Date': 'Date',
                'Open Price': 'Open',
                'High Price': 'High',
                'Low Price': 'Low',
                'Close Price': 'Close',
                'Total Traded Quantity': 'Volume'
            }
            
            # Rename columns that exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)
                    
            # Convert Date to datetime
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df['Time'] = df['Date'].dt.time
                
            # Add symbol column
            df['Symbol'] = symbol
                
            return df
            
        except Exception as e:
            logger.error(f"Error fetching stock CSV data for {symbol}: {str(e)}")
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
    
    def get_top_gainers_losers(self):
        """Fetch top gainers and losers for the day"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/live-analysis-variations"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Create dataframes
            gainers = pd.DataFrame(data.get('FO', {}).get('gainers', []))
            losers = pd.DataFrame(data.get('FO', {}).get('losers', []))
            
            return {
                'gainers': gainers,
                'losers': losers
            }
            
        except Exception as e:
            logger.error(f"Error fetching top gainers and losers: {str(e)}")
            raise
    
    def get_advances_declines(self):
        """Get advances and declines information"""
        self._check_session()
        
        try:
            url = "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
            response = self.session.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Get market breadth info
            advances = data.get('advance', {}).get('advances', 0)
            declines = data.get('advance', {}).get('declines', 0)
            unchanged = data.get('advance', {}).get('unchanged', 0)
            total = advances + declines + unchanged
            
            return {
                'advances': advances,
                'declines': declines,
                'unchanged': unchanged,
                'total': total,
                'advance_decline_ratio': advances / declines if declines > 0 else float('inf')
            }
            
        except Exception as e:
            logger.error(f"Error fetching advances and declines: {str(e)}")
            raise