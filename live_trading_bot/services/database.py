"""
Database service for the trading bot.
"""
import os
import logging
import sqlite3
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseService:
    """
    Database service for the trading bot.
    Handles storing and retrieving trading data.
    """
    
    def __init__(self, db_path):
        """
        Initialize the database service
        
        Args:
            db_path (str): Path to the SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database connection and tables
        self._init_db()
        
        logger.info(f"Database service initialized with path: {db_path}")
    
    def _init_db(self):
        """Initialize the database connection and create tables if they don't exist"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            c = self.conn.cursor()
            
            # Create tables
            
            # Model training progress
            c.execute('''
                CREATE TABLE IF NOT EXISTS model_training_progress (
                    symbol TEXT PRIMARY KEY,
                    last_trained_steps INTEGER,
                    last_trained_date TEXT
                )
            ''')
            
            # Trades table
            c.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    order_id TEXT,
                    transaction_type TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    profit_loss REAL,
                    profit_loss_percent REAL,
                    exit_type TEXT,
                    market_regime TEXT,
                    stop_loss REAL,
                    target_price REAL,
                    trailing_stop REAL,
                    days_in_trade INTEGER,
                    buy_signal_strength REAL,
                    sell_signal_strength REAL,
                    transaction_costs REAL
                )
            ''')
            
            # Daily performance table
            c.execute('''
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    open_balance REAL,
                    close_balance REAL,
                    daily_profit_loss REAL,
                    daily_profit_loss_percent REAL,
                    trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    market_regime TEXT,
                    max_drawdown REAL,
                    PRIMARY KEY (date, symbol)
                )
            ''')
            
            # Price data table
            c.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            
            self.conn.commit()
            logger.info("Database tables initialized")
            
        except Exception as e:
            logger.exception(f"Error initializing database: {str(e)}")
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def record_trade(self, trade_data):
        """
        Record a trade in the database
        
        Args:
            trade_data (dict): Trade data to record
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.conn:
                self._init_db()
                
            if not self.conn:
                logger.error("Database connection not available")
                return False
                
            c = self.conn.cursor()
            
            # Prepare SQL
            sql = '''
                INSERT INTO trades (
                    symbol, order_id, transaction_type, price, quantity, timestamp,
                    entry_price, exit_price, profit_loss, profit_loss_percent, exit_type,
                    market_regime, stop_loss, target_price, trailing_stop, days_in_trade,
                    buy_signal_strength, sell_signal_strength, transaction_costs
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            '''
            
            # Prepare data
            values = (
                trade_data.get('symbol'),
                trade_data.get('order_id'),
                trade_data.get('transaction_type'),
                trade_data.get('price'),
                trade_data.get('quantity'),
                trade_data.get('timestamp', datetime.now().isoformat()),
                trade_data.get('entry_price'),
                trade_data.get('exit_price'),
                trade_data.get('profit_loss'),
                trade_data.get('profit_loss_percent'),
                trade_data.get('exit_type'),
                trade_data.get('market_regime'),
                trade_data.get('stop_loss'),
                trade_data.get('target_price'),
                trade_data.get('trailing_stop'),
                trade_data.get('days_in_trade'),
                trade_data.get('buy_signal_strength'),
                trade_data.get('sell_signal_strength'),
                trade_data.get('transaction_costs')
            )
            
            c.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Trade recorded for {trade_data.get('symbol')}: {trade_data.get('transaction_type')} at {trade_data.get('price')}")
            return True
            
        except Exception as e:
            logger.exception(f"Error recording trade: {str(e)}")
            return False
    
    def record_daily_performance(self, performance_data):
        """
        Record daily performance in the database
        
        Args:
            performance_data (dict): Performance data to record
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.conn:
                self._init_db()
                
            if not self.conn:
                logger.error("Database connection not available")
                return False
                
            c = self.conn.cursor()
            
            # Prepare SQL
            sql = '''
                INSERT OR REPLACE INTO daily_performance (
                    date, symbol, open_balance, close_balance, daily_profit_loss,
                    daily_profit_loss_percent, trades, winning_trades, losing_trades,
                    market_regime, max_drawdown
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            '''
            
            # Prepare data
            values = (
                performance_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                performance_data.get('symbol'),
                performance_data.get('open_balance'),
                performance_data.get('close_balance'),
                performance_data.get('daily_profit_loss'),
                performance_data.get('daily_profit_loss_percent'),
                performance_data.get('trades'),
                performance_data.get('winning_trades'),
                performance_data.get('losing_trades'),
                performance_data.get('market_regime'),
                performance_data.get('max_drawdown')
            )
            
            c.execute(sql, values)
            self.conn.commit()
            
            logger.info(f"Daily performance recorded for {performance_data.get('symbol')} on {performance_data.get('date')}")
            return True
            
        except Exception as e:
            logger.exception(f"Error recording daily performance: {str(e)}")
            return False
    
    def record_price_data(self, symbol, price_data):
        """
        Record price data in the database
        
        Args:
            symbol (str): Trading symbol
            price_data (pandas.DataFrame): DataFrame with OHLCV data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.conn:
                self._init_db()
                
            if not self.conn:
                logger.error("Database connection not available")
                return False
                
            # Ensure we have required columns
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            for col in required_columns:
                if col not in price_data.columns:
                    logger.error(f"Missing required column in price data: {col}")
                    return False
            
            # Add symbol column
            price_data['symbol'] = symbol
            
            # Write to database
            price_data.to_sql('price_data', self.conn, if_exists='append', index=False)
            
            logger.info(f"Recorded {len(price_data)} price records for {symbol}")
            return True
            
        except Exception as e:
            logger.exception(f"Error recording price data: {str(e)}")
            return False
    
    def get_trades(self, symbol=None, start_date=None, end_date=None):
        """
        Get trades from the database
        
        Args:
            symbol (str, optional): Filter by trading symbol
            start_date (str, optional): Filter by start date (ISO format)
            end_date (str, optional): Filter by end date (ISO format)
            
        Returns:
            pandas.DataFrame: DataFrame with trade data
        """
        try:
            if not self.conn:
                self._init_db()
                
            if not self.conn:
                logger.error("Database connection not available")
                return pd.DataFrame()
                
            # Build query
            query = "SELECT * FROM trades"
            params = []
            
            where_clauses = []
            
            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol)
                
            if start_date:
                where_clauses.append("timestamp >= ?")
                params.append(start_date)
                
            if end_date:
                where_clauses.append("timestamp <= ?")
                params.append(end_date)
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += " ORDER BY timestamp DESC"
            
            # Execute query
            return pd.read_sql_query(query, self.conn, params=params)
            
        except Exception as e:
            logger.exception(f"Error getting trades: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_performance(self, symbol=None, start_date=None, end_date=None):
        """
        Get daily performance from the database
        
        Args:
            symbol (str, optional): Filter by trading symbol
            start_date (str, optional): Filter by start date (YYYY-MM-DD)
            end_date (str, optional): Filter by end date (YYYY-MM-DD)
            
        Returns:
            pandas.DataFrame: DataFrame with daily performance data
        """
        try:
            if not self.conn:
                self._init_db()
                
            if not self.conn:
                logger.error("Database connection not available")
                return pd.DataFrame()
                
            # Build query
            query = "SELECT * FROM daily_performance"
            params = []
            
            where_clauses = []
            
            if symbol:
                where_clauses.append("symbol = ?")
                params.append(symbol)
                
            if start_date:
                where_clauses.append("date >= ?")
                params.append(start_date)
                
            if end_date:
                where_clauses.append("date <= ?")
                params.append(end_date)
                
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
                
            query += " ORDER BY date DESC"
            
            # Execute query
            return pd.read_sql_query(query, self.conn, params=params)
            
        except Exception as e:
            logger.exception(f"Error getting daily performance: {str(e)}")
            return pd.DataFrame()
    
    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None