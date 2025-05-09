import os
import sqlite3
import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime, date
from config.trading_config import DB_PATH

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database utility functions for the paper trading system"""
    
    def __init__(self, db_path=DB_PATH):
        """Initialize the database manager"""
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_db()
        
        logger.info(f"Database initialized at {db_path}")
    
    def _initialize_db(self):
        """Initialize the database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    value REAL NOT NULL,
                    transaction_costs REAL,
                    date TEXT,
                    time TEXT,
                    entry_price REAL,
                    pnl REAL,
                    pnl_pct REAL,
                    balance_after REAL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    value REAL NOT NULL,
                    entry_date TEXT,
                    entry_time TEXT,
                    days_in_trade INTEGER DEFAULT 0,
                    last_update TEXT DEFAULT CURRENT_TIMESTAMP,
                    stop_loss REAL,
                    target REAL,
                    position_type TEXT DEFAULT 'long',
                    UNIQUE(symbol)
                )
            ''')
            
            # Create portfolio history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    time TEXT NOT NULL,
                    balance REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    total_value REAL NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create daily stats table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL UNIQUE,
                    open_balance REAL NOT NULL,
                    close_balance REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL,
                    trade_count INTEGER NOT NULL,
                    winning_trades INTEGER NOT NULL,
                    losing_trades INTEGER NOT NULL,
                    largest_win REAL,
                    largest_loss REAL,
                    positions_held INTEGER NOT NULL
                )
            ''')
            
            # Create model training progress table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_training_progress (
                    symbol TEXT PRIMARY KEY,
                    last_trained_steps INTEGER,
                    last_training_date TEXT,
                    mean_reward REAL,
                    model_path TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def save_trade(self, trade_data):
        """Save trade data to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert transaction_costs dict to string if present
            if 'transaction_costs' in trade_data and isinstance(trade_data['transaction_costs'], dict):
                trade_data['transaction_costs'] = float(trade_data['transaction_costs'].get('total', 0))
            
            # Format dates correctly
            if 'date' in trade_data and isinstance(trade_data['date'], (date, datetime)):
                trade_data['date'] = trade_data['date'].isoformat()
            
            if 'time' in trade_data and hasattr(trade_data['time'], 'isoformat'):
                trade_data['time'] = trade_data['time'].isoformat()
            
            # Convert to DataFrame and save
            df = pd.DataFrame([trade_data])
            df.to_sql('trades', conn, if_exists='append', index=False)
            
            conn.close()
            
            logger.debug(f"Trade saved to database: {trade_data.get('symbol')} {trade_data.get('action')}")
            return True
        except Exception as e:
            logger.error(f"Error saving trade to database: {str(e)}")
            return False
    
    def save_position(self, position_data):
        """Save position data to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Format dates correctly
            if 'entry_date' in position_data and isinstance(position_data['entry_date'], (date, datetime)):
                position_data['entry_date'] = position_data['entry_date'].isoformat()
            
            if 'entry_time' in position_data and hasattr(position_data['entry_time'], 'isoformat'):
                position_data['entry_time'] = position_data['entry_time'].isoformat()
            
            # Check if position already exists
            cursor.execute("SELECT id FROM positions WHERE symbol = ?", (position_data['symbol'],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing position
                placeholders = ', '.join([f"{k} = ?" for k in position_data.keys()])
                values = list(position_data.values())
                values.append(position_data['symbol'])
                
                cursor.execute(f"UPDATE positions SET {placeholders} WHERE symbol = ?", values)
            else:
                # Insert new position
                columns = ', '.join(position_data.keys())
                placeholders = ', '.join(['?'] * len(position_data))
                values = list(position_data.values())
                
                cursor.execute(f"INSERT INTO positions ({columns}) VALUES ({placeholders})", values)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Position saved to database: {position_data.get('symbol')}")
            return True
        except Exception as e:
            logger.error(f"Error saving position to database: {str(e)}")
            return False
    
    def delete_position(self, symbol):
        """Delete a position from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Position deleted from database: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error deleting position from database: {str(e)}")
            return False
    
    def save_portfolio_history(self, portfolio_data):
        """Save portfolio history data to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Format dates correctly
            if 'date' in portfolio_data and isinstance(portfolio_data['date'], (date, datetime)):
                portfolio_data['date'] = portfolio_data['date'].isoformat()
            
            if 'time' in portfolio_data and hasattr(portfolio_data['time'], 'isoformat'):
                portfolio_data['time'] = portfolio_data['time'].isoformat()
            
            # Convert to DataFrame and save
            df = pd.DataFrame([portfolio_data])
            df.to_sql('portfolio_history', conn, if_exists='append', index=False)
            
            conn.close()
            
            logger.debug(f"Portfolio history saved to database")
            return True
        except Exception as e:
            logger.error(f"Error saving portfolio history to database: {str(e)}")
            return False
    
    def save_daily_stats(self, stats_data):
        """Save daily statistics to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Format date correctly
            if 'date' in stats_data and isinstance(stats_data['date'], (date, datetime)):
                stats_data['date'] = stats_data['date'].isoformat()
            
            # Check if stats for this date already exist
            cursor.execute("SELECT id FROM daily_stats WHERE date = ?", (stats_data['date'],))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing stats
                placeholders = ', '.join([f"{k} = ?" for k in stats_data.keys()])
                values = list(stats_data.values())
                values.append(stats_data['date'])
                
                cursor.execute(f"UPDATE daily_stats SET {placeholders} WHERE date = ?", values)
            else:
                # Insert new stats
                columns = ', '.join(stats_data.keys())
                placeholders = ', '.join(['?'] * len(stats_data))
                values = list(stats_data.values())
                
                cursor.execute(f"INSERT INTO daily_stats ({columns}) VALUES ({placeholders})", values)
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Daily stats saved to database for {stats_data.get('date')}")
            return True
        except Exception as e:
            logger.error(f"Error saving daily stats to database: {str(e)}")
            return False
    
    def update_model_training_progress(self, symbol, trained_steps, mean_reward=None, model_path=None):
        """Update model training progress in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if entry exists
            cursor.execute("SELECT last_trained_steps FROM model_training_progress WHERE symbol = ?", (symbol,))
            existing = cursor.fetchone()
            
            current_date = datetime.now().isoformat()
            
            if existing:
                # Update existing entry
                cursor.execute(
                    "UPDATE model_training_progress SET last_trained_steps = ?, last_training_date = ?, mean_reward = ?, model_path = ? WHERE symbol = ?",
                    (trained_steps, current_date, mean_reward, model_path, symbol)
                )
            else:
                # Insert new entry
                cursor.execute(
                    "INSERT INTO model_training_progress (symbol, last_trained_steps, last_training_date, mean_reward, model_path) VALUES (?, ?, ?, ?, ?)",
                    (symbol, trained_steps, current_date, mean_reward, model_path)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"Model training progress updated for {symbol}: {trained_steps} steps")
            return True
        except Exception as e:
            logger.error(f"Error updating model training progress: {str(e)}")
            return False
    
    def get_trades(self, symbol=None, start_date=None, end_date=None):
        """Get trades from the database with optional filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM trades"
            params = []
            
            # Add filters
            filters = []
            if symbol:
                filters.append("symbol = ?")
                params.append(symbol)
            
            if start_date:
                if isinstance(start_date, (date, datetime)):
                    start_date = start_date.isoformat()
                filters.append("date >= ?")
                params.append(start_date)
            
            if end_date:
                if isinstance(end_date, (date, datetime)):
                    end_date = end_date.isoformat()
                filters.append("date <= ?")
                params.append(end_date)
            
            # Add WHERE clause if filters exist
            if filters:
                query += " WHERE " + " AND ".join(filters)
            
            # Add order by
            query += " ORDER BY date, time"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            return df
        except Exception as e:
            logger.error(f"Error getting trades from database: {str(e)}")
            return pd.DataFrame()
    
    def get_positions(self):
        """Get current positions from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query("SELECT * FROM positions", conn)
            
            conn.close()
            
            return df
        except Exception as e:
            logger.error(f"Error getting positions from database: {str(e)}")
            return pd.DataFrame()
    
    def get_portfolio_history(self, start_date=None, end_date=None):
        """Get portfolio history from the database with optional date filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM portfolio_history"
            params = []
            
            # Add filters
            filters = []
            if start_date:
                if isinstance(start_date, (date, datetime)):
                    start_date = start_date.isoformat()
                filters.append("date >= ?")
                params.append(start_date)
            
            if end_date:
                if isinstance(end_date, (date, datetime)):
                    end_date = end_date.isoformat()
                filters.append("date <= ?")
                params.append(end_date)
            
            # Add WHERE clause if filters exist
            if filters:
                query += " WHERE " + " AND ".join(filters)
            
            # Add order by
            query += " ORDER BY date, time"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            return df
        except Exception as e:
            logger.error(f"Error getting portfolio history from database: {str(e)}")
            return pd.DataFrame()
    
    def get_daily_stats(self, start_date=None, end_date=None):
        """Get daily statistics from the database with optional date filtering"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM daily_stats"
            params = []
            
            # Add filters
            filters = []
            if start_date:
                if isinstance(start_date, (date, datetime)):
                    start_date = start_date.isoformat()
                filters.append("date >= ?")
                params.append(start_date)
            
            if end_date:
                if isinstance(end_date, (date, datetime)):
                    end_date = end_date.isoformat()
                filters.append("date <= ?")
                params.append(end_date)
            
            # Add WHERE clause if filters exist
            if filters:
                query += " WHERE " + " AND ".join(filters)
            
            # Add order by
            query += " ORDER BY date"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            conn.close()
            
            return df
        except Exception as e:
            logger.error(f"Error getting daily stats from database: {str(e)}")
            return pd.DataFrame()
    
    def get_model_training_progress(self, symbol=None):
        """Get model training progress from the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM model_training_progress"
            params = []
            
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)
            
            # Execute query
            if symbol:
                cursor = conn.cursor()
                cursor.execute(query, params)
                row = cursor.fetchone()
                
                if row:
                    column_names = [description[0] for description in cursor.description]
                    result = dict(zip(column_names, row))
                    conn.close()
                    return result
                else:
                    conn.close()
                    return None
            else:
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                return df
        except Exception as e:
            logger.error(f"Error getting model training progress from database: {str(e)}")
            return None if symbol else pd.DataFrame()
    
    def delete_all_data(self, confirmation=False):
        """Delete all data from the database (dangerous operation, requires confirmation)"""
        if not confirmation:
            logger.warning("delete_all_data called without confirmation")
            return False
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete all data from all tables
            cursor.execute("DELETE FROM trades")
            cursor.execute("DELETE FROM positions")
            cursor.execute("DELETE FROM portfolio_history")
            cursor.execute("DELETE FROM daily_stats")
            cursor.execute("DELETE FROM model_training_progress")
            
            conn.commit()
            conn.close()
            
            logger.warning("All data deleted from database")
            return True
        except Exception as e:
            logger.error(f"Error deleting all data from database: {str(e)}")
            return False
    
    def backup_database(self, backup_path=None):
        """Backup the database to a specified path"""
        if backup_path is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{os.path.splitext(self.db_path)[0]}_backup_{now}.db"
        
        try:
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            return False
    
    def restore_database(self, backup_path):
        """Restore the database from a backup"""
        try:
            import shutil
            
            # Backup current database first
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_backup = f"{os.path.splitext(self.db_path)[0]}_pre_restore_{now}.db"
            shutil.copy2(self.db_path, temp_backup)
            
            # Restore from backup
            shutil.copy2(backup_path, self.db_path)
            
            logger.info(f"Database restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error restoring database: {str(e)}")
            return False
    
    def export_to_csv(self, table_name, output_path=None):
        """Export a table to CSV"""
        if output_path is None:
            now = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/exports/{table_name}_{now}.csv"
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Check if table exists
            cursor = conn.cursor()
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                logger.error(f"Table {table_name} does not exist")
                conn.close()
                return False
            
            # Export to CSV
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            df.to_csv(output_path, index=False)
            
            conn.close()
            
            logger.info(f"Table {table_name} exported to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting table to CSV: {str(e)}")
            return False
    
    def execute_custom_query(self, query, params=None):
        """Execute a custom SQL query"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # For SELECT queries
            if query.strip().upper().startswith("SELECT"):
                df = pd.read_sql_query(query, conn, params=params)
                conn.close()
                return df
            # For other queries (INSERT, UPDATE, DELETE)
            else:
                cursor = conn.cursor()
                if params:
                    cursor.execute(query, params)
                else:
                    cursor.execute(query)
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error executing custom query: {str(e)}")
            if 'conn' in locals():
                conn.close()
            return None
    
    def get_performance_metrics(self, start_date=None, end_date=None):
        """Calculate and return performance metrics from database"""
        try:
            # Get trades data
            trades_df = self.get_trades(start_date=start_date, end_date=end_date)
            
            if trades_df.empty:
                logger.warning("No trades found for performance metrics calculation")
                return {}
            
            # Filter to only sell trades (completed trades)
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            # Calculate metrics
            total_trades = len(sell_trades)
            winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
            losing_trades = len(sell_trades[sell_trades['pnl'] <= 0])
            
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Profit metrics
            total_profit = sell_trades[sell_trades['pnl'] > 0]['pnl'].sum()
            total_loss = abs(sell_trades[sell_trades['pnl'] <= 0]['pnl'].sum())
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Get portfolio history
            portfolio_df = self.get_portfolio_history(start_date=start_date, end_date=end_date)
            
            # Calculate returns and drawdown
            if not portfolio_df.empty and len(portfolio_df) > 1:
                portfolio_df['return'] = portfolio_df['total_value'].pct_change()
                
                # Calculate total return
                initial_value = portfolio_df['total_value'].iloc[0]
                final_value = portfolio_df['total_value'].iloc[-1]
                total_return = (final_value - initial_value) / initial_value
                
                # Calculate drawdown
                portfolio_df['cum_max'] = portfolio_df['total_value'].cummax()
                portfolio_df['drawdown'] = (portfolio_df['cum_max'] - portfolio_df['total_value']) / portfolio_df['cum_max']
                max_drawdown = portfolio_df['drawdown'].max()
                
                # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
                daily_returns_std = portfolio_df['return'].std()
                avg_daily_return = portfolio_df['return'].mean()
                sharpe_ratio = avg_daily_return / daily_returns_std * np.sqrt(252) if daily_returns_std > 0 else 0
                
                # Calculate Calmar ratio (annualized return / max drawdown)
                days = len(portfolio_df)
                annualized_return = (1 + total_return) ** (252 / days) - 1
                calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else float('inf')
            else:
                total_return = 0
                max_drawdown = 0
                sharpe_ratio = 0
                calmar_ratio = 0
                annualized_return = 0
            
            # Return all metrics
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'total_loss': total_loss,
                'profit_factor': profit_factor,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'calmar_ratio': calmar_ratio
            }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def get_symbol_performance(self, start_date=None, end_date=None):
        """Calculate performance metrics by symbol"""
        try:
            # Get trades data
            trades_df = self.get_trades(start_date=start_date, end_date=end_date)
            
            if trades_df.empty:
                logger.warning("No trades found for symbol performance calculation")
                return []
            
            # Group by symbol
            symbols = trades_df['symbol'].unique()
            symbol_metrics = []
            
            for symbol in symbols:
                symbol_trades = trades_df[trades_df['symbol'] == symbol]
                symbol_sells = symbol_trades[symbol_trades['action'] == 'SELL']
                
                if len(symbol_sells) == 0:
                    continue
                
                # Calculate metrics
                total_trades = len(symbol_sells)
                winning_trades = len(symbol_sells[symbol_sells['pnl'] > 0])
                losing_trades = len(symbol_sells[symbol_sells['pnl'] <= 0])
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Calculate total P&L
                total_pnl = symbol_sells['pnl'].sum()
                
                # Calculate return
                first_buy = symbol_trades[symbol_trades['action'] == 'BUY'].iloc[0] if len(symbol_trades[symbol_trades['action'] == 'BUY']) > 0 else None
                
                if first_buy is not None:
                    initial_investment = first_buy['value']
                    symbol_return = total_pnl / initial_investment if initial_investment > 0 else 0
                else:
                    symbol_return = 0
                
                symbol_metrics.append({
                    'symbol': symbol,
                    'trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'pnl': total_pnl,
                    'return': symbol_return
                })
            
            # Sort by P&L (descending)
            symbol_metrics.sort(key=lambda x: x['pnl'], reverse=True)
            
            return symbol_metrics
        except Exception as e:
            logger.error(f"Error calculating symbol performance: {str(e)}")
            return []
    
    def get_pnl_distribution(self, bin_size=1000, start_date=None, end_date=None):
        """Calculate P&L distribution for histogram visualization"""
        try:
            # Get trades data
            trades_df = self.get_trades(start_date=start_date, end_date=end_date)
            
            if trades_df.empty:
                logger.warning("No trades found for P&L distribution calculation")
                return []
            
            # Filter to only sell trades
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(sell_trades) == 0:
                return []
            
            # Get P&L values
            pnl_values = sell_trades['pnl'].values
            
            # Calculate bin edges
            min_pnl = min(pnl_values)
            max_pnl = max(pnl_values)
            
            # Ensure minimum range
            if max_pnl - min_pnl < bin_size * 5:
                center = (max_pnl + min_pnl) / 2
                min_pnl = center - bin_size * 2.5
                max_pnl = center + bin_size * 2.5
            
            # Create bins
            bin_edges = np.arange(
                np.floor(min_pnl / bin_size) * bin_size,
                np.ceil(max_pnl / bin_size) * bin_size + bin_size,
                bin_size
            )
            
            # Count occurrences in each bin
            counts, edges = np.histogram(pnl_values, bins=bin_edges)
            
            # Format for visualization
            distribution = []
            for i in range(len(counts)):
                bin_start = edges[i]
                bin_end = edges[i+1]
                
                # Format bin range
                if bin_start >= 0:
                    range_str = f"₹{bin_start:.0f} - ₹{bin_end:.0f}"
                else:
                    range_str = f"-₹{abs(bin_start):.0f} - "
                    if bin_end >= 0:
                        range_str += f"₹{bin_end:.0f}"
                    else:
                        range_str += f"-₹{abs(bin_end):.0f}"
                
                distribution.append({
                    'range': range_str,
                    'count': int(counts[i])
                })
            
            return distribution
        except Exception as e:
            logger.error(f"Error calculating P&L distribution: {str(e)}")
            return []
    
    def get_drawdown_data(self, start_date=None, end_date=None):
        """Get drawdown data for visualization"""
        try:
            # Get portfolio history
            portfolio_df = self.get_portfolio_history(start_date=start_date, end_date=end_date)
            
            if portfolio_df.empty:
                logger.warning("No portfolio history found for drawdown calculation")
                return []
            
            # Calculate drawdown
            portfolio_df['cum_max'] = portfolio_df['total_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['cum_max'] - portfolio_df['total_value']) / portfolio_df['cum_max']
            
            # Format for visualization
            drawdown_data = []
            for _, row in portfolio_df.iterrows():
                drawdown_data.append({
                    'date': row['date'],
                    'time': row['time'],
                    'value': float(row['total_value']),
                    'drawdown': float(row['drawdown'])
                })
            
            return drawdown_data
        except Exception as e:
            logger.error(f"Error calculating drawdown data: {str(e)}")
            return []