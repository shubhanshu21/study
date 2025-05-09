"""
Performance metrics for trading.
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Performance metrics for trading.
    """
    
    def __init__(self, config):
        """
        Initialize performance metrics
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        logger.info("Performance metrics initialized")
    
    def calculate_metrics(self, trades_df, capital_history_df):
        """
        Calculate performance metrics
        
        Args:
            trades_df (pandas.DataFrame): DataFrame with trade data
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            
        Returns:
            dict: Performance metrics
        """
        try:
            # Check if dataframes are empty
            if trades_df.empty or capital_history_df.empty:
                logger.warning("Empty dataframes provided")
                return {}
                
            # Calculate returns
            returns_df = self._calculate_returns(capital_history_df)
            
            # Calculate trade metrics
            trade_metrics = self._calculate_trade_metrics(trades_df)
            
            # Calculate return metrics
            return_metrics = self._calculate_return_metrics(returns_df)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(returns_df, capital_history_df)
            
            # Combine all metrics
            metrics = {
                **trade_metrics,
                **return_metrics,
                **risk_metrics,
                'timestamp': datetime.now().isoformat()
            }
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def _calculate_returns(self, capital_history_df):
        """
        Calculate returns from capital history
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            
        Returns:
            pandas.DataFrame: DataFrame with returns
        """
        try:
            # Create a copy to avoid modifying the original
            df = capital_history_df.copy()
            
            # Ensure 'timestamp' is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Ensure 'capital' column exists
            if 'capital' not in df.columns and 'net_worth' in df.columns:
                df['capital'] = df['net_worth']
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate returns
            df['return'] = df['capital'].pct_change()
            
            # Calculate cumulative returns
            df['cum_return'] = (1 + df['return']).cumprod() - 1
            
            # Calculate log returns
            df['log_return'] = np.log(df['capital'] / df['capital'].shift(1))
            
            # Calculate date columns
            df['date'] = df['timestamp'].dt.date
            
            return df
            
        except Exception as e:
            logger.exception(f"Error calculating returns: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_trade_metrics(self, trades_df):
        """
        Calculate trade-based metrics
        
        Args:
            trades_df (pandas.DataFrame): DataFrame with trade data
            
        Returns:
            dict: Trade metrics
        """
        try:
            # Create a copy to avoid modifying the original
            df = trades_df.copy()
            
            # Filter to completed trades
            buy_trades = df[df['type'] == 'buy']
            sell_trades = df[df['type'] == 'sell']
            
            # Calculate total trades
            total_trades = len(sell_trades)
            
            if total_trades == 0:
                logger.warning("No completed trades found")
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'avg_profit_pct': 0,
                    'avg_loss_pct': 0,
                    'profit_factor': 0,
                    'avg_trade': 0,
                    'avg_trade_pct': 0,
                    'largest_profit': 0,
                    'largest_loss': 0,
                    'avg_bars_in_winning_trades': 0,
                    'avg_bars_in_losing_trades': 0
                }
                
            # Calculate winning and losing trades
            winning_trades = sell_trades[sell_trades['profit'] > 0]
            losing_trades = sell_trades[sell_trades['profit'] <= 0]
            
            num_winning = len(winning_trades)
            num_losing = len(losing_trades)
            
            # Calculate win rate
            win_rate = num_winning / total_trades if total_trades > 0 else 0
            
            # Calculate average profit and loss
            avg_profit = winning_trades['profit'].mean() if num_winning > 0 else 0
            avg_loss = losing_trades['profit'].mean() if num_losing > 0 else 0
            
            # Calculate average profit and loss percentage
            avg_profit_pct = winning_trades['profit_pct'].mean() if num_winning > 0 else 0
            avg_loss_pct = losing_trades['profit_pct'].mean() if num_losing > 0 else 0
            
            # Calculate profit factor
            total_profit = winning_trades['profit'].sum() if num_winning > 0 else 0
            total_loss = abs(losing_trades['profit'].sum()) if num_losing > 0 else 0
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            # Calculate average trade
            avg_trade = sell_trades['profit'].mean()
            avg_trade_pct = sell_trades['profit_pct'].mean()
            
            # Calculate largest profit and loss
            largest_profit = winning_trades['profit'].max() if num_winning > 0 else 0
            largest_loss = losing_trades['profit'].min() if num_losing > 0 else 0
            
            # Calculate average bars (days) in winning and losing trades
            avg_bars_in_winning_trades = winning_trades['days_held'].mean() if num_winning > 0 else 0
            avg_bars_in_losing_trades = losing_trades['days_held'].mean() if num_losing > 0 else 0
            
            # Return metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': num_winning,
                'losing_trades': num_losing,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'avg_profit_pct': avg_profit_pct,
                'avg_loss_pct': avg_loss_pct,
                'profit_factor': profit_factor,
                'avg_trade': avg_trade,
                'avg_trade_pct': avg_trade_pct,
                'largest_profit': largest_profit,
                'largest_loss': largest_loss,
                'avg_bars_in_winning_trades': avg_bars_in_winning_trades,
                'avg_bars_in_losing_trades': avg_bars_in_losing_trades
            }
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating trade metrics: {str(e)}")
            return {}
    
    def _calculate_return_metrics(self, returns_df):
        """
        Calculate return-based metrics
        
        Args:
            returns_df (pandas.DataFrame): DataFrame with returns
            
        Returns:
            dict: Return metrics
        """
        try:
            # Check if dataframe is empty
            if returns_df.empty:
                logger.warning("Empty returns dataframe")
                return {}
                
            # Calculate total return
            initial_capital = returns_df['capital'].iloc[0]
            final_capital = returns_df['capital'].iloc[-1]
            total_return = final_capital / initial_capital - 1
            
            # Calculate annualized return
            days = (returns_df['timestamp'].iloc[-1] - returns_df['timestamp'].iloc[0]).days
            years = days / 365.25
            annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Calculate daily returns statistics
            daily_returns = returns_df.groupby('date')['return'].sum()
            
            # Calculate monthly returns
            if 'timestamp' in returns_df.columns:
                returns_df['month'] = returns_df['timestamp'].dt.to_period('M')
                monthly_returns = returns_df.groupby('month')['return'].sum()
            else:
                monthly_returns = pd.Series([])
            
            # Calculate average daily and monthly returns
            avg_daily_return = daily_returns.mean()
            avg_monthly_return = monthly_returns.mean()
            
            # Calculate standard deviation of returns
            daily_std = daily_returns.std()
            monthly_std = monthly_returns.std()
            
            # Calculate Sharpe ratio
            risk_free_rate = float(self.config.get('RISK_FREE_RATE', 0.05))
            daily_risk_free = (1 + risk_free_rate) ** (1/252) - 1
            sharpe_ratio = (avg_daily_return - daily_risk_free) / daily_std * np.sqrt(252) if daily_std > 0 else 0
            
            # Calculate Sortino ratio
            negative_returns = daily_returns[daily_returns < 0]
            downside_deviation = negative_returns.std()
            sortino_ratio = (avg_daily_return - daily_risk_free) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
            
            # Calculate percentage of positive days and months
            positive_days = (daily_returns > 0).mean()
            positive_months = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0
            
            # Return metrics
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'avg_daily_return': avg_daily_return,
                'avg_monthly_return': avg_monthly_return,
                'daily_std': daily_std,
                'monthly_std': monthly_std,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'positive_days': positive_days,
                'positive_months': positive_months
            }
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating return metrics: {str(e)}")
            return {}
    
    def _calculate_risk_metrics(self, returns_df, capital_history_df):
        """
        Calculate risk-based metrics
        
        Args:
            returns_df (pandas.DataFrame): DataFrame with returns
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            
        Returns:
            dict: Risk metrics
        """
        try:
            # Check if dataframes are empty
            if returns_df.empty or capital_history_df.empty:
                logger.warning("Empty dataframes provided")
                return {}
                
            # Calculate maximum drawdown
            if 'cum_return' in returns_df.columns:
                # Calculate running maximum
                returns_df['running_max'] = returns_df['cum_return'].cummax()
                
                # Calculate drawdown
                returns_df['drawdown'] = returns_df['running_max'] - returns_df['cum_return']
                
                # Get maximum drawdown
                max_drawdown = returns_df['drawdown'].max()
                
                # Calculate average drawdown
                avg_drawdown = returns_df['drawdown'].mean()
                
                # Calculate time in drawdown
                time_in_drawdown = (returns_df['drawdown'] > 0).mean()
            else:
                # Calculate running maximum from capital
                capital_history_df['running_max'] = capital_history_df['capital'].cummax()
                
                # Calculate drawdown
                capital_history_df['drawdown'] = 1 - capital_history_df['capital'] / capital_history_df['running_max']
                
                # Get maximum drawdown
                max_drawdown = capital_history_df['drawdown'].max()
                
                # Calculate average drawdown
                avg_drawdown = capital_history_df['drawdown'].mean()
                
                # Calculate time in drawdown
                time_in_drawdown = (capital_history_df['drawdown'] > 0).mean()
            
            # Calculate Calmar ratio
            annual_return = self._calculate_return_metrics(returns_df).get('annual_return', 0)
            calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else float('inf')
            
            # Calculate volatility
            if 'return' in returns_df.columns:
                volatility = returns_df['return'].std() * np.sqrt(252)
            else:
                volatility = 0
            
            # Return metrics
            metrics = {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'time_in_drawdown': time_in_drawdown,
                'calmar_ratio': calmar_ratio,
                'volatility': volatility
            }
            
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating risk metrics: {str(e)}")
            return {}
    
    def calculate_rolling_metrics(self, capital_history_df, window=30):
        """
        Calculate rolling performance metrics
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            window (int): Rolling window size
            
        Returns:
            pandas.DataFrame: DataFrame with rolling metrics
        """
        try:
            # Check if dataframe is empty
            if capital_history_df.empty:
                logger.warning("Empty capital history dataframe")
                return pd.DataFrame()
                
            # Create a copy to avoid modifying the original
            df = capital_history_df.copy()
            
            # Ensure 'timestamp' is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            # Ensure 'capital' column exists
            if 'capital' not in df.columns and 'net_worth' in df.columns:
                df['capital'] = df['net_worth']
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Calculate returns
            df['return'] = df['capital'].pct_change()
            df['log_return'] = np.log(df['capital'] / df['capital'].shift(1))
            
            # Calculate rolling metrics
            df['rolling_return'] = df['capital'].pct_change(window)
            df['rolling_annual_return'] = (1 + df['rolling_return']) ** (252 / window) - 1
            df['rolling_volatility'] = df['return'].rolling(window).std() * np.sqrt(252)
            df['rolling_sharpe'] = df['rolling_annual_return'] / df['rolling_volatility']
            
            # Calculate rolling drawdown
            df['rolling_max'] = df['capital'].rolling(window, min_periods=1).max()
            df['rolling_drawdown'] = 1 - df['capital'] / df['rolling_max']
            df['rolling_max_drawdown'] = df['rolling_drawdown'].rolling(window).max()
            
            # Calculate rolling Calmar ratio
            df['rolling_calmar'] = df['rolling_annual_return'] / df['rolling_max_drawdown']
            
            # Calculate rolling win rate
            df['positive_return'] = df['return'] > 0
            df['rolling_win_rate'] = df['positive_return'].rolling(window).mean()
            
            # Replace infinite and NaN values
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            
            return df
            
        except Exception as e:
            logger.exception(f"Error calculating rolling metrics: {str(e)}")
            return pd.DataFrame()
    
    def calculate_symbol_metrics(self, trades_df, capital_history_df, symbol=None):
        """
        Calculate metrics for a specific symbol
        
        Args:
            trades_df (pandas.DataFrame): DataFrame with trade data
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            symbol (str, optional): Trading symbol
            
        Returns:
            dict: Symbol metrics
        """
        try:
            # Filter data by symbol if provided
            if symbol:
                filtered_trades = trades_df[trades_df['symbol'] == symbol]
                
                # Filter capital history if symbol column exists
                if 'symbol' in capital_history_df.columns:
                    filtered_capital = capital_history_df[capital_history_df['symbol'] == symbol]
                else:
                    filtered_capital = capital_history_df
            else:
                filtered_trades = trades_df
                filtered_capital = capital_history_df
            
            # Calculate metrics
            metrics = self.calculate_metrics(filtered_trades, filtered_capital)
            
            # Add symbol to metrics
            if symbol:
                metrics['symbol'] = symbol
                
            return metrics
            
        except Exception as e:
            logger.exception(f"Error calculating symbol metrics: {str(e)}")
            return {}