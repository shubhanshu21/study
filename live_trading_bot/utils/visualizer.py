"""
Visualization utilities for trading data.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64

logger = logging.getLogger(__name__)

class TradingVisualizer:
    """
    Visualization utilities for trading data.
    """
    
    def __init__(self, config=None):
        """
        Initialize visualizer
        
        Args:
            config (optional): Application configuration
        """
        self.config = config
        
        # Set default style
        sns.set_style('darkgrid')
        
        logger.info("Trading visualizer initialized")
    
    def plot_capital_history(self, capital_history_df, title="Trading Performance"):
        """
        Plot capital history
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
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
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot capital history
            ax.plot(df['timestamp'], df['capital'], linewidth=2)
            
            # Add initial capital as horizontal line if config available
            if self.config and 'DEFAULT_INITIAL_BALANCE' in self.config:
                initial_balance = float(self.config.get('DEFAULT_INITIAL_BALANCE', 100000))
                ax.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial Capital')
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Capital (₹)')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend()
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting capital history: {str(e)}")
            return None
    
    def plot_drawdown(self, capital_history_df, title="Drawdown Analysis"):
        """
        Plot drawdown
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
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
            
            # Calculate running maximum
            df['running_max'] = df['capital'].cummax()
            
            # Calculate drawdown
            df['drawdown'] = 1 - df['capital'] / df['running_max']
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot drawdown
            ax.fill_between(df['timestamp'], df['drawdown'], 0, alpha=0.3, color='r')
            ax.plot(df['timestamp'], df['drawdown'], color='r', linewidth=1)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.set_title(title)
            
            # Invert y-axis
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting drawdown: {str(e)}")
            return None
    
    def plot_returns_distribution(self, capital_history_df, title="Returns Distribution"):
        """
        Plot returns distribution
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
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
            
            # Drop NaN values
            df = df.dropna(subset=['return'])
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot returns distribution
            sns.histplot(df['return'], kde=True, ax=ax)
            
            # Add normal distribution for comparison
            x = np.linspace(df['return'].min(), df['return'].max(), 100)
            y = pd.Series(x).plot.kde()
            
            # Set labels and title
            ax.set_xlabel('Return')
            ax.set_ylabel('Frequency')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting returns distribution: {str(e)}")
            return None
    
    def plot_monthly_returns(self, capital_history_df, title="Monthly Returns"):
        """
        Plot monthly returns heatmap
        
        Args:
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
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
            
            # Calculate daily returns
            df['return'] = df['capital'].pct_change()
            
            # Extract year and month
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            
            # Calculate monthly returns
            monthly_returns = df.groupby(['year', 'month'])['return'].sum().unstack()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot heatmap
            sns.heatmap(monthly_returns, annot=True, fmt='.2%', cmap='RdYlGn', center=0, ax=ax)
            
            # Set labels and title
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            ax.set_title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting monthly returns: {str(e)}")
            return None
    
    def plot_trades(self, trades_df, capital_history_df, title="Trading Activity"):
        """
        Plot trades on equity curve
        
        Args:
            trades_df (pandas.DataFrame): DataFrame with trade data
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Create copies to avoid modifying the originals
            trades = trades_df.copy()
            capital = capital_history_df.copy()
            
            # Ensure 'timestamp' is datetime
            if 'timestamp' in capital.columns:
                capital['timestamp'] = pd.to_datetime(capital['timestamp'])
                
            # Ensure 'capital' column exists
            if 'capital' not in capital.columns and 'net_worth' in capital.columns:
                capital['capital'] = capital['net_worth']
            
            # Sort by timestamp
            capital = capital.sort_values('timestamp')
            
            # Convert trade timestamps to datetime
            if 'date' in trades.columns:
                trades['date'] = pd.to_datetime(trades['date'])
                
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot capital history
            ax.plot(capital['timestamp'], capital['capital'], linewidth=2)
            
            # Extract buy and sell trades
            buy_trades = trades[trades['type'] == 'buy']
            sell_trades = trades[trades['type'] == 'sell']
            
            # Plot buy trades
            if 'date' in buy_trades.columns and 'price' in buy_trades.columns:
                for _, trade in buy_trades.iterrows():
                    # Find closest capital history point
                    idx = (capital['timestamp'] - trade['date']).abs().idxmin()
                    
                    # Plot buy marker
                    ax.plot(capital.loc[idx, 'timestamp'], capital.loc[idx, 'capital'], 'g^', markersize=10)
            
            # Plot sell trades
            if 'date' in sell_trades.columns and 'price' in sell_trades.columns:
                for _, trade in sell_trades.iterrows():
                    # Find closest capital history point
                    idx = (capital['timestamp'] - trade['date']).abs().idxmin()
                    
                    # Determine marker color based on profit
                    marker_color = 'g' if trade.get('profit', 0) > 0 else 'r'
                    
                    # Plot sell marker
                    ax.plot(capital.loc[idx, 'timestamp'], capital.loc[idx, 'capital'], f'{marker_color}v', markersize=10)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
            
            # Set labels and title
            ax.set_xlabel('Date')
            ax.set_ylabel('Capital (₹)')
            ax.set_title(title)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(['Equity Curve', 'Buy', 'Sell Profit', 'Sell Loss'])
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting trades: {str(e)}")
            return None
    
    def plot_performance_summary(self, metrics, title="Performance Summary"):
        """
        Plot performance summary
        
        Args:
            metrics (dict): Performance metrics
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        try:
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Flatten axes array
            axes = axes.flatten()
            
            # Plot key metrics as bar chart
            key_metrics = {
                'Total Return': metrics.get('total_return', 0),
                'Annual Return': metrics.get('annual_return', 0),
                'Max Drawdown': -metrics.get('max_drawdown', 0),  # Negative to show as a loss
                'Sharpe Ratio': metrics.get('sharpe_ratio', 0)
            }
            
            # Create bar colors based on values
            bar_colors = ['g' if v >= 0 else 'r' for v in key_metrics.values()]
            
            # Plot bar chart
            bars = axes[0].bar(key_metrics.keys(), key_metrics.values(), color=bar_colors)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.2%}' if abs(height) < 10 else f'{height:.2f}',
                          ha='center', va='bottom')
                else:
                    axes[0].text(bar.get_x() + bar.get_width()/2., height - 0.01,
                          f'{height:.2%}' if abs(height) < 10 else f'{height:.2f}',
                          ha='center', va='top')
            
            # Set title and grid
            axes[0].set_title('Key Metrics')
            axes[0].grid(True, alpha=0.3)
            
            # Plot win rate as pie chart
            win_rate = metrics.get('win_rate', 0)
            win_labels = ['Winning Trades', 'Losing Trades']
            win_sizes = [win_rate, 1 - win_rate]
            win_colors = ['g', 'r']
            
            axes[1].pie(win_sizes, labels=win_labels, colors=win_colors, autopct='%1.1f%%', startangle=90)
            axes[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
            axes[1].set_title('Win Rate')
            
            # Plot average trade as bar chart
            avg_metrics = {
                'Avg Profit': metrics.get('avg_profit_pct', 0),
                'Avg Loss': metrics.get('avg_loss_pct', 0),
                'Avg Trade': metrics.get('avg_trade_pct', 0)
            }
            
            # Create bar colors based on values
            avg_bar_colors = ['g' if v >= 0 else 'r' for v in avg_metrics.values()]
            
            # Plot bar chart
            avg_bars = axes[2].bar(avg_metrics.keys(), avg_metrics.values(), color=avg_bar_colors)
            
            # Add value labels
            for bar in avg_bars:
                height = bar.get_height()
                if height >= 0:
                    axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                          f'{height:.2%}',
                          ha='center', va='bottom')
                else:
                    axes[2].text(bar.get_x() + bar.get_width()/2., height - 0.001,
                          f'{height:.2%}',
                          ha='center', va='top')
            
            # Set title and grid
            axes[2].set_title('Average Trade Performance')
            axes[2].grid(True, alpha=0.3)
            
            # Plot trade count
            trade_counts = {
                'Total Trades': metrics.get('total_trades', 0),
                'Winning Trades': metrics.get('winning_trades', 0),
                'Losing Trades': metrics.get('losing_trades', 0)
            }
            
            # Plot bar chart
            axes[3].bar(trade_counts.keys(), trade_counts.values(), color='blue')
            
            # Add value labels
            for i, v in enumerate(trade_counts.values()):
                axes[3].text(i, v + 0.5, str(int(v)), ha='center')
            
            # Set title and grid
            axes[3].set_title('Trade Count')
            axes[3].grid(True, alpha=0.3)
            
            # Set main title
            fig.suptitle(title, fontsize=16)
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)
            
            return fig
            
        except Exception as e:
            logger.exception(f"Error plotting performance summary: {str(e)}")
            return None
    
    def figure_to_base64(self, fig):
        """
        Convert matplotlib figure to base64 encoded string
        
        Args:
            fig: Matplotlib figure
            
        Returns:
            str: Base64 encoded string
        """
        try:
            if fig is None:
                return None
                
            # Create buffer
            buf = io.BytesIO()
            
            # Save figure to buffer
            fig.savefig(buf, format='png', dpi=100)
            
            # Get buffer value
            buf.seek(0)
            
            # Encode buffer to base64
            img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Close buffer
            buf.close()
            
            return img_str
            
        except Exception as e:
            logger.exception(f"Error converting figure to base64: {str(e)}")
            return None
    
    def create_html_report(self, metrics, capital_history_df, trades_df, filename=None):
        """
        Create HTML report
        
        Args:
            metrics (dict): Performance metrics
            capital_history_df (pandas.DataFrame): DataFrame with capital history
            trades_df (pandas.DataFrame): DataFrame with trade data
            filename (str, optional): Filename to save report
            
        Returns:
            str: HTML report
        """
        try:
            # Generate figures
            capital_fig = self.plot_capital_history(capital_history_df)
            drawdown_fig = self.plot_drawdown(capital_history_df)
            returns_fig = self.plot_returns_distribution(capital_history_df)
            trades_fig = self.plot_trades(trades_df, capital_history_df)
            summary_fig = self.plot_performance_summary(metrics)
            
            # Convert figures to base64
            capital_img = self.figure_to_base64(capital_fig)
            drawdown_img = self.figure_to_base64(drawdown_fig)
            returns_img = self.figure_to_base64(returns_fig)
            trades_img = self.figure_to_base64(trades_fig)
            summary_img = self.figure_to_base64(summary_fig)
            
            # Create HTML report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Trading Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    h1, h2 {{ color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; }}
                    .metrics {{ display: flex; flex-wrap: wrap; }}
                    .metric {{ width: 25%; padding: 10px; box-sizing: border-box; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; }}
                    .metric-name {{ font-size: 14px; color: #666; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart {{ margin-bottom: 30px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Trading Performance Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="section">
                        <h2>Summary</h2>
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('total_return', 0) >= 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</div>
                                <div class="metric-name">Total Return</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('annual_return', 0) >= 0 else 'negative'}">{metrics.get('annual_return', 0):.2%}</div>
                                <div class="metric-name">Annual Return</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2%}</div>
                                <div class="metric-name">Max Drawdown</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('sharpe_ratio', 0) >= 0 else 'negative'}">{metrics.get('sharpe_ratio', 0):.2f}</div>
                                <div class="metric-name">Sharpe Ratio</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('sortino_ratio', 0) >= 0 else 'negative'}">{metrics.get('sortino_ratio', 0):.2f}</div>
                                <div class="metric-name">Sortino Ratio</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('calmar_ratio', 0) >= 0 else 'negative'}">{metrics.get('calmar_ratio', 0):.2f}</div>
                                <div class="metric-name">Calmar Ratio</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">{metrics.get('win_rate', 0):.2%}</div>
                                <div class="metric-name">Win Rate</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value">{metrics.get('profit_factor', 0):.2f}</div>
                                <div class="metric-name">Profit Factor</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Performance Charts</h2>
                        <div class="chart">
                            <img src="data:image/png;base64,{summary_img}" alt="Performance Summary" style="width: 100%;" />
                        </div>
                        <div class="chart">
                            <img src="data:image/png;base64,{capital_img}" alt="Capital History" style="width: 100%;" />
                        </div>
                        <div class="chart">
                            <img src="data:image/png;base64,{drawdown_img}" alt="Drawdown Analysis" style="width: 100%;" />
                        </div>
                        <div class="chart">
                            <img src="data:image/png;base64,{returns_img}" alt="Returns Distribution" style="width: 100%;" />
                        </div>
                        <div class="chart">
                            <img src="data:image/png;base64,{trades_img}" alt="Trading Activity" style="width: 100%;" />
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Trade Metrics</h2>
                        <div class="metrics">
                            <div class="metric">
                                <div class="metric-value">{metrics.get('total_trades', 0)}</div>
                                <div class="metric-name">Total Trades</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value positive">{metrics.get('winning_trades', 0)}</div>
                                <div class="metric-name">Winning Trades</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value negative">{metrics.get('losing_trades', 0)}</div>
                                <div class="metric-name">Losing Trades</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('avg_profit', 0) >= 0 else 'negative'}">{metrics.get('avg_profit', 0):.2f}</div>
                                <div class="metric-name">Avg Profit</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('avg_loss', 0) >= 0 else 'negative'}">{metrics.get('avg_loss', 0):.2f}</div>
                                <div class="metric-name">Avg Loss</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value {'positive' if metrics.get('avg_trade', 0) >= 0 else 'negative'}">{metrics.get('avg_trade', 0):.2f}</div>
                                <div class="metric-name">Avg Trade</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value positive">{metrics.get('largest_profit', 0):.2f}</div>
                                <div class="metric-name">Largest Profit</div>
                            </div>
                            <div class="metric">
                                <div class="metric-value negative">{metrics.get('largest_loss', 0):.2f}</div>
                                <div class="metric-name">Largest Loss</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Recent Trades</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Type</th>
                                    <th>Price</th>
                                    <th>Shares</th>
                                    <th>P/L</th>
                                    <th>P/L %</th>
                                    <th>Exit Type</th>
                                </tr>
                            </thead>
                            <tbody>
                                {self._generate_trade_rows(trades_df)}
                            </tbody>
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save to file if filename provided
            if filename:
                with open(filename, 'w') as f:
                    f.write(html)
                logger.info(f"HTML report saved to {filename}")
                
            return html
            
        except Exception as e:
            logger.exception(f"Error creating HTML report: {str(e)}")
            return ""
    
    def _generate_trade_rows(self, trades_df, max_rows=20):
        """
        Generate HTML table rows for trades
        
        Args:
            trades_df (pandas.DataFrame): DataFrame with trade data
            max_rows (int): Maximum number of rows to generate
            
        Returns:
            str: HTML table rows
        """
        try:
            # Create a copy to avoid modifying the original
            df = trades_df.copy()
            
            # Sort by date/step (descending)
            if 'date' in df.columns:
                df = df.sort_values('date', ascending=False)
            elif 'step' in df.columns:
                df = df.sort_values('step', ascending=False)
                
            # Only include sell trades (completed trades)
            df = df[df['type'] == 'sell']
            
            # Limit to max_rows
            df = df.head(max_rows)
            
            # Generate HTML rows
            rows = ""
            for _, trade in df.iterrows():
                # Format date
                date_str = trade.get('date', 'N/A')
                if isinstance(date_str, (datetime, pd.Timestamp)):
                    date_str = date_str.strftime('%Y-%m-%d')
                    
                # Format price
                price = trade.get('price', 0)
                price_str = f"₹{price:.2f}" if price else 'N/A'
                
                # Format shares
                shares = trade.get('shares', 0)
                
                # Format profit/loss
                profit = trade.get('profit', 0)
                profit_str = f"₹{profit:.2f}" if profit else 'N/A'
                
                # Format profit/loss percentage
                profit_pct = trade.get('profit_pct', 0)
                profit_pct_str = f"{profit_pct:.2%}" if profit_pct else 'N/A'
                
                # Format exit type
                exit_type = trade.get('exit_type', 'N/A')
                
                # Set class based on profit
                profit_class = 'positive' if profit > 0 else 'negative'
                
                # Generate row
                rows += f"""
                <tr>
                    <td>{date_str}</td>
                    <td>{trade.get('type', 'N/A').upper()}</td>
                    <td>{price_str}</td>
                    <td>{shares}</td>
                    <td class="{profit_class}">{profit_str}</td>
                    <td class="{profit_class}">{profit_pct_str}</td>
                    <td>{exit_type}</td>
                </tr>
                """
                
            return rows
            
        except Exception as e:
            logger.exception(f"Error generating trade rows: {str(e)}")
            return ""