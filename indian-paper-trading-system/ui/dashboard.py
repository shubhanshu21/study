import dash
from dash import dcc, html, callback, Output, Input, dash_table
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config.trading_config import DASHBOARD_PORT

logger = logging.getLogger(__name__)

def create_dashboard(trading_env, scheduler, strategies, symbols):
    """Create Dash app for trading dashboard"""
    app = dash.Dash(__name__, title="Indian Paper Trading System")
    
    # Layout
    app.layout = html.Div([
        html.H1("Indian Paper Trading System Dashboard", style={'textAlign': 'center'}),
        
        # Market Status and Account Overview
        html.Div([
            html.Div([
                html.H3("Market Status"),
                html.Div(id="market-status"),
                html.Div(id="market-time"),
                html.Button("Refresh Data", id="refresh-button", n_clicks=0),
            ], className="six columns"),
            
            html.Div([
                html.H3("Account Overview"),
                html.Div(id="account-overview"),
            ], className="six columns"),
        ], className="row"),
        
        html.Hr(),
        
        # Positions Table
        html.Div([
            html.H3("Current Positions"),
            html.Div(id="positions-table"),
        ]),
        
        html.Hr(),
        
        # Recent Orders
        html.Div([
            html.H3("Recent Orders"),
            html.Div(id="orders-table"),
        ]),
        
        html.Hr(),
        
        # Performance Metrics
        html.Div([
            html.H3("Performance Metrics"),
            html.Div(id="performance-metrics"),
        ]),
        
        html.Hr(),
        
        # Portfolio Value Chart
        html.Div([
            html.H3("Portfolio Value History"),
            dcc.Graph(id="portfolio-chart"),
        ]),
        
        html.Hr(),
        
        # Symbol Selection for Charts
        html.Div([
            html.H3("Symbol Chart"),
            dcc.Dropdown(
                id="symbol-dropdown",
                options=[{"label": s, "value": s} for s in symbols],
                value=symbols[0] if symbols else None,
            ),
            dcc.Graph(id="symbol-chart"),
        ]),
        
        # Update interval
        dcc.Interval(
            id="interval-component",
            interval=30 * 1000,  # in milliseconds (30 seconds)
            n_intervals=0
        ),
    ])
    
    @app.callback(
        [
            Output("market-status", "children"),
            Output("market-time", "children"),
            Output("account-overview", "children"),
            Output("positions-table", "children"),
            Output("orders-table", "children"),
            Output("performance-metrics", "children"),
            Output("portfolio-chart", "figure"),
        ],
        [
            Input("interval-component", "n_intervals"),
            Input("refresh-button", "n_clicks")
        ]
    )
    def update_dashboard(n_intervals, n_clicks):
        """Update dashboard components"""
        # Get current market status
        market_status = "Open" if scheduler.market_config.is_market_open() else "Closed"
        status_color = "green" if market_status == "Open" else "red"
        
        market_status_div = html.Div([
            html.Span("Status: "),
            html.Span(market_status, style={"color": status_color, "fontWeight": "bold"}),
        ])
        
        # Current time
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        market_time_div = html.Div(f"Current Time: {time_str}")
        
        # Account overview
        portfolio = trading_env.get_portfolio_summary()
        
        account_overview_div = html.Div([
            html.Div([
                html.Span("Portfolio Value: "),
                html.Span(f"₹{portfolio['portfolio_value']:.2f}", style={"fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Cash Balance: "),
                html.Span(f"₹{portfolio['current_balance']:.2f}", style={"fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Position Value: "),
                html.Span(f"₹{portfolio['total_position_value']:.2f}", style={"fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Open Positions: "),
                html.Span(f"{portfolio['position_count']}", style={"fontWeight": "bold"}),
            ]),
            html.Div([
                html.Span("Cash Ratio: "),
                html.Span(f"{portfolio['cash_ratio']:.2%}", style={"fontWeight": "bold"}),
            ]),
        ])
        
        # Positions table
        if portfolio['positions']:
            positions_data = portfolio['positions']
            positions_df = pd.DataFrame(positions_data)
            positions_table = dash_table.DataTable(
                data=positions_df.to_dict('records'),
                columns=[
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Quantity", "id": "quantity"},
                    {"name": "Avg Price", "id": "avg_price", "type": "numeric", "format": {"specifier": "₹,.2f"}},
                    {"name": "Current Price", "id": "current_price", "type": "numeric", "format": {"specifier": "₹,.2f"}},
                    {"name": "Value", "id": "value", "type": "numeric", "format": {"specifier": "₹,.2f"}},
                    {"name": "P&L", "id": "unrealized_pnl", "type": "numeric", "format": {"specifier": "₹,.2f"}},
                    {"name": "P&L %", "id": "unrealized_pnl_pct", "type": "numeric", "format": {"specifier": ".2%"}},
                    {"name": "Days Held", "id": "days_held", "type": "numeric"},
                ],
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{unrealized_pnl} > 0',
                            'column_id': 'unrealized_pnl'
                        },
                        'color': 'green'
                    },
                    {
                        'if': {
                            'filter_query': '{unrealized_pnl} < 0',
                            'column_id': 'unrealized_pnl'
                        },
                        'color': 'red'
                    },
                    {
                        'if': {
                            'filter_query': '{unrealized_pnl_pct} > 0',
                            'column_id': 'unrealized_pnl_pct'
                        },
                        'color': 'green'
                    },
                    {
                        'if': {
                            'filter_query': '{unrealized_pnl_pct} < 0',
                            'column_id': 'unrealized_pnl_pct'
                        },
                        'color': 'red'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'center'
                }
            )
        else:
            positions_table = html.Div("No open positions")
        
        # Recent orders
        if trading_env.orders:
            # Get the last 10 orders
            recent_orders = trading_env.orders[-10:]
            orders_df = pd.DataFrame(recent_orders)
            
            # Handle nested transaction_costs
            if 'transaction_costs' in orders_df.columns:
                orders_df['txn_cost'] = orders_df['transaction_costs'].apply(
                    lambda x: x['total'] if isinstance(x, dict) and 'total' in x else 0
                )
            
            orders_table = dash_table.DataTable(
                data=orders_df.to_dict('records'),
                columns=[
                    {"name": "Order ID", "id": "order_id"},
                    {"name": "Symbol", "id": "symbol"},
                    {"name": "Type", "id": "order_type"},
                    {"name": "Quantity", "id": "quantity"},
                    {"name": "Price", "id": "price", "type": "numeric", "format": {"specifier": "₹,.2f"}},
                    {"name": "Status", "id": "status"},
                    {"name": "Create Time", "id": "create_time"},
                ],
                style_data_conditional=[
                    {
                        'if': {
                            'filter_query': '{order_type} = "BUY"',
                            'column_id': 'order_type'
                        },
                        'color': 'green'
                    },
                    {
                        'if': {
                            'filter_query': '{order_type} = "SELL"',
                            'column_id': 'order_type'
                        },
                        'color': 'red'
                    },
                    {
                        'if': {
                            'filter_query': '{status} = "EXECUTED"',
                            'column_id': 'status'
                        },
                        'color': 'green'
                    },
                    {
                        'if': {
                            'filter_query': '{status} = "PENDING"',
                            'column_id': 'status'
                        },
                        'color': 'orange'
                    },
                    {
                        'if': {
                            'filter_query': '{status} = "REJECTED" || {status} = "CANCELLED"',
                            'column_id': 'status'
                        },
                        'color': 'red'
                    }
                ],
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'textAlign': 'center'
                }
            )
        else:
            orders_table = html.Div("No orders")
        
        # Performance metrics
        metrics = trading_env.get_performance_metrics()
        
        metrics_div = html.Div([
            html.Div([
                html.Div([
                    html.Span("Total Return: "),
                    html.Span(f"{metrics['total_return']:.2%}", 
                              style={"color": "green" if metrics['total_return'] > 0 else "red", "fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Win Rate: "),
                    html.Span(f"{metrics['win_rate']:.2%}", style={"fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Profit Factor: "),
                    html.Span(f"{metrics['profit_factor']:.2f}", style={"fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Sharpe Ratio: "),
                    html.Span(f"{metrics['sharpe_ratio']:.2f}", style={"fontWeight": "bold"}),
                ], className="three columns"),
            ], className="row"),
            
            html.Div([
                html.Div([
                    html.Span("Max Drawdown: "),
                    html.Span(f"{metrics['max_drawdown']:.2%}", style={"color": "red", "fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Total Trades: "),
                    html.Span(f"{metrics['total_trades']}", style={"fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Wins/Losses: "),
                    html.Span(f"{metrics['win_count']}/{metrics['loss_count']}", style={"fontWeight": "bold"}),
                ], className="three columns"),
                
                html.Div([
                    html.Span("Avg Win/Loss Ratio: "),
                    html.Span(f"{metrics['avg_win_loss_ratio']:.2f}", style={"fontWeight": "bold"}),
                ], className="three columns"),
            ], className="row"),
        ])
        
        # Portfolio chart
        net_worth_history = trading_env.net_worth_history
        
        if len(net_worth_history) > 1:
            portfolio_fig = go.Figure()
            
            portfolio_fig.add_trace(go.Scatter(
                y=net_worth_history,
                mode='lines',
                name='Portfolio Value',
                line=dict(color='rgb(0, 100, 80)')
            ))
            
            portfolio_fig.update_layout(
                title="Portfolio Value History",
                xaxis_title="Time",
                yaxis_title="Value (₹)",
                template="plotly_white",
                height=400,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            # Add initial balance reference line
            portfolio_fig.add_shape(
                type="line",
                x0=0,
                y0=trading_env.initial_balance,
                x1=len(net_worth_history) - 1,
                y1=trading_env.initial_balance,
                line=dict(
                    color="rgba(255, 0, 0, 0.5)",
                    width=2,
                    dash="dash",
                )
            )
        else:
            portfolio_fig = go.Figure()
            portfolio_fig.update_layout(
                title="Portfolio Value History (No Data Yet)",
                height=400
            )
        
        return market_status_div, market_time_div, account_overview_div, positions_table, orders_table, metrics_div, portfolio_fig
    
    @app.callback(
        Output("symbol-chart", "figure"),
        [
            Input("symbol-dropdown", "value"),
            Input("interval-component", "n_intervals")
        ]
    )
    def update_symbol_chart(symbol, n_intervals):
        """Update individual symbol chart"""
        if not symbol or symbol not in trading_env.historical_data:
            return go.Figure()
        
        # Get historical data
        df = trading_env.historical_data[symbol]
        
        # Create candlestick chart
        fig = go.Figure()
        
        # Add candlesticks
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
        
        # Add EMA lines if available
        if 'EMA20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['EMA20'],
                mode='lines',
                name='EMA20',
                line=dict(color='rgba(255, 165, 0, 0.7)')
            ))
        
        if 'EMA50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['EMA50'],
                mode='lines',
                name='EMA50',
                line=dict(color='rgba(0, 0, 255, 0.7)')
            ))
        
        # Add Bollinger Bands if available
        if all(x in df.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                mode='lines',
                name='BB Upper',
                line=dict(color='rgba(173, 216, 230, 0.7)')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Middle'],
                mode='lines',
                name='BB Middle',
                line=dict(color='rgba(70, 130, 180, 0.7)')
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                mode='lines',
                name='BB Lower',
                line=dict(color='rgba(173, 216, 230, 0.7)')
            ))
        
        # Add buys and sells
        buy_indices = []
        buy_prices = []
        sell_indices = []
        sell_prices = []
        
        for trade in trading_env.trades:
            if trade['symbol'] == symbol:
                if trade['action'] == 'BUY':
                    # Find the index for this date
                    try:
                        if 'Date' in df.columns:
                            matching_idx = df[df['Date'] == trade['date']].index[0]
                            buy_indices.append(matching_idx)
                            buy_prices.append(trade['price'])
                    except (IndexError, KeyError):
                        pass
                elif trade['action'] == 'SELL':
                    try:
                        if 'Date' in df.columns:
                            matching_idx = df[df['Date'] == trade['date']].index[0]
                            sell_indices.append(matching_idx)
                            sell_prices.append(trade['price'])
                    except (IndexError, KeyError):
                        pass
        
        if buy_indices:
            fig.add_trace(go.Scatter(
                x=buy_indices,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(
                    color='green',
                    size=10,
                    symbol='triangle-up'
                )
            ))
        
        if sell_indices:
            fig.add_trace(go.Scatter(
                x=sell_indices,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='triangle-down'
                )
            ))
        
        # Add current position info
        if symbol in trading_env.positions:
            position = trading_env.positions[symbol]
            
            # Add entry price line
            fig.add_shape(
                type="line",
                x0=0,
                y0=position['avg_price'],
                x1=len(df) - 1,
                y1=position['avg_price'],
                line=dict(
                    color="rgba(0, 128, 0, 0.7)",
                    width=2,
                    dash="dash",
                )
            )
            
            # Add stop loss if exists
            if position.get('stop_loss'):
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=position['stop_loss'],
                    x1=len(df) - 1,
                    y1=position['stop_loss'],
                    line=dict(
                        color="rgba(255, 0, 0, 0.7)",
                        width=2,
                        dash="dash",
                    )
                )
            
            # Add target if exists
            if position.get('target'):
                fig.add_shape(
                    type="line",
                    x0=0,
                    y0=position['target'],
                    x1=len(df) - 1,
                    y1=position['target'],
                    line=dict(
                        color="rgba(0, 255, 0, 0.7)",
                        width=2,
                        dash="dash",
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{symbol} Price Chart",
            xaxis_title="Time",
            yaxis_title="Price (₹)",
            template="plotly_white",
            height=600,
            xaxis_rangeslider_visible=False,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=40, r=40, t=40, b=40)
        )
        
        return fig
    
    return app