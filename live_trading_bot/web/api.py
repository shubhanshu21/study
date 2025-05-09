"""
API endpoints for the trading bot.
"""
import logging
import json
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS

logger = logging.getLogger(__name__)

class TradingAPI:
    """
    API endpoints for the trading bot.
    Provides a web interface for monitoring and controlling the bot.
    """
    
    def __init__(self, broker, trading_strategies, db_service, config, notification_service):
        """
        Initialize the API
        
        Args:
            broker: Broker instance
            trading_strategies (dict): Dictionary of symbol -> strategy instances
            db_service: Database service instance
            config: Application configuration
            notification_service: Notification service instance
        """
        self.broker = broker
        self.trading_strategies = trading_strategies
        self.db_service = db_service
        self.config = config
        self.notification_service = notification_service
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for all routes
        
        # Register routes
        self._register_routes()
        
        logger.info("API initialized")
    
    def _register_routes(self):
        """Register API routes"""
        # Health check
        @self.app.route('/api/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'ok',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
        
        # Account status
        @self.app.route('/api/status', methods=['GET'])
        def account_status():
            try:
                # Get account info
                funds = self.broker.get_funds()
                positions = self.broker.get_positions()
                
                # Build response
                response = {
                    'timestamp': datetime.now().isoformat(),
                    'account': {
                        'balance': funds.get('balance', 0) if funds else 0,
                        'used_margin': funds.get('used_margin', 0) if funds else 0
                    },
                    'positions': positions,
                    'strategies': {}
                }
                
                # Add strategy metrics
                for symbol, strategy in self.trading_strategies.items():
                    response['strategies'][symbol] = strategy.get_performance_metrics()
                
                return jsonify(response)
                
            except Exception as e:
                logger.exception(f"Error getting account status: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        # Strategy status
        @self.app.route('/api/strategy/<symbol>', methods=['GET'])
        def strategy_status(symbol):
            try:
                if symbol not in self.trading_strategies:
                    return jsonify({
                        'error': f"Strategy not found for symbol: {symbol}",
                        'timestamp': datetime.now().isoformat()
                    }), 404
                    
                strategy = self.trading_strategies[symbol]
                
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'metrics': strategy.get_performance_metrics()
                })
                
            except Exception as e:
                logger.exception(f"Error getting strategy status: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        # Get trades
        @self.app.route('/api/trades', methods=['GET'])
        def get_trades():
            try:
                # Parse query parameters
                symbol = request.args.get('symbol')
                start_date = request.args.get('start_date')
                end_date = request.args.get('end_date')
                
                # Get trades from database
                trades_df = self.db_service.get_trades(symbol, start_date, end_date)
                
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'trades': trades_df.to_dict('records'),
                    'count': len(trades_df)
                })
                
            except Exception as e:
                logger.exception(f"Error getting trades: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        # Get performance
        @self.app.route('/api/performance', methods=['GET'])
        def get_performance():
            try:
                # Parse query parameters
                symbol = request.args.get('symbol')
                days = request.args.get('days', 30)
                
                # Calculate date range
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=int(days))).strftime('%Y-%m-%d')
                
                # Get performance from database
                performance_df = self.db_service.get_daily_performance(symbol, start_date, end_date)
                
                return jsonify({
                    'timestamp': datetime.now().isoformat(),
                    'performance': performance_df.to_dict('records'),
                    'start_date': start_date,
                    'end_date': end_date
                })
                
            except Exception as e:
                logger.exception(f"Error getting performance: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        # Manual trade
        @self.app.route('/api/trade', methods=['POST'])
        def manual_trade():
            try:
                # Parse request body
                data = request.json
                
                symbol = data.get('symbol')
                action = data.get('action')  # 'buy' or 'sell'
                quantity = int(data.get('quantity', 0))
                
                if not symbol or not action or quantity <= 0:
                    return jsonify({
                        'error': "Invalid request parameters",
                        'timestamp': datetime.now().isoformat()
                    }), 400
                    
                if symbol not in self.trading_strategies:
                    return jsonify({
                        'error': f"Strategy not found for symbol: {symbol}",
                        'timestamp': datetime.now().isoformat()
                    }), 404
                    
                strategy = self.trading_strategies[symbol]
                
                # Execute trade
                if action.lower() == 'buy':
                    if strategy.position_type:
                        return jsonify({
                            'error': f"Already have a position for {symbol}",
                            'timestamp': datetime.now().isoformat()
                        }), 400
                        
                    # Place buy order
                    order_id = self.broker.place_order(
                        symbol=symbol,
                        transaction_type='BUY',
                        quantity=quantity,
                        order_type='MARKET',
                        tag="MANUAL_BUY"
                    )
                    
                    if not order_id:
                        return jsonify({
                            'error': f"Failed to place buy order for {symbol}",
                            'timestamp': datetime.now().isoformat()
                        }), 500
                        
                    # Update strategy state
                    current_price = self.broker.get_ltp(symbol)
                    strategy.position_type = 'long'
                    strategy.entry_price = current_price
                    strategy.entry_time = datetime.now()
                    strategy.days_in_trade = 0
                    
                    # Set up risk management
                    strategy._setup_risk_management()
                    
                    # Notify
                    self.notification_service.send_alert(
                        f"Manual Buy Order: {symbol}",
                        f"Placed manual buy order for {quantity} shares of {symbol} at ~{current_price}"
                    )
                    
                    return jsonify({
                        'timestamp': datetime.now().isoformat(),
                        'message': f"Buy order placed for {quantity} shares of {symbol}",
                        'order_id': order_id
                    })
                    
                elif action.lower() == 'sell':
                    # Get current position
                    position = self.broker.get_position(symbol)
                    
                    if not position or position.get('quantity', 0) <= 0:
                        return jsonify({
                            'error': f"No position to sell for {symbol}",
                            'timestamp': datetime.now().isoformat()
                        }), 400
                        
                    # Adjust quantity if needed
                    shares_to_sell = min(quantity, position.get('quantity', 0))
                    
                    # Place sell order
                    order_id = self.broker.place_order(
                        symbol=symbol,
                        transaction_type='SELL',
                        quantity=shares_to_sell,
                        order_type='MARKET',
                        tag="MANUAL_SELL"
                    )
                    
                    if not order_id:
                        return jsonify({
                            'error': f"Failed to place sell order for {symbol}",
                            'timestamp': datetime.now().isoformat()
                        }), 500
                        
                    # Update strategy state
                    current_price = self.broker.get_ltp(symbol)
                    
                    # Calculate profit/loss
                    entry_price = strategy.entry_price or position.get('average_price', 0)
                    profit_loss = (current_price - entry_price) * shares_to_sell
                    profit_loss_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
                    
                    # Reset position state if selling all
                    if shares_to_sell >= position.get('quantity', 0):
                        strategy.position_type = None
                        strategy.entry_price = 0
                        strategy.entry_time = None
                        strategy.stop_loss = 0
                        strategy.trailing_stop = 0
                        strategy.target_price = 0
                    
                    # Notify
                    self.notification_service.send_alert(
                        f"Manual Sell Order: {symbol}",
                        f"Placed manual sell order for {shares_to_sell} shares of {symbol} at ~{current_price}\n"
                        f"P/L: {profit_loss:.2f} ({profit_loss_pct:.2%})"
                    )
                    
                    return jsonify({
                        'timestamp': datetime.now().isoformat(),
                        'message': f"Sell order placed for {shares_to_sell} shares of {symbol}",
                        'order_id': order_id
                    })
                    
                else:
                    return jsonify({
                        'error': f"Invalid action: {action}",
                        'timestamp': datetime.now().isoformat()
                    }), 400
                    
            except Exception as e:
                logger.exception(f"Error executing manual trade: {str(e)}")
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        logger.info("API routes registered")
    
    def start(self, host='0.0.0.0', port=5000):
        """
        Start the API server
        
        Args:
            host (str): Host to bind the server to
            port (int): Port to bind the server to
        """
        logger.info(f"Starting API server on {host}:{port}")
        self.app.run(host=host, port=port, threaded=True)