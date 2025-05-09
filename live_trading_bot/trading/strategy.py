"""
RL-based trading strategy implementation.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class RLTradingStrategy:
    """
    Reinforcement learning-based trading strategy implementation.
    Uses a trained model to make trading decisions.
    """
    
    def __init__(self, symbol, model, data_loader, regime_detector, broker, config):
        """
        Initialize the trading strategy
        
        Args:
            symbol (str): Trading symbol
            model: Trained RL model
            data_loader: Data loader instance
            regime_detector: Market regime detector
            broker: Broker instance for order execution
            config: Application configuration
        """
        self.symbol = symbol
        self.model = model
        self.data_loader = data_loader
        self.regime_detector = regime_detector
        self.broker = broker
        self.config = config
        
        # Strategy state
        self.position = None
        self.entry_price = 0
        self.entry_time = None
        self.stop_loss = 0
        self.trailing_stop = 0
        self.target_price = 0
        self.stop_loss_order_id = None
        self.target_order_id = None
        self.highest_price_since_buy = 0
        self.position_type = None
        self.days_in_trade = 0
        self.regime_params = None
        self.current_market_regime = "unknown"
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        # Technical signals
        self.buy_signal_strength = 0
        self.sell_signal_strength = 0
        
        # Risk management
        self.daily_trades = 0
        self.daily_profit = 0
        self.daily_loss = 0
        self.max_drawdown = 0
        self.max_capital = 0
        self.position_size_pct = config.MAX_POSITION_SIZE_PCT
        self.trading_stopped_for_day = False
        self.trading_stopped_reason = None
        
        # Current data
        self.df = None
        self.current_step = 0
        self.last_update_time = None
        
        # Initialize data
        self._init_data()
        
        logger.info(f"Initialized RL trading strategy for {symbol}")
    
    def _init_data(self):
        """Initialize data for the strategy"""
        try:
            # Get historical data for initialization
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)  # Get 30 days of data
            
            # Get historical data
            data = self.data_loader.get_historical_data(
                symbol=self.symbol,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d'),
                interval='5minute'  # Use 5-minute candles
            )
            
            if not data:
                logger.error(f"Could not initialize data for {self.symbol}")
                return
                
            # Convert to dataframe
            self.df = pd.DataFrame(data)
            
            # Calculate technical indicators
            self._calculate_indicators()
            
            # Initialize position info
            position = self.broker.get_position(self.symbol)
            if position and position.get('quantity', 0) > 0:
                self.position = position
                self.position_type = 'long'  # Assuming long position
                self.entry_price = position.get('average_price', 0)
                
                # Set up proper stop loss and target
                self._setup_risk_management()
                
            # Set last update time
            self.last_update_time = datetime.now()
            
            logger.info(f"Initialized data for {self.symbol}: {len(self.df)} candles")
            
        except Exception as e:
            logger.exception(f"Error initializing data for {self.symbol}: {str(e)}")
    
    def _calculate_indicators(self):
        """Calculate technical indicators for the data"""
        try:
            from utils.indicators import calculate_indicators
            
            # Skip if dataframe is empty
            if self.df is None or self.df.empty:
                logger.warning(f"No data available for calculating indicators for {self.symbol}")
                return
                
            # Calculate all technical indicators
            self.df = calculate_indicators(self.df)
            
            logger.debug(f"Calculated technical indicators for {self.symbol}")
            
        except Exception as e:
            logger.exception(f"Error calculating indicators for {self.symbol}: {str(e)}")
    
    def update_data(self):
        """Update data with the latest market information"""
        try:
            # Skip if last update was recent (within 1 minute)
            now = datetime.now()
            if self.last_update_time and (now - self.last_update_time).total_seconds() < 60:
                return False
                
            # Get latest candle
            latest_data = self.data_loader.get_latest_candle(self.symbol)
            
            if not latest_data:
                logger.warning(f"No latest candle available for {self.symbol}")
                return False
                
            # Add to dataframe
            self.df = pd.concat([self.df, pd.DataFrame([latest_data])], ignore_index=True)
            
            # Remove old data to prevent dataframe from growing too large
            if len(self.df) > 500:  # Keep last 500 candles
                self.df = self.df.iloc[-500:]
                
            # Recalculate indicators
            self._calculate_indicators()
            
            # Update current step
            self.current_step = len(self.df) - 1
            
            # Update last update time
            self.last_update_time = now
            
            # Update technical signals
            self._update_technical_signals()
            
            # Update market regime
            self._detect_market_regime()
            
            logger.debug(f"Updated data for {self.symbol}: {len(self.df)} candles")
            return True
            
        except Exception as e:
            logger.exception(f"Error updating data for {self.symbol}: {str(e)}")
            return False
    
    def _detect_market_regime(self):
        """Detect current market regime and update parameters"""
        try:
            if self.df is None or self.df.empty:
                logger.warning(f"No data available for detecting market regime for {self.symbol}")
                return
                
            # Detect market regime
            self.current_market_regime = self.regime_detector.detect_regime(self.df, self.current_step)
            
            # Get optimal parameters for the current regime
            self.regime_params = self.regime_detector.get_regime_parameters(self.current_market_regime)
            
            # Update position size based on market regime
            self.position_size_pct = min(
                self.regime_params.get('position_size_pct', self.config.MAX_POSITION_SIZE_PCT),
                self.config.MAX_POSITION_SIZE_PCT
            )
            
            logger.debug(f"Market regime for {self.symbol}: {self.current_market_regime}")
            
        except Exception as e:
            logger.exception(f"Error detecting market regime for {self.symbol}: {str(e)}")
    
    def _update_technical_signals(self):
        """Calculate and update technical trading signals"""
        try:
            # Reset signals
            self.buy_signal_strength = 0
            self.sell_signal_strength = 0
            
            # Skip if not enough data
            if self.current_step < 20 or self.df is None or self.df.empty:
                return
                
            # Get current window of data
            window = self.df.iloc[self.current_step-20:self.current_step+1]
            current_price = window['close'].iloc[-1]
            
            # 1. Trend signals
            if 'EMA20' in window.columns and 'EMA50' in window.columns:
                # Bullish trend (EMA20 above EMA50)
                if window['EMA20'].iloc[-1] > window['EMA50'].iloc[-1]:
                    self.buy_signal_strength += 1
                    # Bullish crossover (EMA20 crosses above EMA50)
                    if window['EMA20'].iloc[-2] <= window['EMA50'].iloc[-2]:
                        self.buy_signal_strength += 2
                # Bearish trend (EMA20 below EMA50)
                elif window['EMA20'].iloc[-1] < window['EMA50'].iloc[-1]:
                    self.sell_signal_strength += 1
                    # Bearish crossover (EMA20 crosses below EMA50)
                    if window['EMA20'].iloc[-2] >= window['EMA50'].iloc[-2]:
                        self.sell_signal_strength += 2
            
            # 2. Bollinger Band signals
            if all(x in window.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                upper = window['BB_Upper'].iloc[-1]
                middle = window['BB_Middle'].iloc[-1]
                lower = window['BB_Lower'].iloc[-1]
                
                # Lower band touch/bounce (bullish)
                if (window['close'].iloc[-2] <= window['BB_Lower'].iloc[-2] and 
                    window['close'].iloc[-1] > window['BB_Lower'].iloc[-1]):
                    self.buy_signal_strength += 2
                    
                # Upper band touch/bounce (bearish)
                if (window['close'].iloc[-2] >= window['BB_Upper'].iloc[-2] and 
                    window['close'].iloc[-1] < window['BB_Upper'].iloc[-1]):
                    self.sell_signal_strength += 2
            
            # 3. RSI signals
            if 'RSI' in window.columns:
                rsi = window['RSI'].iloc[-1]
                
                # Oversold zone (bullish)
                if rsi < 30:
                    self.buy_signal_strength += 2
                # Overbought zone (bearish)
                elif rsi > 70:
                    self.sell_signal_strength += 2
            
            # 4. MACD signals
            if all(x in window.columns for x in ['MACD', 'Signal']):
                macd = window['MACD'].iloc[-1]
                signal = window['Signal'].iloc[-1]
                prev_macd = window['MACD'].iloc[-2]
                prev_signal = window['Signal'].iloc[-2]
                
                # MACD crossover (bullish)
                if prev_macd < prev_signal and macd > signal:
                    self.buy_signal_strength += 2
                # MACD crossover (bearish)
                elif prev_macd > prev_signal and macd < signal:
                    self.sell_signal_strength += 2
            
            # 5. Volume signals
            if 'volume' in window.columns:
                avg_volume = window['volume'].iloc[-5:].mean()
                current_volume = window['volume'].iloc[-1]
                
                # Volume spike with price movement
                if current_volume > avg_volume * 1.5:
                    # Volume spike with price increase (bullish)
                    if window['close'].iloc[-1] > window['close'].iloc[-2]:
                        self.buy_signal_strength += 1
                    # Volume spike with price decrease (bearish)
                    elif window['close'].iloc[-1] < window['close'].iloc[-2]:
                        self.sell_signal_strength += 1
            
            logger.debug(f"Technical signals for {self.symbol}: Buy={self.buy_signal_strength}, Sell={self.sell_signal_strength}")
            
        except Exception as e:
            logger.exception(f"Error updating technical signals for {self.symbol}: {str(e)}")
    
    def _get_observation(self):
        """Get the current observation vector for the model"""
        try:
            # Skip if not enough data
            if self.current_step < 20 or self.df is None or self.df.empty:
                logger.warning(f"Not enough data to get observation for {self.symbol}")
                return None
                
            # Get current price data
            current_price = self.df.iloc[self.current_step]["close"]
            
            # Calculate position metrics
            position_value = 0
            if self.position_type == 'long':
                position_info = self.broker.get_position(self.symbol)
                if position_info:
                    position_value = position_info.get('quantity', 0) * current_price
            
            # Get account information
            funds = self.broker.get_funds()
            balance = funds.get('balance', 0) if funds else 0
            
            # Basic account state features
            obs = np.array([
                balance / self.config.DEFAULT_INITIAL_BALANCE,  # Normalized balance
                position_value / (balance if balance > 0 else 1),  # Position size relative to balance
                self.entry_price / current_price if self.entry_price > 0 else 0,  # Normalized entry price
                current_price / self.df["close"].iloc[max(0, self.current_step-20):self.current_step+1].mean(),  # Normalized price
                1 if self.position_type == 'long' else 0,  # Is in long position
                0,  # Is in short position (not supported in this implementation)
                self.trailing_stop / current_price if self.trailing_stop > 0 else 0,  # Normalized trailing stop
                self.target_price / current_price if self.target_price > 0 else 0,  # Normalized target
                self.days_in_trade / 10 if self.days_in_trade > 0 else 0,  # Days in trade
                1 if self.trading_stopped_for_day else 0,  # Trading stopped for day
                self.broker.get_positions().get(self.symbol, {}).get('quantity', 0) / 100,  # Normalized position size
                self.daily_trades / self.config.MAX_TRADES_PER_DAY,  # Daily trades ratio
                self.max_drawdown * 10,  # Amplified drawdown signal
                self.consecutive_losses / 5 if self.consecutive_losses > 0 else 0,  # Consecutive losses
                self.consecutive_wins / 5 if self.consecutive_wins > 0 else 0,  # Consecutive wins
                self.buy_signal_strength / 10,  # Buy signal strength
                self.sell_signal_strength / 10,  # Sell signal strength
                # Market regime (one-hot encoding)
                1 if self.current_market_regime == "trending_up" else 0,
                1 if self.current_market_regime == "trending_down" else 0,
                1 if self.current_market_regime == "ranging" else 0,
                1 if self.current_market_regime == "volatile" else 0
            ])
            
            # Add technical indicators from current row
            window_data = self.df.iloc[self.current_step-self.config.WINDOW_SIZE+1:self.current_step+1]
            
            # Normalize OHLC
            close_mean = window_data["close"].mean()
            
            # Add OHLC and volume to observation
            obs = np.append(obs, [
                window_data['open'].iloc[-1] / close_mean,
                window_data['high'].iloc[-1] / close_mean,
                window_data['low'].iloc[-1] / close_mean,
                window_data['close'].iloc[-1] / close_mean,
                window_data['volume'].iloc[-1] / window_data['volume'].mean() if 'volume' in window_data else 0,
            ])
            
            # Add SMA values if available
            for period in [5, 10, 20, 50]:
                if f'SMA{period}' in window_data.columns:
                    sma = window_data[f'SMA{period}'].iloc[-1] / close_mean
                else:
                    sma = window_data["close"].rolling(window=min(period, len(window_data))).mean().iloc[-1] / close_mean
                obs = np.append(obs, [sma])
            
            # Add EMA values if available
            for period in [5, 10, 20, 50]:
                if f'EMA{period}' in window_data.columns:
                    ema = window_data[f'EMA{period}'].iloc[-1] / close_mean
                else:
                    ema = window_data["close"].ewm(span=min(period, len(window_data)), adjust=False).mean().iloc[-1] / close_mean
                obs = np.append(obs, [ema])
            
            # Add RSI if available
            if 'RSI' in window_data.columns:
                rsi = window_data['RSI'].iloc[-1] / 100  # Normalize RSI
            else:
                rsi = 0.5  # Default middle value if not available
            obs = np.append(obs, [rsi])
            
            # Add MACD if available
            if all(x in window_data.columns for x in ['MACD', 'Signal']):
                macd = window_data['MACD'].iloc[-1] / close_mean
                signal = window_data['Signal'].iloc[-1] / close_mean
            else:
                macd = 0
                signal = 0
            obs = np.append(obs, [macd, signal])
            
            # Add Bollinger Bands if available
            if all(x in window_data.columns for x in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
                upper_band = window_data['BB_Upper'].iloc[-1] / close_mean
                middle_band = window_data['BB_Middle'].iloc[-1] / close_mean
                lower_band = window_data['BB_Lower'].iloc[-1] / close_mean
            else:
                upper_band = 1.1
                middle_band = 1.0
                lower_band = 0.9
            obs = np.append(obs, [upper_band, middle_band, lower_band])
            
            # Add price momentum indicators
            ret_1d = window_data['close'].pct_change(1).iloc[-1]
            ret_5d = window_data['close'].pct_change(5).iloc[-1] if len(window_data) > 5 else 0
            ret_20d = window_data['close'].pct_change(20).iloc[-1] if len(window_data) > 20 else 0
            obs = np.append(obs, [ret_1d, ret_5d, ret_20d])
            
            # Ensure observation has the correct length (e.g., 60 elements)
            expected_length = 60
            if len(obs) > expected_length:
                obs = obs[:expected_length]
            elif len(obs) < expected_length:
                obs = np.append(obs, np.zeros(expected_length - len(obs)))
            
            return obs.astype(np.float32)
            
        except Exception as e:
            logger.exception(f"Error getting observation for {self.symbol}: {str(e)}")
            return None
    
    def make_trading_decision(self):
        """Make a trading decision using the trained model"""
        try:
            # Skip if market is closed or trading is stopped for the day
            if not self._is_market_open() or self.trading_stopped_for_day:
                return False
                
            # Update data
            if not self.update_data():
                logger.warning(f"Could not update data for {self.symbol}")
                return False
                
            # Get observation
            obs = self._get_observation()
            if obs is None:
                logger.warning(f"Could not get observation for {self.symbol}")
                return False
                
            # Get model prediction
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Process the action
            return self._process_action(action)
            
        except Exception as e:
            logger.exception(f"Error making trading decision for {self.symbol}: {str(e)}")
            return False
    
    def _process_action(self, action):
        """
        Process the action from the model
        
        Args:
            action (int): Action from the model
                0: Hold
                1: Buy with 25% of allowed position size
                2: Buy with 50% of allowed position size
                3: Buy with 75% of allowed position size
                4: Buy with 100% of allowed position size
                5: Sell
                
        Returns:
            bool: True if action was processed successfully, False otherwise
        """
        try:
            # Get current price
            current_price = self.df.iloc[self.current_step]["close"]
            
            # Check if we already have a position
            position = self.broker.get_position(self.symbol)
            have_position = position and position.get('quantity', 0) > 0
            
            # Check risk management constraints
            can_trade = (not self.trading_stopped_for_day and 
                         self.daily_trades < self.config.MAX_TRADES_PER_DAY)
                         
            # Daily open positions count from broker
            open_positions_count = len([p for p in self.broker.get_positions().values() 
                                      if p.get('quantity', 0) > 0])
            can_buy = can_trade and open_positions_count < self.config.MAX_OPEN_POSITIONS
            
            # Normalize action based on risk management
            if not can_trade and action >= 1 and action <= 4:
                logger.info(f"Trading stopped for day, forcing hold for {self.symbol}")
                action = 0  # Force hold if trading stopped
                
            if not can_buy and action >= 1 and action <= 4:
                logger.info(f"Max positions reached, forcing hold for {self.symbol}")
                action = 0  # Force hold if max positions reached
                
            # Process action
            if action == 0:  # Hold
                logger.debug(f"Hold for {self.symbol}")
                # Update trailing stop if in position
                if have_position:
                    self._update_trailing_stop(current_price)
                    # Check stop loss and take profit
                    exit_triggered, exit_type = self._check_exit_conditions(current_price)
                    if exit_triggered:
                        logger.info(f"Exit triggered for {self.symbol}: {exit_type}")
                        # Force sell action
                        return self._execute_sell(exit_type)
                return True
                
            elif action >= 1 and action <= 4 and can_buy and not have_position:  # Buy
                # Calculate position size based on action
                position_size_multiplier = action * 0.25  # 25%, 50%, 75%, or 100% of allowed size
                return self._execute_buy(position_size_multiplier)
                
            elif action == 5 and have_position:  # Sell
                return self._execute_sell("model_decision")
                
            else:
                logger.debug(f"No action taken for {self.symbol} (action={action}, can_trade={can_trade}, can_buy={can_buy}, have_position={have_position})")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing action for {self.symbol}: {str(e)}")
            return False
    
    def _execute_buy(self, position_size_multiplier):
        """
        Execute a buy order
        
        Args:
            position_size_multiplier (float): Multiplier for position size (0.25, 0.50, 0.75, 1.0)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current price
            current_price = self.df.iloc[self.current_step]["close"]
            
            # Get available funds
            funds = self.broker.get_funds()
            available_balance = funds.get('balance', 0) if funds else 0
            
            if available_balance <= 0:
                logger.warning(f"Insufficient funds for {self.symbol}")
                return False
                
            # Calculate position size
            max_position_size = available_balance * self.position_size_pct
            position_size = max_position_size * position_size_multiplier
            
            # Ensure we don't use more than 95% of available balance
            position_size = min(position_size, available_balance * 0.95)
            
            # Calculate shares to buy
            shares_to_buy = int(position_size // current_price)
            
            if shares_to_buy <= 0:
                logger.warning(f"Calculated shares to buy <= 0 for {self.symbol}")
                return False
                
            # Place the buy order
            order_id = self.broker.place_order(
                symbol=self.symbol,
                transaction_type='BUY',
                quantity=shares_to_buy,
                order_type='MARKET',
                tag="RL_BOT_BUY"
            )
            
            if not order_id:
                logger.error(f"Failed to place buy order for {self.symbol}")
                return False
                
            logger.info(f"Buy order placed for {self.symbol}: {shares_to_buy} shares at ~{current_price}")
            
            # Update strategy state
            self.position_type = 'long'
            self.entry_price = current_price
            self.entry_time = datetime.now()
            self.days_in_trade = 0
            self.total_trades += 1
            self.daily_trades += 1
            
            # Set up stop loss and target
            self._setup_risk_management()
            
            # Place stop loss order
            if self.stop_loss > 0:
                self.stop_loss_order_id = self.broker.place_stoploss_order(
                    symbol=self.symbol,
                    transaction_type='SELL',
                    quantity=shares_to_buy,
                    trigger_price=self.stop_loss
                )
                
                if not self.stop_loss_order_id:
                    logger.warning(f"Failed to place stop loss order for {self.symbol}")
                    
            # Place target order
            if self.target_price > 0:
                self.target_order_id = self.broker.place_target_order(
                    symbol=self.symbol,
                    transaction_type='SELL',
                    quantity=shares_to_buy,
                    price=self.target_price
                )
                
                if not self.target_order_id:
                    logger.warning(f"Failed to place target order for {self.symbol}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error executing buy for {self.symbol}: {str(e)}")
            return False
    
    def _execute_sell(self, exit_type="manual"):
        """
        Execute a sell order
        
        Args:
            exit_type (str): Type of exit (stop_loss, target, model_decision, etc.)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get current position
            position = self.broker.get_position(self.symbol)
            
            if not position or position.get('quantity', 0) <= 0:
                logger.warning(f"No position to sell for {self.symbol}")
                return False
                
            shares_to_sell = position.get('quantity', 0)
            
            # Place the sell order
            order_id = self.broker.place_order(
                symbol=self.symbol,
                transaction_type='SELL',
                quantity=shares_to_sell,
                order_type='MARKET',
                tag=f"RL_BOT_SELL_{exit_type.upper()}"
            )
            
            if not order_id:
                logger.error(f"Failed to place sell order for {self.symbol}")
                return False
                
            # Get current price
            current_price = self.df.iloc[self.current_step]["close"]
            
            logger.info(f"Sell order placed for {self.symbol}: {shares_to_sell} shares at ~{current_price} ({exit_type})")
            
            # Calculate profit/loss
            entry_price = self.entry_price or position.get('average_price', 0)
            profit_loss = (current_price - entry_price) * shares_to_sell
            profit_loss_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0
            
            # Update performance tracking
            if profit_loss > 0:
                self.winning_trades += 1
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                self.daily_profit += profit_loss
            else:
                self.losing_trades += 1
                self.consecutive_losses += 1
                self.consecutive_wins = 0
                self.daily_loss += abs(profit_loss)
                
            # Cancel any pending stop loss or target orders
            if self.stop_loss_order_id:
                self.broker.cancel_order(self.stop_loss_order_id)
                self.stop_loss_order_id = None
                
            if self.target_order_id:
                self.broker.cancel_order(self.target_order_id)
                self.target_order_id = None
                
            # Reset position state
            self.position_type = None
            self.entry_price = 0
            self.entry_time = None
            self.stop_loss = 0
            self.trailing_stop = 0
            self.target_price = 0
            self.highest_price_since_buy = 0
            self.days_in_trade = 0
            
            # Log trade
            logger.info(f"Trade completed for {self.symbol}: {exit_type}, P/L: {profit_loss:.2f} ({profit_loss_pct:.2%})")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error executing sell for {self.symbol}: {str(e)}")
            return False
    
    def _setup_risk_management(self):
        """Set up stop loss, trailing stop, and target price based on current regime"""
        try:
            # Get current price
            current_price = self.df.iloc[self.current_step]["close"]
            
            # Calculate ATR for dynamic stop loss and targets
            if 'ATR' in self.df.columns:
                current_atr = self.df.iloc[self.current_step]['ATR']
            else:
                current_atr = current_price * 0.02  # Default to 2% of price
                
            # Use regime-specific parameters
            if self.regime_params:
                atr_multiplier = self.regime_params.get('trailing_stop_atr_multiplier', 2.0)
                target_multiplier = self.regime_params.get('target_atr_multiplier', 3.0)
            else:
                atr_multiplier = 2.0
                target_multiplier = 3.0
                
            # Adjust for consecutive losses (tighten stops)
            if self.consecutive_losses >= 3:
                atr_multiplier *= 0.8  # Reduce by 20% after 3 consecutive losses
                
            # Calculate stop loss
            self.stop_loss = current_price - (current_atr * atr_multiplier)
            self.trailing_stop = self.stop_loss
            
            # Calculate target price
            min_rr_ratio = self.regime_params.get('min_rr_ratio', 1.5) if self.regime_params else 1.5
            stop_distance = current_price - self.stop_loss
            target_distance = stop_distance * min_rr_ratio
            target_distance = max(target_distance, current_atr * target_multiplier)
            self.target_price = current_price + target_distance
            
            # Initialize highest price
            self.highest_price_since_buy = current_price
            
            logger.info(f"Risk management set up for {self.symbol}: Stop={self.stop_loss:.2f}, Target={self.target_price:.2f}")
        
        except Exception as e:
            logger.exception(f"Error setting up risk management for {self.symbol}: {str(e)}")
    
    def _update_trailing_stop(self, current_price):
        """Update trailing stop if price moves in favorable direction"""
        try:
            if not self.position_type:
                return
                
            # Use regime-specific trailing settings
            use_trailing = self.regime_params.get('use_trailing_stop', True) if self.regime_params else True
            
            if not use_trailing:
                return
                
            # For long positions
            if self.position_type == 'long':
                if current_price > self.highest_price_since_buy:
                    self.highest_price_since_buy = current_price
                    
                    # Calculate profit percentage
                    profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
                    
                    # Tighten trail as profit increases
                    trail_multiplier = self.regime_params.get('trailing_stop_atr_multiplier', 2.0) if self.regime_params else 2.0
                    
                    if profit_pct > 0.10:  # Over 10% profit
                        trail_multiplier *= 0.7  # Tighter trail to protect profits
                    elif profit_pct > 0.05:  # Over 5% profit
                        trail_multiplier *= 0.8
                        
                    # Get current ATR
                    if 'ATR' in self.df.columns:
                        current_atr = self.df.iloc[self.current_step]['ATR']
                    else:
                        current_atr = current_price * 0.02  # Default to 2% of price
                    
                    # Calculate new stop
                    new_stop = current_price - (current_atr * trail_multiplier)
                    
                    # Only update if new stop is higher
                    if new_stop > self.trailing_stop:
                        self.trailing_stop = new_stop
                        
                        # Update stop loss order if exists
                        if self.stop_loss_order_id:
                            self.broker.modify_order(
                                order_id=self.stop_loss_order_id,
                                trigger_price=self.trailing_stop
                            )
                        
                        logger.debug(f"Updated trailing stop for {self.symbol} to {self.trailing_stop:.2f}")
                        
                        # Move to breakeven after sufficient profit
                        if profit_pct > 0.03 and self.trailing_stop < self.entry_price:
                            self.trailing_stop = max(self.trailing_stop, self.entry_price * 1.001)  # 0.1% above entry
                            
                            # Update stop loss order
                            if self.stop_loss_order_id:
                                self.broker.modify_order(
                                    order_id=self.stop_loss_order_id,
                                    trigger_price=self.trailing_stop
                                )
                            
                            logger.info(f"Moved trailing stop to breakeven for {self.symbol}: {self.trailing_stop:.2f}")
        
        except Exception as e:
            logger.exception(f"Error updating trailing stop for {self.symbol}: {str(e)}")
    
    def _check_exit_conditions(self, current_price):
        """
        Check if exit conditions (stop loss, target) are met
        
        Args:
            current_price (float): Current price
            
        Returns:
            tuple: (exit_triggered, exit_type)
        """
        try:
            if not self.position_type:
                return False, None
                
            hit_stop = False
            hit_target = False
            
            # Check if near market close
            is_near_close = self._is_near_market_close()
            
            # Force exit if near market close
            if is_near_close and self.position_type:
                logger.info(f"Exiting position for {self.symbol} due to market close")
                return True, "market_close"
                
            # Normal stop and target checks
            if self.position_type == 'long':
                if current_price <= self.trailing_stop:
                    hit_stop = True
                    exit_type = 'stop_loss'
                elif current_price >= self.target_price:
                    hit_target = True
                    exit_type = 'target'
                    
                # Calculate current profit percentage
                profit_pct = (current_price - self.entry_price) / self.entry_price if self.entry_price > 0 else 0
                
                # Dynamic exits based on market conditions
                if self.current_market_regime == "volatile" and profit_pct > 0.04:
                    # Take higher profits in volatile markets
                    hit_target = True
                    exit_type = 'regime_volatile_take_profit'
                    
                elif self.current_market_regime == "ranging" and profit_pct > 0.05:
                    # Take higher profits in ranging markets
                    hit_target = True
                    exit_type = 'regime_ranging_take_profit'
                    
                # Time-based exit for swing trading
                if self.days_in_trade > 8 and profit_pct < 0.02:
                    # Exit trades not performing after 8 days
                    hit_stop = True
                    exit_type = 'time_based_exit'
                    
                # RSI-based exit
                if 'RSI' in self.df.columns and self.df.iloc[self.current_step]['RSI'] > 85 and profit_pct > 0.03:
                    hit_target = True
                    exit_type = 'rsi_extreme'
            
            return hit_stop or hit_target, exit_type if (hit_stop or hit_target) else None
            
        except Exception as e:
            logger.exception(f"Error checking exit conditions for {self.symbol}: {str(e)}")
            return False, None
    
    def _is_market_open(self):
        """Check if market is currently open"""
        try:
            now = datetime.now(self.config.TIMEZONE)
            
            # If weekend, market is closed
            if now.weekday() >= 5:  # 5 is Saturday, 6 is Sunday
                return False
                
            # Check market hours
            market_open = now.replace(
                hour=self.config.MARKET_OPEN_HOUR, 
                minute=self.config.MARKET_OPEN_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            market_close = now.replace(
                hour=self.config.MARKET_CLOSE_HOUR, 
                minute=self.config.MARKET_CLOSE_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            return market_open <= now <= market_close
            
        except Exception as e:
            logger.exception(f"Error checking if market is open: {str(e)}")
            return False
    
    def _is_near_market_close(self):
        """Check if it's near market close time"""
        try:
            now = datetime.now(self.config.TIMEZONE)
            
            # Calculate market close time
            market_close = now.replace(
                hour=self.config.MARKET_CLOSE_HOUR, 
                minute=self.config.MARKET_CLOSE_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            # Check if within last N minutes of market
            time_to_close = (market_close - now).total_seconds() / 60
            return 0 <= time_to_close <= self.config.AVOID_LAST_MINUTES
            
        except Exception as e:
            logger.exception(f"Error checking if near market close: {str(e)}")
            return False
    
    def update_daily_status(self):
        """Update daily status - reset counters at start of day"""
        try:
            now = datetime.now(self.config.TIMEZONE)
            
            # Reset daily counters at market open
            market_open = now.replace(
                hour=self.config.MARKET_OPEN_HOUR, 
                minute=self.config.MARKET_OPEN_MINUTE, 
                second=0, 
                microsecond=0
            )
            
            # If we're at market open
            if abs((now - market_open).total_seconds()) < 60:  # Within 1 minute of market open
                logger.info(f"Resetting daily counters for {self.symbol}")
                
                # Reset daily counters
                self.daily_trades = 0
                self.daily_profit = 0
                self.daily_loss = 0
                self.trading_stopped_for_day = False
                self.trading_stopped_reason = None
                
                # Increment days in trade if position open
                if self.position_type:
                    self.days_in_trade += 1
            
            # Update max drawdown
            if self.position_type:
                # Get position value
                position_info = self.broker.get_position(self.symbol)
                if position_info:
                    current_value = position_info.get('quantity', 0) * self.df.iloc[self.current_step]["close"]
                    max_value = position_info.get('quantity', 0) * self.highest_price_since_buy
                    
                    if max_value > 0:
                        current_drawdown = (max_value - current_value) / max_value
                        self.max_drawdown = max(self.max_drawdown, current_drawdown)
            
            # Check daily loss limit
            funds = self.broker.get_funds()
            if funds:
                # Update max capital
                current_capital = funds.get('balance', 0)
                self.max_capital = max(self.max_capital, current_capital)
                
                # Calculate daily PnL
                if self.max_capital > 0:
                    daily_pnl_pct = (current_capital - self.max_capital) / self.max_capital
                    
                    # Stop trading if daily loss limit hit
                    if daily_pnl_pct < -self.config.DAILY_LOSS_LIMIT_PCT and not self.trading_stopped_for_day:
                        self.trading_stopped_for_day = True
                        self.trading_stopped_reason = "daily_loss_limit"
                        logger.warning(f"Trading stopped for day due to daily loss limit: {daily_pnl_pct:.2%}")
                        
                    # Stop trading if daily profit target hit
                    elif daily_pnl_pct > self.config.DAILY_PROFIT_TARGET_PCT and not self.trading_stopped_for_day:
                        # Don't stop trading on daily profit target in strong uptrends
                        if self.current_market_regime != "trending_up":
                            self.trading_stopped_for_day = True
                            self.trading_stopped_reason = "daily_profit_target"
                            logger.info(f"Trading stopped for day due to reaching profit target: {daily_pnl_pct:.2%}")
            
            return True
            
        except Exception as e:
            logger.exception(f"Error updating daily status for {self.symbol}: {str(e)}")
            return False
    
    def get_performance_metrics(self):
        """Get strategy performance metrics"""
        try:
            # Get position info
            position_info = self.broker.get_position(self.symbol)
            position_value = 0
            if position_info:
                position_value = position_info.get('quantity', 0) * self.df.iloc[self.current_step]["close"]
                
            # Get account info
            funds = self.broker.get_funds()
            balance = funds.get('balance', 0) if funds else 0
            
            # Calculate metrics
            win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
            
            return {
                'symbol': self.symbol,
                'balance': balance,
                'position_value': position_value,
                'total_value': balance + position_value,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'win_rate': win_rate,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses,
                'daily_trades': self.daily_trades,
                'daily_profit': self.daily_profit,
                'daily_loss': self.daily_loss,
                'max_drawdown': self.max_drawdown,
                'current_market_regime': self.current_market_regime,
                'position_type': self.position_type,
                'entry_price': self.entry_price,
                'days_in_trade': self.days_in_trade,
                'buy_signal_strength': self.buy_signal_strength,
                'sell_signal_strength': self.sell_signal_strength,
                'trading_stopped_for_day': self.trading_stopped_for_day,
                'trading_stopped_reason': self.trading_stopped_reason
            }
            
        except Exception as e:
            logger.exception(f"Error getting performance metrics for {self.symbol}: {str(e)}")
            return {}