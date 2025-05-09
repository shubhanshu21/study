# Database Directory

This directory contains database files for the RL Trading Bot.

## Files

- `trading.db`: Main SQLite database file (created at runtime)

## Tables

The database contains the following tables:

1. `model_training_progress`: Tracks model training progress
   - `symbol`: Trading symbol (PRIMARY KEY)
   - `last_trained_steps`: Number of training steps completed
   - `last_trained_date`: Date of last training

2. `trades`: Records all executed trades
   - `id`: Unique ID (PRIMARY KEY AUTOINCREMENT)
   - `symbol`: Trading symbol
   - `order_id`: Order ID
   - `transaction_type`: Transaction type (BUY/SELL)
   - `price`: Execution price
   - `quantity`: Number of shares
   - `timestamp`: Execution timestamp
   - `entry_price`: Entry price (for SELL trades)
   - `exit_price`: Exit price (for SELL trades)
   - `profit_loss`: Profit/loss amount
   - `profit_loss_percent`: Profit/loss percentage
   - `exit_type`: Exit type (stop_loss, target, etc.)
   - `market_regime`: Market regime at time of trade
   - `stop_loss`: Stop loss price
   - `target_price`: Target price
   - `trailing_stop`: Trailing stop price
   - `days_in_trade`: Number of days in trade
   - `buy_signal_strength`: Buy signal strength
   - `sell_signal_strength`: Sell signal strength
   - `transaction_costs`: Transaction costs

3. `daily_performance`: Records daily performance metrics
   - `date`: Date (PRIMARY KEY)
   - `symbol`: Trading symbol (PRIMARY KEY)
   - `open_balance`: Balance at market open
   - `close_balance`: Balance at market close
   - `daily_profit_loss`: Daily profit/loss amount
   - `daily_profit_loss_percent`: Daily profit/loss percentage
   - `trades`: Number of trades executed
   - `winning_trades`: Number of winning trades
   - `losing_trades`: Number of losing trades
   - `market_regime`: Market regime for the day
   - `max_drawdown`: Maximum drawdown for the day

4. `price_data`: Records historical price data
   - `symbol`: Trading symbol (PRIMARY KEY)
   - `timestamp`: Timestamp (PRIMARY KEY)
   - `open`: Open price
   - `high`: High price
   - `low`: Low price
   - `close`: Close price
   - `volume`: Volume