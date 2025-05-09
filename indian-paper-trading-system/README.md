# Indian Stock Market Paper Trading System

A comprehensive paper trading system for the Indian stock market that runs during market hours. This system uses real-time data to simulate trading with advanced technical analysis and risk management.

## Features

- **Live Market Data Integration**: Connect to Zerodha (primary) or NSE (backup) for real-time market data
- **Automated Trading**: Execute trades based on configurable strategies
- **Multiple Trading Strategies**:
  - Multi-Technical: Combines multiple technical indicators for signal generation
  - Swing Trading: Strategy optimized for longer holding periods
  - RL-Based: Strategy using pre-trained Reinforcement Learning models
- **Advanced Risk Management**: Dynamic position sizing, trailing stops, daily limits
- **Market Hours Scheduling**: Automatically runs during Indian market hours (9:15 AM - 3:30 PM IST)
- **Interactive Dashboard**: Real-time monitoring of trades, positions, and performance
- **Simulation Mode**: Test strategies using historical data

## Requirements

- Python 3.8+
- Zerodha API access (optional, for live data)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/indian-paper-trading.git
   cd indian-paper-trading
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables (create a `.env` file):
   ```
   ZERODHA_API_KEY=your_api_key
   ZERODHA_API_SECRET=your_api_secret
   ZERODHA_ACCESS_TOKEN=your_access_token
   DEFAULT_INITIAL_BALANCE=100000
   ```

## Folder Structure

```
indian-paper-trading-system/
├── config/               # Configuration settings
├── data/                 # Data storage
│   ├── fetchers/         # Data fetching modules
│   ├── processors/       # Data processing modules
│   ├── historical/       # Historical data storage
│   ├── states/           # System state snapshots
│   └── metrics/          # Performance metrics
├── engine/               # Trading engine components
├── models/               # ML models storage
├── scheduler/            # Market hour scheduling
├── ui/                   # Dashboard interface
├── utils/                # Utility functions
├── .env                  # Environment variables
├── app.py                # Main application entry point
└── requirements.txt      # Dependencies
```

## Usage

### Running Live Trading

```bash
python app.py --symbols HDFCBANK,RELIANCE,INFY --balance 100000 --strategy multi_technical --dashboard
```

Options:
- `--symbols`: Comma-separated list of stock symbols (default: predefined list)
- `--balance`: Initial paper trading balance (default: 100000)
- `--strategy`: Trading strategy (`multi_technical`, `swing`, or `rl`) (default: `multi_technical`)
- `--dashboard`: Enable web dashboard (default: enabled if configured)
- `--debug`: Enable debug logging

### Running in Simulation Mode

```bash
python app.py --symbols HDFCBANK,RELIANCE,INFY --balance 100000 --strategy swing --simulate --days 5
```

Additional options:
- `--simulate`: Run in simulation mode
- `--days`: Number of days to simulate (default: 1)

### Accessing the Dashboard

If the dashboard is enabled, you can access it at:
```
http://localhost:8050
```

## Dashboard Features

- Real-time portfolio overview
- Position monitoring
- Order history
- Performance metrics
- Interactive price charts with trades and indicators

## Configuration

Advanced configuration options can be set in:
- `config/market_config.py`: Market timing settings
- `config/trading_config.py`: Trading parameters and API credentials

## Risk Management

The system includes sophisticated risk management features:
- Dynamic position sizing based on market regime
- Daily loss limits and profit targets
- Maximum drawdown protection
- Trailing stop management

## Development

### Adding New Strategies

Create a new strategy class in `engine/strategy.py` that inherits from `TradingStrategy` and implements the `generate_signals` method.

### Adding New Data Sources

Create a new data fetcher class in `data/fetchers/` following the pattern of existing fetchers.

## License

MIT License

## Disclaimer

This system is for paper trading and educational purposes only. Always practice proper risk management when applying these strategies to real trading.