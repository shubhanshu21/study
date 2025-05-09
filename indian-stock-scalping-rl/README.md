# Indian Stock Market Scalping with Reinforcement Learning

A production-grade implementation of a Reinforcement Learning agent for scalping in the Indian stock market using Gymnasium and Stable Baselines 3.

## Features

- **RL-based Trading**: Utilizes Proximal Policy Optimization (PPO) algorithm from Stable Baselines 3
- **Indian Market Specifics**: Handles all taxes, brokerage fees, and market rules specific to Indian exchanges
- **Production-Ready**: Error handling, logging, and optimized design for real-world use
- **Multiple Brokers**: Support for major Indian brokers (Zerodha, Upstox)
- **Paper & Live Trading**: Test in paper mode before deploying real money
- **Risk Management**: Stop loss, max drawdown protection, and performance tracking
- **Complete Pipeline**: Training, backtesting, and live trading in one solution

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/indian-stock-scalping-rl.git
cd indian-stock-scalping-rl
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your broker credentials:
```
API_KEY=your_api_key
API_SECRET=your_api_secret
USER_ID=your_user_id
PASSWORD=your_password
TOTP_KEY=your_totp_key
INITIAL_BALANCE=100000
DEFAULT_SYMBOL=RELIANCE
```

## Usage

### Training a Model

```bash
python -m src.main --mode train --symbol RELIANCE --steps 100000
```

### Backtesting

```bash
python -m src.main --mode backtest --symbol RELIANCE --model models/scalping_model.zip
```

### Live Trading (Paper Mode)

```bash
python -m src.main --mode trade --symbol RELIANCE --model models/scalping_model.zip --paper
```

### Live Trading (Real Money)

```bash
python -m src.main --mode trade --symbol RELIANCE --model models/scalping_model.zip
```

## Project Structure

```
indian-stock-scalping-rl/
│
├── .env                              # Environment variables
├── requirements.txt                  # Project dependencies
├── README.md                         # Documentation
│
├── src/                              # Source code directory
│   ├── __init__.py
│   ├── main.py                       # Main entry point
│   ├── constants.py                  # Market constants and config
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py           # Data fetching from brokers
│   │   └── indicators.py             # Technical indicators
│   │
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── costs.py                  # Trading costs calculator
│   │   └── order_manager.py          # Order management
│   │
│   └── rl/
│       ├── __init__.py
│       ├── environment.py            # Gymnasium environment
│       ├── trainer.py                # Training utilities
│       └── trader.py                 # Live trading implementation
│
├── models/                           # Saved models directory
│   └── .gitkeep
│
├── data/                             # Data storage (optional)
│   └── .gitkeep
│
└── logs/                             # Logs directory
    └── .gitkeep
```

## Customization

You can modify the following parameters for your trading strategy:

- **Window Size**: Change the observation window in `constants.py` (TrainingConfig.WINDOW_SIZE)
- **Risk Parameters**: Adjust stop loss percentage in `constants.py` (TradingConfig.DEFAULT_STOP_LOSS_PCT)
- **Reward Function**: Customize the reward function in `environment.py` to match your trading objectives
- **RL Parameters**: Tune learning rate, batch size, etc. in `constants.py` (TrainingConfig)

## Trading Costs

This implementation includes accurate calculation of all Indian trading costs:

- Securities Transaction Tax (STT)
- Exchange Transaction Charges
- GST
- SEBI Charges
- Stamp Duty
- Brokerage

## Warning

Trading with real money involves significant risk. Use this software at your own risk. The authors are not responsible for any financial losses incurred.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.