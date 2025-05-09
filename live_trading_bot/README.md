# Zerodha RL Trading Bot

A production-grade reinforcement learning-based trading system for Indian stock market using Zerodha.

## Features

- **Reinforcement Learning**: Uses PPO (Proximal Policy Optimization) for decision making
- **Market Regime Detection**: Adapts strategy based on current market conditions
- **Risk Management**: Advanced position sizing, stop loss, and profit target management
- **Real-time Data**: Connects to Zerodha for live market data and order execution
- **Performance Tracking**: Detailed metrics and trade history tracking
- **Web Dashboard**: Monitor and control the bot through a web interface
- **Notifications**: Email and Telegram alerts for important events

## Installation

### Prerequisites

- Python 3.8 or higher
- Zerodha trading account with API access
- Docker (optional, for containerized deployment)

### Using Docker

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rl-trading-bot.git
cd rl-trading-bot



# docker-compose up -d
# git clone https://github.com/yourusername/rl-trading-bot.git
# cd rl-trading-bot
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate
# pip install -r requirements.txt
# python main.py

# python main.py --mode live --log-level INFO --config config/app_config.py

# to train models
# python tools/train_model.py --symbol ASIANPAINT --days 365 --trials 500



# Set up monitoring:

# Configure email or Telegram notifications
# Add server monitoring (e.g., Prometheus, Grafana)
# Check logs regularly: docker-compose logs -f trading_bot