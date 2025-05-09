#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="rl_trading_bot",
    version="1.0.0",
    description="Reinforcement Learning Trading Bot for Zerodha",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=1.7.0",
        "torch>=1.12.0",
        "kiteconnect>=4.1.0",
        "pyotp>=2.6.0",
        "schedule>=1.1.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "requests>=2.26.0",
        "python-dotenv>=0.19.0",
        "pytz>=2021.3",
        "tqdm>=4.62.0",
        "rich>=12.0.0",
        "tabulate>=0.8.0",
    ],
    entry_points={
        "console_scripts": [
            "rl_trading_bot=rl_trading_bot.main:main",
        ],
    },
    python_requires=">=3.8",
)