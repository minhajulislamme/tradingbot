# Binance Trading Bot

A sophisticated automated trading bot for Binance Futures, featuring advanced technical analysis strategies, real-time data processing, and risk management capabilities.

## Features

- **Advanced Trading Strategies**: Dynamic Strategy with volatility-based position sizing and market condition detection
- **Real-Time Data Processing**: Utilizes WebSockets for up-to-the-second market data
- **Comprehensive Risk Management**: Dynamic position sizing, stop-loss, and take-profit management
- **Performance Tracking**: Detailed logs, trade history, and performance reporting
- **Backtesting Capabilities**: Test strategies on historical data before trading live
- **Telegram Integration**: Real-time notifications and status updates
- **Adaptive Market Analysis**: Detects and adapts to different market conditions
- **Institutional Order Flow Analysis**: Identifies and acts on large market orders
- **Auto-Compounding**: Automatically adjusts position sizes based on performance

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tradingbot.git
   cd tradingbot
   ```

2. Run the setup script to create the virtual environment and install dependencies:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Configure your Binance API keys in the `.env` file:
   ```
   BINANCE_API_KEY=your_api_key
   BINANCE_API_SECRET=your_api_secret
   ```

## Configuration

Edit the `.env` file to customize trading parameters:

- `TRADING_SYMBOL`: The cryptocurrency pair to trade (e.g., BTCUSDT, ETHUSDT)
- `STRATEGY`: Trading strategy to use (default: RaysolDynamicStrategy)
- `TIMEFRAME`: Trading timeframe (e.g., 15m, 1h)
- Strategy-specific parameters (risk levels, indicator settings, etc.)

## Usage

### Start the Trading Bot

```bash
./run_bot.sh
```

### Check Bot Status

```bash
./check_bot_status.sh
```

### Stop the Bot

```bash
./stop_bot_manual.sh
```

### Run Backtests

```bash
python main.py --backtest --symbol BTCUSDT --timeframe 15m --start-date "2023-01-01"
```

### Generate Performance Reports

```bash
python main.py --report
```

## System Requirements

- Python 3.8+
- Linux/macOS/Windows
- Internet connection
- Binance account with API access

## Dependencies

- python-binance
- numpy
- pandas
- ta (Technical Analysis library)
- matplotlib
- websocket-client
- schedule
- scikit-learn
- scipy
- requests
- tqdm

## Directory Structure

- `main.py`: Main bot executable
- `modules/`: Core components (strategies, risk manager, Binance client)
- `state/`: Persists bot state between restarts
- `logs/`: Trading logs
- `reports/`: Performance reports
- `backtest_results/`: Backtest output files

## Disclaimer

This trading bot is for educational and research purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. The developer is not responsible for any financial losses incurred while using this software.

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
