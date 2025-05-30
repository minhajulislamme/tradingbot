# 🚀 Binance Trading Bot

<div align="center">
  
![Version](https://img.shields.io/badge/version-0.1.0-blue.svg?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white)
![Binance](https://img.shields.io/badge/Binance-Futures-yellow.svg?style=flat-square&logo=binance&logoColor=white)

</div>

A sophisticated automated trading bot for Binance Futures, featuring advanced technical analysis strategies, real-time data processing, and risk management capabilities. Designed to provide a complete solution for algorithmic cryptocurrency trading.

## ✨ Features

- **🧠 Advanced Trading Strategies**: Dynamic Strategy with volatility-based position sizing and market condition detection
- **⚡ Real-Time Data Processing**: Utilizes WebSockets for up-to-the-second market data
- **🛡️ Comprehensive Risk Management**: Dynamic position sizing, stop-loss, and take-profit management
- **📊 Performance Tracking**: Detailed logs, trade history, and performance reporting
- **🔍 Backtesting Capabilities**: Test strategies on historical data before trading live
- **📱 Telegram Integration**: Real-time notifications and status updates
- **📈 Adaptive Market Analysis**: Detects and adapts to different market conditions
- **🔎 Institutional Order Flow Analysis**: Identifies and acts on large market orders
- **💰 Auto-Compounding**: Automatically adjusts position sizes based on performance

## 🛠️ Installation

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

## ⚙️ Configuration

Edit the `.env` file to customize trading parameters:

| Parameter        | Description                      | Example Values            |
| ---------------- | -------------------------------- | ------------------------- |
| `TRADING_SYMBOL` | The cryptocurrency pair to trade | BTCUSDT, ETHUSDT, SUIUSDT |
| `STRATEGY`       | Trading strategy to use          | RaysolDynamicStrategy     |
| `TIMEFRAME`      | Trading timeframe                | 15m, 1h, 4h               |
| `RSI_PERIOD`     | RSI indicator period             | 14                        |
| `LEVERAGE`       | Trading leverage                 | 1, 5, 10                  |
| `USE_TELEGRAM`   | Enable Telegram notifications    | True, False               |

## 🚀 Usage

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

## 🖥️ Command Line Arguments

| Argument                | Description                                  |
| ----------------------- | -------------------------------------------- |
| `--backtest`            | Run in backtest mode only                    |
| `--symbol SYMBOL`       | Trading symbol (default: from config)        |
| `--timeframe TIMEFRAME` | Timeframe for trading (default: from config) |
| `--strategy STRATEGY`   | Strategy to use (default: from config)       |
| `--start-date DATE`     | Start date for backtest (YYYY-MM-DD)         |
| `--end-date DATE`       | End date for backtest (YYYY-MM-DD)           |
| `--report`              | Generate performance report only             |
| `--small-account`       | Run with small account mode                  |
| `--skip-validation`     | Skip strategy validation                     |

## 💻 System Requirements

- Python 3.8+
- Linux/macOS/Windows
- Internet connection
- Binance account with API access

## 📦 Dependencies

<div align="center">
  
![python-binance](https://img.shields.io/badge/python--binance-1.0.28-blue.svg?style=flat-square)
![pandas](https://img.shields.io/badge/pandas-1.3.0+-green.svg?style=flat-square)
![numpy](https://img.shields.io/badge/numpy-1.20.0+-orange.svg?style=flat-square)
![ta](https://img.shields.io/badge/ta-0.10.0+-yellow.svg?style=flat-square)
  
</div>

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

## 📁 Directory Structure

```
tradingbot/
├── main.py                 # Main bot executable
├── setup.sh                # Setup script
├── run_bot.sh              # Bot startup script
├── stop_bot_manual.sh      # Bot shutdown script
├── check_bot_status.sh     # Status checking script
├── requirements.txt        # Python dependencies
├── modules/                # Core components
│   ├── binance_client.py   # Binance API interface
│   ├── strategies.py       # Trading strategies
│   ├── risk_manager.py     # Risk management
│   ├── websocket_handler.py # Real-time data handling
│   ├── config.py           # Configuration
│   └── backtest.py         # Backtesting engine
├── state/                  # Persists bot state
├── logs/                   # Trading logs
├── reports/                # Performance reports
└── backtest_results/       # Backtest output files
```

## ⚠️ Disclaimer

This trading bot is for educational and research purposes only. Use at your own risk. Cryptocurrency trading involves substantial risk of loss and is not suitable for every investor. The developer is not responsible for any financial losses incurred while using this software.

## 📊 Performance Visualization

The bot generates detailed performance reports and visualizations after backtesting:

```
backtest_results/
└── SUIUSDT_RaysolDynamicStrategy_20250530_230844/
    ├── equity.csv
    ├── results.json
    ├── summary.md
    ├── trades.csv
    └── plots/
        ├── drawdown.png
        ├── equity_curve.png
        └── monthly_returns.png
```

## 📜 License

MIT License

## 👨‍💻 Author

Developed by **Minhajul Islam**

<div align="center">
  
[![GitHub](https://img.shields.io/badge/GitHub-minhajulislamme-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/minhajulislamme)

  
</div>

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
