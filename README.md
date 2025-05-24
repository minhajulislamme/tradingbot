# Binance Layer Trading Bot

A professional-grade algorithmic trading bot for Binance Futures, specifically optimized for LAYER token trading with adaptive grid strategies.

**Author: Minhajul Islam**

![Binance Layer Trading Bot](https://i.imgur.com/example-image.png)

## 🚀 Features

- **Dynamic Grid Trading**: Adaptive grid spacing based on market volatility
- **Market Condition Detection**: Automatically adapts to trending, ranging, and volatile markets
- **Multi-Asset Support**: Primary support for LAYER with extensibility for other assets (XRP, SOL, etc.)
- **Real-time WebSocket Integration**: Low-latency price and order book monitoring
- **Sophisticated Risk Management**: Volatility-based position sizing and dynamic stop-loss placement
- **Auto-compounding**: Automatically reinvests profits to grow account size
- **Advanced Technical Indicators**:
  - Supertrend for trend detection
  - TTM Squeeze for volatility breakouts
  - Multi-timeframe analysis
  - Volume profile analysis
- **Telegram Integration**: Real-time notifications and performance reporting
- **Comprehensive Backtesting**: Validate strategies against historical data

## 📋 Requirements

- Python 3.8+
- Binance Futures account with API access
- Linux environment (recommended for 24/7 operation)

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/binanclayer.git
cd binanclayer
```

2. **Set up virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Configure API credentials**

Create a `.env` file in the project root:

```
# Binance API credentials
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Trading settings
TRADING_SYMBOL=LAYERUSDT
STRATEGY=LayerDynamicGrid
TIMEFRAME=15m

# Risk settings
LEVERAGE=10
RISK_PER_TRADE=0.10

# Optional Telegram notifications
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

5. **Run the setup script**

```bash
chmod +x setup.sh
./setup.sh
```

## 🚦 Usage

### Running the Bot

**Start the bot with default settings (LAYER trading)**:

```bash
python main.py
```

The bot will:

1. Send a startup notification
2. Execute a test trade to verify functionality
3. Start live trading

**Custom Trading Parameters**:

```bash
python main.py --symbol LAYERUSDT --strategy LayerDynamicGrid --timeframe 15m
```

### Backtesting

Test your strategy against historical data before deploying:

```bash
python main.py --backtest --symbol LAYERUSDT --strategy LayerDynamicGrid --start-date "30 days ago"
```

### Additional Options

- `--test-trade`: Only execute a test trade to verify API connectivity
- `--report`: Generate a performance report from existing trading data
- `--small-account`: Optimize settings for accounts under $50
- `--skip-validation`: Skip backtest validation before live trading
- `--interval`: Trading check interval in minutes (default: 5)

## 📈 Strategy Details

### LayerDynamicGrid Strategy

This strategy implements a sophisticated grid trading approach specifically optimized for LAYER token:

1. **Adaptive Grid Spacing**: Adjusts grid levels based on market volatility
2. **Trend Detection**: Uses EMA crossovers and Supertrend indicators
3. **Entry/Exit Conditions**:
   - Long entries on bullish Supertrend + squeeze release in uptrend
   - Short entries on bearish Supertrend + squeeze release in downtrend
4. **Position Management**:
   - Partial take profits at key levels
   - Trailing stops for maximizing profit in strong trends
   - Dynamic stop loss based on ATR and support/resistance levels

### AvaxDynamicGrid Strategy

Specialized for AVAX's higher volatility characteristics:

1. **Wider Grid Spacing**: Adjusted for AVAX's larger price movements
2. **Enhanced Volatility Handling**:
   - More responsive to momentum shifts
   - Optimized for AVAX's faster market cycles
3. **Modified Market Detection**:
   - Earlier trend identification
   - Adjusted RSI thresholds for AVAX's volatility patterns
4. **Breakout Focus**:
   - More aggressive positioning for post-consolidation breakouts
   - Fibonacci-based trade targets calibrated for AVAX's price action

## ⚙️ Configuration

The bot can be fully customized through the `.env` file and command-line arguments. Key configuration parameters:

### Risk Management

- `RISK_PER_TRADE`: Percentage of account to risk per trade (default: 10%)
- `LEVERAGE`: Trading leverage (default: 10×)
- `STOP_LOSS_PCT`: Default stop loss percentage (separate settings for different market conditions)
- `TRAILING_STOP`: Enable/disable trailing stops (default: true)
- `AUTO_COMPOUND`: Enable/disable profit compounding (default: true)

### Strategy Parameters

- `LAYER_GRID_LEVELS`: Number of grid levels (default: 5)
- `LAYER_GRID_SPACING_PCT`: Base grid spacing percentage (default: 1.2%)
- `LAYER_VOLATILITY_MULTIPLIER`: Adjusts grid spacing based on volatility (default: 1.1)
- `LAYER_TREND_EMA_FAST`: Fast EMA period for trend detection (default: 8)
- `LAYER_TREND_EMA_SLOW`: Slow EMA period for trend detection (default: 21)

### Notification Settings

- `USE_TELEGRAM`: Enable/disable Telegram notifications
- `SEND_DAILY_REPORT`: Enable/disable daily performance reports
- `DAILY_REPORT_TIME`: Time to send daily reports (24-hour format)

## 🛡️ Safety Features

- Backtest validation before live trading
- Test trades to verify API connectivity
- Position verification after order placement
- Failsafes for partial fills and connectivity issues
- Automatic recovery from API errors

## 📊 Performance Monitoring

The bot includes comprehensive performance tracking:

- Real-time profit/loss monitoring
- Daily performance reports
- Equity curve visualization
- Win/loss statistics
- Detailed trade logs

## ⚠️ Risk Disclaimer

Trading cryptocurrencies involves significant risk. This bot is provided as-is with no guarantee of profitability. Always start with small amounts and use at your own risk.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support or questions, create an issue or contact us through Telegram.
# binancekaito
