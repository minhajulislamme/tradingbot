import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters - RAYSOL only
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'SUIUSDT')
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # ISOLATED or CROSSED
STRATEGY = os.getenv('STRATEGY', 'RaysolDynamicStrategy')

# RAYSOL-specific strategy settings
RAYSOL_TREND_EMA_FAST = int(os.getenv('RAYSOL_TREND_EMA_FAST', '8'))
RAYSOL_TREND_EMA_SLOW = int(os.getenv('RAYSOL_TREND_EMA_SLOW', '21'))
RAYSOL_VOLATILITY_LOOKBACK = int(os.getenv('RAYSOL_VOLATILITY_LOOKBACK', '20'))
RAYSOL_VOLUME_MA_PERIOD = int(os.getenv('RAYSOL_VOLUME_MA_PERIOD', '20'))
# RAYSOL-specific advanced parameters
RAYSOL_VOLATILITY_MULTIPLIER = float(os.getenv('RAYSOL_VOLATILITY_MULTIPLIER', '1.1'))
RAYSOL_TREND_CONDITION_MULTIPLIER = float(os.getenv('RAYSOL_TREND_CONDITION_MULTIPLIER', '1.3'))

# No other cryptocurrency settings - RAYSOL only

# RAYSOL market condition detection settings
RAYSOL_ADX_PERIOD = int(os.getenv('RAYSOL_ADX_PERIOD', '14'))
RAYSOL_ADX_THRESHOLD = int(os.getenv('RAYSOL_ADX_THRESHOLD', '25'))
RAYSOL_SIDEWAYS_THRESHOLD = int(os.getenv('RAYSOL_SIDEWAYS_THRESHOLD', '15'))

# Position sizing
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))
FIXED_TRADE_PERCENTAGE = float(os.getenv('FIXED_TRADE_PERCENTAGE', '0.75'))  # Use 75% of balance per trade
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '6'))

# Margin safety settings
MARGIN_SAFETY_FACTOR = float(os.getenv('MARGIN_SAFETY_FACTOR', '0.95'))  # Use at most 95% of available margin
MAX_POSITION_SIZE_PCT = float(os.getenv('MAX_POSITION_SIZE_PCT', '0.50'))  # Max position size as % of balance
MIN_FREE_BALANCE_PCT = float(os.getenv('MIN_FREE_BALANCE_PCT', '0.05'))  # Keep at least 5% of balance free

# Multi-instance configuration for running separate bot instances per trading pair
MULTI_INSTANCE_MODE = os.getenv('MULTI_INSTANCE_MODE', 'True').lower() == 'true'
MAX_POSITIONS_PER_SYMBOL = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'))

# Auto-compounding settings
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')

# Technical indicator parameters
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))
FAST_EMA = int(os.getenv('FAST_EMA', '8'))
SLOW_EMA = int(os.getenv('SLOW_EMA', '21'))
TIMEFRAME = os.getenv('TIMEFRAME', '15m')

# Risk management - Standard settings
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.03'))
TRAILING_STOP = os.getenv('TRAILING_STOP', 'True').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.03'))

# Remove complex adaptive risk management settings since we're simplifying
# Keep these variables for compatibility but use simple values
STOP_LOSS_PCT_BULLISH = STOP_LOSS_PCT
STOP_LOSS_PCT_BEARISH = STOP_LOSS_PCT
STOP_LOSS_PCT_SIDEWAYS = STOP_LOSS_PCT

# For compatibility with existing code
USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'True').lower() == 'true'
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.06'))
TRAILING_TAKE_PROFIT = os.getenv('TRAILING_TAKE_PROFIT', 'True').lower() == 'true'
TRAILING_TAKE_PROFIT_PCT = float(os.getenv('TRAILING_TAKE_PROFIT_PCT', '0.03'))

# Simple mapping for compatibility
TAKE_PROFIT_PCT_BULLISH = TAKE_PROFIT_PCT
TAKE_PROFIT_PCT_BEARISH = TAKE_PROFIT_PCT
TAKE_PROFIT_PCT_SIDEWAYS = TAKE_PROFIT_PCT

TRAILING_STOP_PCT_BULLISH = TRAILING_STOP_PCT
TRAILING_STOP_PCT_BEARISH = TRAILING_STOP_PCT
TRAILING_STOP_PCT_SIDEWAYS = TRAILING_STOP_PCT

TRAILING_TAKE_PROFIT_PCT_BULLISH = TRAILING_TAKE_PROFIT_PCT
TRAILING_TAKE_PROFIT_PCT_BEARISH = TRAILING_TAKE_PROFIT_PCT
TRAILING_TAKE_PROFIT_PCT_SIDEWAYS = TRAILING_TAKE_PROFIT_PCT

# Backtesting parameters
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '')  # Empty means use current date
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '50.0'))
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))  # 0.04% taker fee
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'

# Pre-live backtest validation
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '5.0'))
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '40.0'))
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '15 days')

# Logging and notifications
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'True').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'True').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '00:00')  # 24-hour format

# Other settings
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))
RETRY_DELAY = int(os.getenv('RETRY_DELAY', '5'))  # seconds