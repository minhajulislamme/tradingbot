import logging
import time
from binance.client import Client
from binance.exceptions import BinanceAPIException
from modules.config import (
    API_KEY, API_SECRET, RETRY_COUNT, RETRY_DELAY, TRADING_TYPE, LEVERAGE, MARGIN_TYPE,
    API_URL, API_TESTNET, RECV_WINDOW
)

logger = logging.getLogger(__name__)

class BinanceClient:
    def __init__(self):
        if not API_KEY or not API_SECRET:
            raise ValueError("Binance API key and secret are required. Please set them in your .env file.")
            
        self.client = self._initialize_client()
        self.futures_initialized = False
        self.use_spot_fallback = False  # Flag to indicate if we should fall back to spot API
        
    def _initialize_client(self):
        for attempt in range(RETRY_COUNT):
            try:
                # Initialize with simple parameters for compatibility
                client = Client(API_KEY, API_SECRET, testnet=API_TESTNET)
                
                # Set proper timeout for API calls - increased timeout for historical data
                client.options = {'timeout': 60, 'recvWindow': RECV_WINDOW}
                
                # Test the connection by making a simple API call
                client.get_server_time()
                
                if API_TESTNET:
                    logger.info("Successfully connected to Binance Testnet API")
                else:
                    logger.info("Successfully connected to Binance Production API")
                
                # Test if we can access futures API
                try:
                    # Will throw exception if we can't access futures
                    client.futures_account()
                except BinanceAPIException as e:
                    if e.code == -2015:  # Permission denied error
                        logger.warning("No permission for Futures API. Will use Spot API as fallback.")
                        self.use_spot_fallback = True
                    elif "<!DOCTYPE html>" in str(e) or e.code == 0:
                        logger.warning("Received HTML response when accessing Futures API. Will use Spot API as fallback.")
                        self.use_spot_fallback = True
                    else:
                        logger.warning(f"Futures API error: {e}. Will attempt to continue.")
                
                # Synchronize local time with server time to avoid timestamp issues
                self._sync_time(client)
                
                return client
            except Exception as e:
                error_str = str(e)
                
                # Check if we received HTML instead of JSON
                if "<!DOCTYPE html>" in error_str:
                    logger.error(f"Binance API returned HTML instead of JSON (attempt {attempt+1}/{RETRY_COUNT})")
                else:
                    logger.error(f"Failed to connect to Binance API: {e} (attempt {attempt+1}/{RETRY_COUNT})")
                    
                if attempt < RETRY_COUNT - 1:
                    wait_time = RETRY_DELAY * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Throw exception after all retries fail
                    raise ConnectionError(f"Failed to connect to Binance API after {RETRY_COUNT} attempts")
    
    def _sync_time(self, client=None):
        """Synchronize local time with Binance server time"""
        if client is None:
            client = self.client
            
        try:
            server_time = client.get_server_time()
            local_time = int(time.time() * 1000)
            time_offset = server_time['serverTime'] - local_time
            
            # Store time offset for future use
            client.time_offset = time_offset
            
            logger.info(f"Time synchronized with Binance server. Offset: {time_offset}ms")
            return time_offset
        except Exception as e:
            logger.error(f"Failed to sync time with Binance: {e}")
            return 0
    
    def initialize_futures(self, symbol, leverage=LEVERAGE, margin_type=MARGIN_TYPE):
        """Set up futures trading settings"""
        if self.futures_initialized:
            return
            
        try:
            # Change margin type (ISOLATED or CROSSED)
            self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
            logger.info(f"Set margin type to {margin_type} for {symbol}")
        except BinanceAPIException as e:
            # Error code 4046 means margin type is already set
            if e.code == -4046:
                logger.info(f"Margin type for {symbol} already set to {margin_type}")
            # Error code 4168 means account is in Multi-Assets mode
            elif e.code == -4168:
                logger.info(f"Account is in Multi-Assets mode. Continuing with current margin setting.")
            else:
                logger.warning(f"Failed to set margin type: {e}")
                
        try:
            # Set leverage
            self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
            logger.info(f"Set leverage to {leverage}x for {symbol}")
            self.futures_initialized = True
        except BinanceAPIException as e:
            logger.error(f"Failed to set leverage: {e}")
            raise
    
    def get_account_balance(self):
        """Get current account balance in USDT"""
        max_retries = 3
        backoff_factor = 2
        
        # First try futures API if we're not in fallback mode
        if not self.use_spot_fallback:
            # Try primary and fallback URLs
            urls_to_try = ["futures_account_balance", "futures_account"]
            
            for method_name in urls_to_try:
                for retry in range(max_retries):
                    try:
                        if method_name == "futures_account_balance":
                            account = self.client.futures_account_balance()
                            for balance in account:
                                if balance['asset'] == 'USDT':
                                    return float(balance['balance'])
                        else:
                            # Try fallback method
                            account = self.client.futures_account()
                            if 'assets' in account:
                                for asset in account['assets']:
                                    if asset['asset'] == 'USDT':
                                        return float(asset['walletBalance'])
                        
                        # If we reach here with no return, try next method or return 0
                        break
                    except Exception as e:
                        error_str = str(e)
                        # Check for common error types that should be retried
                        retry_errors = [
                            "Invalid JSON",
                            "Connection reset",
                            "Read timed out",
                            "Connection aborted",
                            "Connection refused",
                            "code=0",
                            "<!DOCTYPE html>"
                        ]
                        
                        should_retry = any(err in error_str for err in retry_errors)
                        
                        if should_retry and retry < max_retries - 1:
                            wait_time = backoff_factor * (2 ** retry)  # Exponential backoff: 2s, 4s, 8s
                            logger.warning(f"Retrying {method_name} due to error: {e}")
                            time.sleep(wait_time)
                        else:
                            if "<!DOCTYPE html>" in error_str:
                                logger.error(f"Binance API returned HTML instead of JSON when using {method_name}.")
                                # Break out of retry loop but continue to next method
                                break
                            else:
                                logger.error(f"Failed to get account balance with {method_name}: {e}")
                                # Try next URL if this one is consistently failing
                                break
        
        # If futures API fails or we're in fallback mode, try spot API
        logger.info("Trying to get spot account balance as fallback")
        for retry in range(max_retries):
            try:
                account = self.client.get_account()
                for balance in account['balances']:
                    if balance['asset'] == 'USDT':
                        return float(balance['free'])
            except Exception as e:
                error_str = str(e)
                if retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying spot get_account due to error: {e}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to get spot account balance: {e}")
        
        # If all methods failed
        logger.error("All methods failed to get account balance. Using default balance.")
        return 0.0
    
    def get_position_info(self, symbol):
        """Get current position information"""
        max_retries = 5  # Increased from 3 to 5
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                # Add a brief initial pause for network stability on retry
                if retry > 0:
                    initial_pause = 0.5 * retry
                    time.sleep(initial_pause)
                
                positions = self.client.futures_position_information()
                for position in positions:
                    if position['symbol'] == symbol:
                        # Create a position data structure with safe access to fields
                        position_data = {
                            'symbol': position['symbol'],
                            'position_amount': float(position.get('positionAmt', 0)),
                            'entry_price': float(position.get('entryPrice', 0)),
                            'unrealized_profit': float(position.get('unRealizedProfit', 0)),
                            # Use get() with default value to avoid KeyError for missing fields
                            'leverage': int(position.get('leverage', 1)),
                            'isolated': position.get('isolated', False),
                        }
                        return position_data
                return None
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>",
                    "RemoteDisconnected"  # Added specific check for the error we're seeing
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)  # Exponential backoff
                    logger.warning(f"Retrying get_position_info due to error: {e}")
                    
                    # For connection-specific errors, try to reset the client connection
                    if "Connection aborted" in error_str or "RemoteDisconnected" in error_str:
                        logger.info("Connection issue detected. Attempting to re-sync time with server...")
                        try:
                            # Sync time to fix potential timestamp issues
                            self._sync_time()
                            # Add a slightly longer delay for connection issues
                            time.sleep(wait_time + 1)
                        except Exception as sync_error:
                            logger.warning(f"Failed to sync time: {sync_error}")
                    else:
                        time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Position info unavailable.")
                        return None
                    elif "RemoteDisconnected" in error_str or "Connection aborted" in error_str:
                        logger.error(f"Connection to Binance API was lost. Will try to rebuild connection on next call.")
                        # Force client re-initialization on next API call
                        try:
                            logger.info("Attempting to re-initialize client connection...")
                            self.client = self._initialize_client()
                        except Exception as reinit_error:
                            logger.error(f"Failed to re-initialize client: {reinit_error}")
                        return None
                    else:
                        logger.error(f"Failed to get position info: {e}")
                        return None
        
        logger.error("Maximum retries reached when getting position info")
        return None
    
    def get_symbol_info(self, symbol):
        """Get symbol information like price precision, quantity precision, etc."""
        max_retries = 3
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                exchange_info = self.client.futures_exchange_info()
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        return {
                            'price_precision': symbol_info['pricePrecision'],
                            'quantity_precision': symbol_info['quantityPrecision'],
                            'min_qty': float([f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['minQty']),
                            'max_qty': float([f for f in symbol_info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['maxQty']),
                            'min_notional': float([f for f in symbol_info['filters'] if f['filterType'] == 'MIN_NOTIONAL'][0]['notional'])
                        }
                return None
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying get_symbol_info due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Symbol info unavailable.")
                        return None
                    else:
                        logger.error(f"Failed to get symbol info: {e}")
                        return None
        
        logger.error("Maximum retries reached when getting symbol info")
        return None
    
    def get_historical_klines(self, symbol, interval, start_str, end_str=None, limit=1000):
        """Get historical candlestick data"""
        max_retries = 5  # Increased from 3 to 5 for historical data
        backoff_factor = 5  # Increased backoff for historical data fetching
        
        # Save current timeout setting
        current_timeout = self.client.options.get('timeout', 60)
        
        # Temporarily increase timeout for historical data
        self.client.options['timeout'] = 120  # 2-minute timeout for historical data
        
        logger.info(f"Fetching historical klines for {symbol}, period: {start_str} to {end_str or 'now'}")
        
        for retry in range(max_retries):
            try:
                # First try the futures API
                try:
                    klines = self.client.futures_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_str,
                        end_str=end_str,
                        limit=limit
                    )
                    logger.info(f"Successfully fetched {len(klines)} historical klines")
                    # Restore original timeout
                    self.client.options['timeout'] = current_timeout
                    return klines
                except Exception as futures_e:
                    logger.warning(f"Futures API failed: {futures_e}, trying spot API as fallback")
                    # Fall back to spot API
                    klines = self.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_str,
                        end_str=end_str,
                        limit=limit
                    )
                    logger.info(f"Successfully fetched {len(klines)} historical klines from spot API")
                    # Restore original timeout
                    self.client.options['timeout'] = current_timeout
                    return klines
                    
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors) or "timed out" in error_str.lower()
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying get_historical_klines (attempt {retry+1}/{max_retries}) due to error: {e}")
                    logger.warning(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Historical data unavailable.")
                        # Restore original timeout
                        self.client.options['timeout'] = current_timeout
                        return []
                    elif "unexpected keyword argument" in error_str:
                        # Handle the specific error about unexpected arguments
                        if "recvWindow" in error_str:
                            logger.warning("Trying historical_klines method without recvWindow parameter")
                            try:
                                # Try again without the problematic parameter
                                klines = self.client.futures_historical_klines(
                                    symbol=symbol,
                                    interval=interval,
                                    start_str=start_str,
                                    end_str=end_str,
                                    limit=limit
                                )
                                # Restore original timeout
                                self.client.options['timeout'] = current_timeout
                                return klines
                            except Exception as inner_e:
                                logger.error(f"Second attempt failed: {inner_e}")
                                # Restore original timeout
                                self.client.options['timeout'] = current_timeout
                                return []
                    else:
                        logger.error(f"Failed to get historical klines: {e}")
                        # Restore original timeout
                        self.client.options['timeout'] = current_timeout
                        return []
        
        logger.error("Maximum retries reached when getting historical klines")
        # Restore original timeout
        self.client.options['timeout'] = current_timeout
        return []
    
    def place_market_order(self, symbol, side, quantity):
        """Place a market order in futures market"""
        max_retries = 3
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,  # "BUY" or "SELL"
                    type="MARKET",
                    quantity=quantity
                )
                logger.info(f"Placed {side} market order for {quantity} {symbol}")
                return order
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying place_market_order due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Order placement failed.")
                        return None
                    else:
                        logger.error(f"Failed to place market order: {e}")
                        return None
        
        logger.error("Maximum retries reached when placing market order")
        return None
    
    def place_limit_order(self, symbol, side, quantity, price):
        """Place a limit order in futures market"""
        max_retries = 3
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                order = self.client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type="LIMIT",
                    timeInForce="GTC",  # Good Till Cancelled
                    quantity=quantity,
                    price=price
                )
                logger.info(f"Placed {side} limit order for {quantity} {symbol} at {price}")
                return order
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying place_limit_order due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Limit order placement failed.")
                        return None
                    else:
                        logger.error(f"Failed to place limit order: {e}")
                        return None
        
        logger.error("Maximum retries reached when placing limit order")
        return None
    
    def place_stop_loss_order(self, symbol, side, quantity, stop_price, price=None):
        """Place a stop loss order"""
        max_retries = 3
        backoff_factor = 2
        
        # First, cancel any existing stop loss orders for this symbol to avoid conflicts
        try:
            existing_orders = self.get_open_orders(symbol)
            for order in existing_orders:
                if (order.get('type') in ['STOP_MARKET', 'STOP'] and 
                    order.get('symbol') == symbol):
                    try:
                        self.client.futures_cancel_order(
                            symbol=symbol, 
                            orderId=order.get('orderId')
                        )
                        logger.info(f"Cancelled existing stop loss order {order.get('orderId')} for {symbol}")
                    except Exception as e:
                        logger.warning(f"Error cancelling existing stop loss order: {e}")
        except Exception as e:
            logger.warning(f"Error checking existing stop loss orders: {e}")
        
        # Place new stop loss order
        for retry in range(max_retries):
            try:
                params = {
                    'symbol': symbol,
                    'side': side,  # Opposite of position side
                    'type': 'STOP_MARKET',
                    'closePosition': 'true',
                    'stopPrice': stop_price,
                }
                if price:
                    params['type'] = 'STOP'
                    params['timeInForce'] = 'GTC'
                    params['quantity'] = quantity
                    params['price'] = price
                    
                order = self.client.futures_create_order(**params)
                logger.info(f"Placed stop loss order at {stop_price}")
                return order
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying place_stop_loss_order due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Stop loss order placement failed.")
                        return None
                    else:
                        logger.error(f"Failed to place stop loss: {e}")
                        return None
        
        logger.error("Maximum retries reached when placing stop loss order")
        return None
    
    def place_take_profit_order(self, symbol, side, quantity, stop_price, price=None):
        """Place a take profit order"""
        max_retries = 3
        backoff_factor = 2
        
        # First, cancel any existing take profit orders for this symbol to avoid conflicts
        try:
            existing_orders = self.get_open_orders(symbol)
            for order in existing_orders:
                if (order.get('type') in ['TAKE_PROFIT_MARKET', 'TAKE_PROFIT'] and 
                    order.get('symbol') == symbol):
                    try:
                        self.client.futures_cancel_order(
                            symbol=symbol, 
                            orderId=order.get('orderId')
                        )
                        logger.info(f"Cancelled existing take profit order {order.get('orderId')} for {symbol}")
                    except Exception as e:
                        logger.warning(f"Error cancelling existing take profit order: {e}")
        except Exception as e:
            logger.warning(f"Error checking existing take profit orders: {e}")
            
        # Place new take profit order
        for retry in range(max_retries):
            try:
                params = {
                    'symbol': symbol,
                    'side': side,  # Opposite of position side
                    'type': 'TAKE_PROFIT_MARKET',
                    'closePosition': 'true',
                    'stopPrice': stop_price,
                }
                if price:
                    params['type'] = 'TAKE_PROFIT'
                    params['timeInForce'] = 'GTC'
                    params['quantity'] = quantity
                    params['price'] = price
                    
                order = self.client.futures_create_order(**params)
                logger.info(f"Placed take profit order at {stop_price}")
                return order
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying place_take_profit_order due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Take profit order placement failed.")
                        return None
                    else:
                        logger.error(f"Failed to place take profit: {e}")
                        return None
        
        logger.error("Maximum retries reached when placing take profit order")
        return None
    
    def cancel_all_open_orders(self, symbol):
        """Cancel all open orders for a symbol"""
        try:
            result = self.client.futures_cancel_all_open_orders(symbol=symbol)
            logger.info(f"Cancelled all open orders for {symbol}")
            return result
        except BinanceAPIException as e:
            logger.error(f"Failed to cancel orders: {e}")
            return None
    
    def get_current_price(self, symbol):
        """Get current price of a symbol"""
        max_retries = 3
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                return float(ticker['price'])
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying get_current_price due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Current price unavailable.")
                        return None
                    else:
                        logger.error(f"Failed to get current price: {e}")
                        return None
        
        logger.error("Maximum retries reached when getting current price")
        return None
    
    def get_open_orders(self, symbol):
        """Get all open orders for a symbol"""
        max_retries = 3
        backoff_factor = 2
        
        for retry in range(max_retries):
            try:
                orders = self.client.futures_get_open_orders(symbol=symbol)
                logger.info(f"Retrieved {len(orders)} open orders for {symbol}")
                return orders
            except Exception as e:
                error_str = str(e)
                # Check for common error types that should be retried
                retry_errors = [
                    "Invalid JSON",
                    "Connection reset",
                    "Read timed out",
                    "Connection aborted",
                    "Connection refused",
                    "code=0",
                    "<!DOCTYPE html>"
                ]
                
                should_retry = any(err in error_str for err in retry_errors)
                
                if should_retry and retry < max_retries - 1:
                    wait_time = backoff_factor * (2 ** retry)
                    logger.warning(f"Retrying get_open_orders due to error: {e}")
                    time.sleep(wait_time)
                else:
                    if "<!DOCTYPE html>" in error_str:
                        logger.error(f"Binance API returned HTML instead of JSON. Open orders unavailable.")
                        return []
                    else:
                        logger.error(f"Failed to get open orders: {e}")
                        return []
        
        logger.error("Maximum retries reached when getting open orders")
        return []
    
    def get_position_related_orders(self, symbol):
        """Get all orders related to a position (stop loss and take profit orders)"""
        orders = self.get_open_orders(symbol)
        position_orders = []
        
        for order in orders:
            # Check if the order is a stop loss or take profit order AND matches our symbol
            if (order.get('type') in ['STOP_MARKET', 'STOP', 'TAKE_PROFIT_MARKET', 'TAKE_PROFIT']
                and order.get('symbol') == symbol):
                position_orders.append(order)
                
        logger.info(f"Found {len(position_orders)} position-related orders for {symbol}")
        return position_orders
    
    def cancel_position_orders(self, symbol):
        """
        Cancel all orders related to a position (stop loss and take profit orders)
        
        In multi-instance mode, this ensures only orders for the specific symbol are cancelled,
        allowing separate bot instances to operate independently for different trading pairs.
        """
        # Specifically get orders related to this symbol's position
        position_orders = self.get_position_related_orders(symbol)
        cancelled = 0
        
        # Double verify that we only cancel orders for the specified symbol
        for order in position_orders:
            try:
                order_id = order.get('orderId')
                order_symbol = order.get('symbol')
                
                # Extra check to ensure we only cancel orders for our specific symbol
                # This is critical for multi-instance mode to prevent interference
                if order_id and order_symbol == symbol:
                    self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                    order_type = order.get('type', 'unknown')
                    logger.info(f"Cancelled {order_type} order {order_id} for {symbol}")
                    cancelled += 1
                elif order_symbol != symbol:
                    # This should never happen given our filtering in get_position_related_orders
                    # But adding a safety check for robustness
                    logger.warning(f"Skipping cancellation of order {order_id} for {order_symbol} (not {symbol})")
            except BinanceAPIException as e:
                logger.error(f"Failed to cancel position order for {symbol}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error cancelling position order for {symbol}: {e}")
                
        logger.info(f"Cancelled {cancelled} position-related orders for {symbol}")
        return cancelled