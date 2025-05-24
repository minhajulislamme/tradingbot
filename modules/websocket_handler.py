import json
import logging
import threading
import time
from typing import Dict, Callable, Any, Optional, List
import websocket
from datetime import datetime

from modules.config import (
    TRADING_SYMBOL, TIMEFRAME, API_KEY, API_SECRET, 
    RETRY_COUNT, RETRY_DELAY, API_URL, RECV_WINDOW,
    API_TESTNET, WS_BASE_URL
)

logger = logging.getLogger(__name__)

class BinanceWebSocketManager:
    """
    WebSocket Manager for real-time Binance data
    Handles kline (candlestick) and user data WebSocket streams
    """
    
    # Use dynamic WebSocket URLs based on testnet setting
    BINANCE_WS_URL = f"{WS_BASE_URL}/ws"
    BINANCE_COMBINED_STREAM_URL = f"{WS_BASE_URL}/stream?streams="
    
    def __init__(self):
        self.ws = None
        self.ws_user = None
        self.running = False
        self.symbols = []  # List of symbols to track
        self.callbacks = {}  # Callbacks for different data types
        self.listen_key = None
        self.last_listen_key_update = None
        self.user_stream_connected = False
        
        # Lock for controlling reconnection attempts
        self.reconnect_lock = threading.Lock()
        self.is_reconnecting = False
        
        # Show testnet info if using testnet
        if API_TESTNET:
            logger.info("Operating in TESTNET mode - using Binance Futures testnet")
        
        # Timeframe mapping from config to WebSocket format
        self.timeframe_mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m',
            '30m': '30m', '1h': '1h', '2h': '2h', '4h': '4h',
            '6h': '6h', '8h': '8h', '12h': '12h', '1d': '1d',
            '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        # Store last received kline data
        self.last_kline_data = {}
        
        # Create threads for WebSocket connections
        self.ws_thread = None
        self.user_ws_thread = None
        self.keep_alive_thread = None
        
    def add_symbol(self, symbol: str):
        """Add a symbol to track via WebSocket"""
        symbol_lower = symbol.lower()
        if symbol_lower not in [s.lower() for s in self.symbols]:
            self.symbols.append(symbol)
            logger.info(f"Added {symbol} to WebSocket tracking")
            
            # If already running, reconnect to include new symbol
            if self.running:
                self.reconnect()
    
    def remove_symbol(self, symbol: str):
        """Remove a symbol from WebSocket tracking"""
        symbol_lower = symbol.lower()
        self.symbols = [s for s in self.symbols if s.lower() != symbol_lower]
        logger.info(f"Removed {symbol} from WebSocket tracking")
        
        # If already running, reconnect to update symbols
        if self.running and self.symbols:
            self.reconnect()
            
    def register_callback(self, data_type: str, callback: Callable):
        """Register a callback function for specific data type"""
        self.callbacks[data_type] = callback
        logger.debug(f"Registered callback for {data_type}")
        
    def _get_listen_key(self) -> Optional[str]:
        """Get a listen key for user data stream"""
        import requests
        
        if not API_KEY or not API_SECRET:
            logger.warning("API credentials not provided. User data stream unavailable.")
            return None
        
        try:
            # Use appropriate URL based on testnet setting
            base_url = API_URL.rstrip('/')
            url = f"{base_url}/fapi/v1/listenKey"
            
            headers = {
                "X-MBX-APIKEY": API_KEY
            }
            
            # For python-binance 1.0.28, we can use proper timeout and headers
            response = requests.post(
                url, 
                headers=headers, 
                timeout=10
            )
            
            # Check if response is valid JSON
            try:
                data = response.json()
            except ValueError:
                logger.error(f"Invalid JSON response from Binance: {response.text[:200]}")
                return None
            
            if "listenKey" in data:
                logger.info("Successfully obtained listen key for user data stream")
                return data["listenKey"]
            else:
                logger.error(f"Failed to get listen key: {data}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error getting listen key: {e}")
            return None
        except Exception as e:
            logger.error(f"Error getting listen key: {e}")
            return None
    
    def _keep_listen_key_alive(self):
        """Keep the listen key alive by pinging it periodically"""
        import requests
        import hmac
        import hashlib
        
        while self.running and self.listen_key:
            try:
                # Sleep for 30 minutes (keep alive required every 60 mins)
                time.sleep(30 * 60)
                
                if not self.running:
                    break
                
                # Create timestamp for authentication
                timestamp = int(time.time() * 1000)
                
                # Create signature for Binance API authentication
                query_string = f"timestamp={timestamp}&recvWindow={RECV_WINDOW}"
                signature = hmac.new(
                    API_SECRET.encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                # Full URL from config instead of hardcoded
                base_url = API_URL.rstrip('/')
                url = f"{base_url}/fapi/v1/listenKey"
                
                headers = {
                    "X-MBX-APIKEY": API_KEY,
                    "Content-Type": "application/json"
                }
                
                params = {
                    "timestamp": timestamp,
                    "recvWindow": RECV_WINDOW,
                    "signature": signature
                }
                
                # Make PUT request with proper timeout to extend listen key validity
                response = requests.put(
                    url, 
                    headers=headers, 
                    params=params, 
                    timeout=10
                )
                
                # Check response
                if response.status_code == 200:
                    logger.debug("Successfully refreshed listen key")
                else:
                    logger.warning(f"Failed to refresh listen key: {response.text}")
                    # Try to get a new listen key
                    self.listen_key = self._get_listen_key()
                    if self.listen_key and self.user_stream_connected:
                        self._restart_user_stream()
                        
            except requests.exceptions.RequestException as e:
                logger.error(f"Network error refreshing listen key: {e}")
                # Try to get a new listen key after network error
                time.sleep(5)  # Short delay to avoid hammering the API
                self.listen_key = self._get_listen_key()
                if self.listen_key and self.user_stream_connected:
                    self._restart_user_stream()
            except Exception as e:
                logger.error(f"Error keeping listen key alive: {e}")
                time.sleep(60)  # Wait before retrying
    
    def start(self):
        """Start WebSocket connections"""
        if not self.symbols:
            self.add_symbol(TRADING_SYMBOL)
            
        self.running = True
        
        # Start market data WebSocket
        self.ws_thread = threading.Thread(target=self._start_market_stream)
        self.ws_thread.daemon = True
        self.ws_thread.start()
        
        # Start user data WebSocket if credentials available
        if API_KEY and API_SECRET:
            self.listen_key = self._get_listen_key()
            if self.listen_key:
                self.user_ws_thread = threading.Thread(target=self._start_user_stream)
                self.user_ws_thread.daemon = True
                self.user_ws_thread.start()
                
                # Start keep-alive thread
                self.keep_alive_thread = threading.Thread(target=self._keep_listen_key_alive)
                self.keep_alive_thread.daemon = True
                self.keep_alive_thread.start()
    
    def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        
        # Close market data WebSocket
        if self.ws:
            self.ws.close()
            self.ws = None
            
        # Close user data WebSocket
        if self.ws_user:
            self.ws_user.close()
            self.ws_user = None
            self.user_stream_connected = False
            
        logger.info("WebSocket connections closed")
    
    def reconnect(self):
        """Reconnect all WebSocket connections"""
        self.stop()
        time.sleep(1)  # Brief pause before reconnecting
        self.start()
    
    def _start_market_stream(self):
        """Start a WebSocket connection for market data"""
        if not self.symbols:
            logger.warning("No symbols provided for market data stream")
            return
        
        # Create subscription list for all symbols
        streams = []
        for symbol in self.symbols:
            symbol_lower = symbol.lower()
            
            # Add kline stream for each symbol
            timeframe = self.timeframe_mapping.get(TIMEFRAME, '15m')
            streams.append(f"{symbol_lower}@kline_{timeframe}")
            
            # Add trade stream
            streams.append(f"{symbol_lower}@trade")
            
            # Add book ticker stream for best bid/ask
            streams.append(f"{symbol_lower}@bookTicker")
        
        # Create combined stream URL
        stream_url = self.BINANCE_COMBINED_STREAM_URL + "/".join(streams)
        
        # Connect WebSocket with retry logic
        for attempt in range(RETRY_COUNT):
            try:
                ws_app = websocket.WebSocketApp(
                    stream_url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open
                )
                
                self.ws = ws_app
                logger.info(f"Starting market data WebSocket connection (attempt {attempt+1})")
                ws_app.run_forever()
                
                # If we reach here, connection closed intentionally or unintentionally
                if not self.running:
                    logger.info("Market data WebSocket closed as requested")
                    break
                    
                logger.warning(f"Market data WebSocket connection lost. Reconnecting...")
                time.sleep(RETRY_DELAY)
                
            except Exception as e:
                logger.error(f"Error in market data WebSocket: {e}")
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Max retry attempts reached. Giving up on market data WebSocket.")
                    break
    
    def _start_user_stream(self):
        """Start a WebSocket connection for user data"""
        if not self.listen_key:
            logger.warning("No listen key available for user data stream")
            return
        
        # Connect to user data stream
        user_stream_url = f"{self.BINANCE_WS_URL}/{self.listen_key}"
        
        # Connect with retry logic
        for attempt in range(RETRY_COUNT):
            try:
                ws_app = websocket.WebSocketApp(
                    user_stream_url,
                    on_message=self._on_user_message,
                    on_error=self._on_user_error,
                    on_close=self._on_user_close,
                    on_open=self._on_user_open
                )
                
                self.ws_user = ws_app
                self.user_stream_connected = True
                logger.info(f"Starting user data WebSocket connection (attempt {attempt+1})")
                ws_app.run_forever()
                
                # If we reach here, connection closed intentionally or unintentionally
                if not self.running:
                    logger.info("User data WebSocket closed as requested")
                    self.user_stream_connected = False
                    break
                    
                logger.warning(f"User data WebSocket connection lost. Reconnecting...")
                time.sleep(RETRY_DELAY)
                self.user_stream_connected = False
                
            except Exception as e:
                logger.error(f"Error in user data WebSocket: {e}")
                self.user_stream_connected = False
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("Max retry attempts reached. Giving up on user data WebSocket.")
                    break
    
    def _restart_user_stream(self):
        """Restart the user data stream with a new listen key"""
        if self.ws_user:
            self.ws_user.close()
            self.ws_user = None
        
        self.user_stream_connected = False
        if self.listen_key:
            self.user_ws_thread = threading.Thread(target=self._start_user_stream)
            self.user_ws_thread.daemon = True
            self.user_ws_thread.start()
    
    def _on_message(self, ws, message):
        """Handle market data WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Check if it's a combined stream format
            if 'data' in data and 'stream' in data:
                stream = data['stream']
                event_data = data['data']
                
                # Handle kline data
                if 'kline' in stream:
                    self._process_kline_data(event_data)
                
                # Handle trade data
                elif 'trade' in stream:
                    self._process_trade_data(event_data)
                
                # Handle book ticker data
                elif 'bookTicker' in stream:
                    self._process_book_ticker_data(event_data)
            else:
                logger.debug(f"Received unknown message format: {message[:100]}...")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_user_message(self, ws, message):
        """Handle user data WebSocket messages"""
        try:
            data = json.loads(message)
            
            # Handle different event types
            event_type = data.get('e', '')
            
            # Account update
            if event_type == 'ACCOUNT_UPDATE':
                self._process_account_update(data)
                
            # Order update
            elif event_type == 'ORDER_TRADE_UPDATE':
                self._process_order_update(data)
                
            # Position update
            elif event_type == 'MARGIN_CALL':
                self._process_margin_call(data)
            
            # Account config update
            elif event_type == 'ACCOUNT_CONFIG_UPDATE':
                logger.info(f"Account configuration updated: {data}")
                
            # Handle listen key expired
            elif event_type == 'listenKeyExpired':
                logger.warning("Listen key expired. Getting a new one...")
                self.listen_key = self._get_listen_key()
                if self.listen_key:
                    self._restart_user_stream()
            
            else:
                logger.debug(f"Received unknown user data event: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing user data message: {e}")
    
    def _on_error(self, ws, error):
        """Handle market data WebSocket errors"""
        logger.error(f"Market data WebSocket error: {error}")
    
    def _on_user_error(self, ws, error):
        """Handle user data WebSocket errors"""
        logger.error(f"User data WebSocket error: {error}")
        self.user_stream_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle market data WebSocket closure"""
        logger.info(f"Market data WebSocket closed: {close_status_code} {close_msg}")
        
        # Auto-reconnect if not intentionally stopped and running
        if self.running:
            # Use lock to prevent multiple reconnection attempts
            with self.reconnect_lock:
                if not self.is_reconnecting:
                    self.is_reconnecting = True
                    time.sleep(1)
                    # Start a new thread for reconnection
                    threading.Thread(target=self._reconnect_market_stream).start()
    
    def _reconnect_market_stream(self):
        """Handle reconnection of market data stream"""
        try:
            # Close existing connection if any
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
                
            # Start a new connection
            logger.info("Attempting to reconnect market data WebSocket...")
            self._start_market_stream()
        finally:
            # Reset reconnection flag
            with self.reconnect_lock:
                self.is_reconnecting = False
    
    def _on_user_close(self, ws, close_status_code, close_msg):
        """Handle user data WebSocket closure"""
        logger.info(f"User data WebSocket closed: {close_status_code} {close_msg}")
        self.user_stream_connected = False
        
        # Auto-reconnect if not intentionally stopped and running
        if self.running:
            time.sleep(1)
            # Create a new thread for user stream reconnection
            new_thread = threading.Thread(target=self._start_user_stream)
            new_thread.daemon = True
            new_thread.start()
    
    def _on_open(self, ws):
        """Handle market data WebSocket opening"""
        logger.info("Market data WebSocket connected")
        
    def _on_user_open(self, ws):
        """Handle user data WebSocket opening"""
        logger.info("User data WebSocket connected")
        self.user_stream_connected = True
    
    def _process_kline_data(self, data):
        """Process kline (candlestick) data"""
        # Extract and format kline data
        kline = data.get('k', {})
        symbol = kline.get('s', '')
        
        # Update last kline data for this symbol
        self.last_kline_data[symbol] = {
            'open_time': kline.get('t'),
            'open': float(kline.get('o')),
            'high': float(kline.get('h')),
            'low': float(kline.get('l')),
            'close': float(kline.get('c')),
            'volume': float(kline.get('v')),
            'close_time': kline.get('T'),
            'is_closed': kline.get('x', False)
        }
        
        # If kline is closed and we have a callback, call it
        if kline.get('x', False) and 'kline' in self.callbacks:
            self.callbacks['kline'](symbol, self.last_kline_data[symbol])
            
        # Always call real-time kline callback if registered
        if 'kline_update' in self.callbacks:
            self.callbacks['kline_update'](symbol, self.last_kline_data[symbol])
    
    def _process_trade_data(self, data):
        """Process trade data"""
        trade_data = {
            'symbol': data.get('s', ''),
            'price': float(data.get('p', 0)),
            'quantity': float(data.get('q', 0)),
            'time': data.get('T', 0),
            'buyer_maker': data.get('m', False),
            'trade_id': data.get('t', 0)
        }
        
        # Call trade callback if registered
        if 'trade' in self.callbacks:
            self.callbacks['trade'](trade_data['symbol'], trade_data)
    
    def _process_book_ticker_data(self, data):
        """Process book ticker data (best bid/ask)"""
        ticker_data = {
            'symbol': data.get('s', ''),
            'bid_price': float(data.get('b', 0)),
            'bid_qty': float(data.get('B', 0)),
            'ask_price': float(data.get('a', 0)),
            'ask_qty': float(data.get('A', 0)),
            'time': data.get('E', 0)
        }
        
        # Call book ticker callback if registered
        if 'book_ticker' in self.callbacks:
            self.callbacks['book_ticker'](ticker_data['symbol'], ticker_data)
    
    def _process_account_update(self, data):
        """Process account update data"""
        update = data.get('a', {})
        
        # Extract balance changes
        balances = update.get('B', [])
        balance_updates = {}
        for balance in balances:
            asset = balance.get('a', '')
            wallet_balance = float(balance.get('wb', 0))
            balance_updates[asset] = wallet_balance
        
        # Extract position changes
        positions = update.get('P', [])
        position_updates = {}
        for position in positions:
            symbol = position.get('s', '')
            position_amount = float(position.get('pa', 0))
            entry_price = float(position.get('ep', 0))
            unrealized_pnl = float(position.get('up', 0))
            position_updates[symbol] = {
                'position_amount': position_amount,
                'entry_price': entry_price,
                'unrealized_pnl': unrealized_pnl
            }
        
        # Call account update callback if registered
        if 'account_update' in self.callbacks:
            self.callbacks['account_update'](balance_updates, position_updates)
    
    def _process_order_update(self, data):
        """Process order update data"""
        order = data.get('o', {})
        order_data = {
            'symbol': order.get('s', ''),
            'client_order_id': order.get('c', ''),
            'side': order.get('S', ''),
            'type': order.get('o', ''),
            'time_in_force': order.get('f', ''),
            'quantity': float(order.get('q', 0)),
            'price': float(order.get('p', 0)),
            'avg_price': float(order.get('ap', 0)),
            'stop_price': float(order.get('sp', 0)),
            'execution_type': order.get('x', ''),
            'order_status': order.get('X', ''),
            'order_id': order.get('i', 0),
            'filled_quantity': float(order.get('l', 0)),
            'cumulative_filled_quantity': float(order.get('z', 0)),
            'last_filled_price': float(order.get('L', 0)),
            'commission': float(order.get('n', 0)),
            'commission_asset': order.get('N', ''),
            'trade_time': order.get('T', 0),
            'trade_id': order.get('t', 0),
            'realized_profit': float(order.get('rp', 0))
        }
        
        # Call order update callback if registered
        if 'order_update' in self.callbacks:
            self.callbacks['order_update'](order_data)
    
    def _process_margin_call(self, data):
        """Process margin call data"""
        positions = data.get('p', [])
        margin_calls = []
        
        for position in positions:
            margin_call = {
                'symbol': position.get('s', ''),
                'position_side': position.get('ps', ''),
                'position_amount': float(position.get('pa', 0)),
                'margin_type': position.get('mt', ''),
                'isolated_wallet': float(position.get('iw', 0)),
                'mark_price': float(position.get('mp', 0)),
                'unrealized_pnl': float(position.get('up', 0)),
                'maintenance_margin_required': float(position.get('mm', 0))
            }
            margin_calls.append(margin_call)
        
        # Call margin call callback if registered
        if 'margin_call' in self.callbacks:
            self.callbacks['margin_call'](margin_calls)
    
    def get_last_kline(self, symbol: str) -> Dict:
        """Get the last received kline data for a symbol"""
        return self.last_kline_data.get(symbol, {})
    
    def get_symbols(self) -> List[str]:
        """Get list of symbols currently tracked"""
        return self.symbols.copy()
    
    def is_connected(self) -> bool:
        """Check if WebSocket is connected"""
        return self.running and self.ws is not None
    
    def is_user_connected(self) -> bool:
        """Check if user data WebSocket is connected"""
        return self.running and self.user_stream_connected