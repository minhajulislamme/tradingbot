import logging
import math
import time
from datetime import datetime, timedelta
from modules.config import (
    MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, 
    TRAILING_STOP, TRAILING_STOP_PCT,
    USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
    TRAILING_TAKE_PROFIT, TRAILING_TAKE_PROFIT_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT, COMPOUND_INTERVAL,
    # Multi-instance mode settings
    MULTI_INSTANCE_MODE, MAX_POSITIONS_PER_SYMBOL,
    # Futures trading settings
    LEVERAGE,
    # Margin safety settings
    MARGIN_SAFETY_FACTOR, MAX_POSITION_SIZE_PCT, MIN_FREE_BALANCE_PCT,
    # Fixed percentage trading
    FIXED_TRADE_PERCENTAGE
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, binance_client):
        """Initialize risk manager with a reference to binance client"""
        self.binance_client = binance_client
        self.current_market_condition = None  # Keeping for compatibility
        self.last_compound_time = None
        self.initial_balance = None
        self.last_balance = None
        self.position_size_multiplier = 1.0  # Default position size multiplier
        
    def set_market_condition(self, market_condition):
        """Set the current market condition for compatibility with existing code"""
        if market_condition in ['BULLISH', 'BEARISH', 'SIDEWAYS', 'EXTREME_BULLISH', 'EXTREME_BEARISH', 'SQUEEZE']:
            if self.current_market_condition != market_condition:
                logger.info(f"Market condition changed to {market_condition}")
                self.current_market_condition = market_condition
        
    def calculate_position_size(self, symbol, side, price, stop_loss_price=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            price: Current market price
            stop_loss_price: Optional stop loss price for calculating risk
            
        Returns:
            quantity: The position size
        """
        # Get account balance
        balance = self.binance_client.get_account_balance()
        
        if balance <= 0:
            logger.error("Insufficient balance to open a position")
            return 0
            
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {symbol}")
            return 0
            
        # Use 75% of account balance instead of RISK_PER_TRADE
        # This is a fixed percentage approach rather than a risk-based approach
        trade_amount = balance * FIXED_TRADE_PERCENTAGE
        
        # Apply position size multiplier from strategy
        adjusted_trade_amount = trade_amount * self.position_size_multiplier
        logger.debug(f"Trade amount adjusted from {trade_amount:.4f} to {adjusted_trade_amount:.4f} (multiplier: {self.position_size_multiplier:.2f})")
        risk_amount = adjusted_trade_amount  # Keep variable name for compatibility
        
        # Calculate position size based on the fixed percentage and stop loss
        if stop_loss_price and USE_STOP_LOSS:
            # If stop loss is provided, calculate size based on it and available margin
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error("Stop loss too close to entry price")
                return 0
                
            # Calculate max quantity based on available funds and 10x leverage
            max_quantity = (risk_amount * LEVERAGE) / price
        else:
            # If no stop loss, use the fixed percentage approach with leverage
            max_quantity = (risk_amount * LEVERAGE) / price
        
        # Apply precision to quantity
        quantity_precision = symbol_info['quantity_precision']
        quantity = round_step_size(max_quantity, get_step_size(symbol_info['min_qty']))
        
        # Check minimum notional
        min_notional = symbol_info['min_notional']
        if quantity * price < min_notional:
            logger.warning(f"Position size too small - below minimum notional of {min_notional}")
            
            # Try to adjust to meet minimum notional
            min_quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
            
            # Make sure we don't use more than MAX_POSITION_SIZE_PCT of balance
            max_safe_quantity = (balance * MAX_POSITION_SIZE_PCT) / price
            max_safe_quantity = math.floor(max_safe_quantity * 10**quantity_precision) / 10**quantity_precision
            
            quantity = min(min_quantity, max_safe_quantity)
            
            if quantity * price > balance * MAX_POSITION_SIZE_PCT:
                logger.warning(f"Position would use more than {MAX_POSITION_SIZE_PCT*100}% of balance - reducing size")
                quantity = math.floor((balance * MAX_POSITION_SIZE_PCT / price) * 10**quantity_precision) / 10**quantity_precision
        
        # FUTURES MARGIN CHECK - Calculate required margin and check if it's within our limits
        # Get current leverage
        leverage = LEVERAGE  # From config
        
        # Calculate required margin for the position
        required_margin = (quantity * price) / leverage
        
        # Use margin safety factor from config
        max_safe_margin = balance * MARGIN_SAFETY_FACTOR
        
        # Always keep a minimum free balance
        min_free_balance = balance * MIN_FREE_BALANCE_PCT
        max_safe_margin = min(max_safe_margin, balance - min_free_balance)
        
        # If required margin exceeds safe limit, adjust position size
        if required_margin > max_safe_margin:
            logger.warning(f"Required margin ({required_margin:.4f} USDT) exceeds safe limit ({max_safe_margin:.4f} USDT)")
            
            # Calculate maximum safe quantity based on available margin
            max_margin_quantity = (max_safe_margin * leverage) / price
            max_margin_quantity = math.floor(max_margin_quantity * 10**quantity_precision) / 10**quantity_precision
            
            # Update quantity to safe margin amount
            old_quantity = quantity
            quantity = max_margin_quantity
            
            logger.warning(f"Reducing position size from {old_quantity} to {quantity} due to margin constraints")
            
        # Final check to ensure we have a valid quantity
        if quantity <= 0:
            logger.error("Balance too low to open even minimum position")
            return 0
                
        logger.info(f"Calculated position size: {quantity} units at {price} per unit")
        # Log margin requirements for transparency
        logger.debug(f"Margin required: {(quantity * price) / leverage:.4f} USDT, Available balance: {balance:.4f} USDT")
        
        return quantity
        
    def should_open_position(self, symbol):
        """Check if a new position should be opened based on risk rules"""
        # Check if we already have an open position for this symbol
        position_info = self.binance_client.get_position_info(symbol)
        if position_info and abs(position_info['position_amount']) > 0:
            logger.info(f"Already have an open position for {symbol}")
            return False
            
        # Check maximum number of open positions
        if MULTI_INSTANCE_MODE:
            # In multi-instance mode, only count positions for the current symbol
            positions = self.binance_client.client.futures_position_information()
            # Check if we've reached the max positions for this symbol
            symbol_positions = [p for p in positions if p['symbol'] == symbol and float(p['positionAmt']) != 0]
            if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
                logger.info(f"Maximum number of positions for {symbol} ({MAX_POSITIONS_PER_SYMBOL}) reached")
                return False
        else:
            # Original behavior - count all positions
            positions = self.binance_client.client.futures_position_information()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.info(f"Maximum number of open positions ({MAX_OPEN_POSITIONS}) reached")
                return False
            
        return True
        
    def calculate_stop_loss(self, symbol, side, entry_price):
        """Calculate stop loss price based on configuration"""
        if not USE_STOP_LOSS:
            return None
            
        if side == "BUY":  # Long position
            stop_price = entry_price * (1 - STOP_LOSS_PCT)
        else:  # Short position
            stop_price = entry_price * (1 + STOP_LOSS_PCT)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_price = round(stop_price, price_precision)
            
        logger.info(f"Calculated stop loss at {stop_price} ({STOP_LOSS_PCT*100}%)")
        return stop_price
        
    def adjust_stop_loss_for_trailing(self, symbol, side, current_price, position_info=None):
        """Adjust stop loss for trailing stop if needed"""
        if not TRAILING_STOP:
            return None
            
        if not position_info:
            # Get position info specifically for this symbol
            position_info = self.binance_client.get_position_info(symbol)
            
        # Only proceed if we have a valid position for this specific symbol
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        # Ensure we're dealing with the right symbol
        if position_info['symbol'] != symbol:
            logger.warning(f"Position symbol mismatch: expected {symbol}, got {position_info['symbol']}")
            return None
            
        entry_price = position_info['entry_price']
        
        # Calculate new stop loss based on current price
        if side == "BUY":  # Long position
            new_stop = current_price * (1 - TRAILING_STOP_PCT)
            # Only move stop loss up, never down
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop <= current_stop:
                logger.debug(f"Not adjusting trailing stop: current ({current_stop}) > calculated ({new_stop})")
                return None
        else:  # Short position
            new_stop = current_price * (1 + TRAILING_STOP_PCT)
            # Only move stop loss down, never up
            current_stop = self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop >= current_stop:
                logger.debug(f"Not adjusting trailing stop: current ({current_stop}) < calculated ({new_stop})")
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop = round(new_stop, price_precision)
            
        logger.info(f"Adjusted trailing stop loss to {new_stop} ({TRAILING_STOP_PCT*100}%)")
        logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Stop loss moved: {current_stop} -> {new_stop}")
        return new_stop
        
    # For API compatibility with existing code
    def calculate_take_profit(self, symbol, side, entry_price):
        """Calculate take profit price based on configuration"""
        if not USE_TAKE_PROFIT:
            return None
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PCT)
        else:  # Short position
            take_profit_price = entry_price * (1 - TAKE_PROFIT_PCT)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        logger.info(f"Calculated take profit at {take_profit_price} ({TAKE_PROFIT_PCT*100}%)")
        return take_profit_price

    def adjust_take_profit_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust take profit target for trailing take profit if enabled
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL' position side
            current_price: Current market price
            position_info: Optional position information (will be fetched if not provided)
            
        Returns:
            float or None: New take profit price if adjusted, None otherwise
        """
        if not TRAILING_TAKE_PROFIT:
            return None
            
        if not position_info:
            # Get position info specifically for this symbol
            position_info = self.binance_client.get_position_info(symbol)
            
        # Only proceed if we have a valid position for this specific symbol
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        # Ensure we're dealing with the right symbol
        if position_info['symbol'] != symbol:
            logger.warning(f"Position symbol mismatch: expected {symbol}, got {position_info['symbol']}")
            return None
            
        entry_price = position_info['entry_price']
        
        # Calculate new take profit based on current price
        if side == "BUY":  # Long position
            new_tp = current_price * (1 + TRAILING_TAKE_PROFIT_PCT)
            # Only move take profit down, never up (to lock in profits)
            current_tp = self.calculate_take_profit(symbol, side, entry_price)
            if current_tp and new_tp >= current_tp:
                logger.debug(f"Not adjusting trailing take profit: current ({current_tp}) < calculated ({new_tp})")
                return None
        else:  # Short position
            new_tp = current_price * (1 - TRAILING_TAKE_PROFIT_PCT)
            # Only move take profit up, never down (to lock in profits)
            current_tp = self.calculate_take_profit(symbol, side, entry_price)
            if current_tp and new_tp <= current_tp:
                logger.debug(f"Not adjusting trailing take profit: current ({current_tp}) > calculated ({new_tp})")
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_tp = round(new_tp, price_precision)
            
        logger.info(f"Adjusted trailing take profit to {new_tp} ({TRAILING_TAKE_PROFIT_PCT*100}%)")
        logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {current_tp} -> {new_tp}")
        return new_tp

    def calculate_partial_take_profits(self, symbol, side, entry_price):
        """
        Calculate multiple take profit levels for partial position closing
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL' position side
            entry_price: Entry price of the position
            
        Returns:
            list: List of dictionaries with 'price', 'percentage', and 'pct_from_entry' keys
        """
        if not USE_TAKE_PROFIT:
            return []
            
        # Define take profit levels as percentages from entry price
        if side == "BUY":  # Long position
            tp_levels = [
                {'pct_from_entry': 2.5, 'percentage': 0.5},   # Take 50% profit at 2.5% gain
                {'pct_from_entry': 5.0, 'percentage': 0.5}    # Take remaining 50% at 5% gain
            ]
        else:  # Short position
            tp_levels = [
                {'pct_from_entry': -2.5, 'percentage': 0.5},  # Take 50% profit at 2.5% gain
                {'pct_from_entry': -5.0, 'percentage': 0.5}   # Take remaining 50% at 5% gain
            ]
        
        # Convert percentage gains to actual prices
        symbol_info = self.binance_client.get_symbol_info(symbol)
        price_precision = symbol_info['price_precision'] if symbol_info else 4
        
        take_profit_levels = []
        for level in tp_levels:
            if side == "BUY":
                price = entry_price * (1 + level['pct_from_entry'] / 100)
            else:
                price = entry_price * (1 - abs(level['pct_from_entry']) / 100)
                
            price = round(price, price_precision)
            
            take_profit_levels.append({
                'price': price,
                'percentage': level['percentage'],
                'pct_from_entry': abs(level['pct_from_entry'])
            })
            
        logger.info(f"Calculated {len(take_profit_levels)} partial take profit levels for {symbol} {side} position")
        return take_profit_levels
    
    def update_balance_for_compounding(self):
        """Update balance tracking for auto-compounding"""
        if not AUTO_COMPOUND:
            return False
            
        # Get current account balance
        current_balance = self.binance_client.get_account_balance()
        
        # Initialize balance tracking if needed
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.last_balance = current_balance
            self.last_compound_time = datetime.now()
            logger.info(f"Initialized compounding with balance: {current_balance}")
            return False
            
        # Check if it's time to compound based on the configured interval
        now = datetime.now()
        compound_interval_days = 1  # Default to daily
        
        if COMPOUND_INTERVAL == 'HOURLY':
            compound_interval_days = 1/24
        elif COMPOUND_INTERVAL == 'DAILY':
            compound_interval_days = 1
        elif COMPOUND_INTERVAL == 'WEEKLY':
            compound_interval_days = 7
        elif COMPOUND_INTERVAL == 'MONTHLY':
            compound_interval_days = 30
            
        time_since_last_compound = now - self.last_compound_time
        
        # Check if it's time to compound
        if time_since_last_compound.total_seconds() < compound_interval_days * 24 * 3600:
            return False
            
        # Calculate profit
        profit = current_balance - self.last_balance
        
        if profit <= 0:
            logger.info(f"No profit to compound. Current balance: {current_balance}, Previous: {self.last_balance}")
            self.last_compound_time = now
            self.last_balance = current_balance
            return False
            
        # Apply compounding by updating risk amount
        # This effectively increases position sizes based on profits
        compound_amount = profit * COMPOUND_REINVEST_PERCENT
        logger.info(f"Compounding {COMPOUND_REINVEST_PERCENT*100}% of profit: {profit} = {compound_amount}")
        
        # Update last compound time and balance
        self.last_compound_time = now
        self.last_balance = current_balance
        
        return True
    
    # For API compatibility with existing code
    def calculate_volatility_based_stop_loss(self, symbol, side, entry_price, klines=None):
        """Simplified to use regular stop loss instead of complex volatility-based logic"""
        return self.calculate_stop_loss(symbol, side, entry_price)
    
    def get_current_risk_level(self, symbol=None):
        """
        Get the current risk level for a symbol
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            float: Current risk level (0.0-1.0)
        """
        # Fixed trade percentage
        base_risk = FIXED_TRADE_PERCENTAGE
        
        # Apply position size multiplier
        dynamic_risk = base_risk * self.position_size_multiplier
        
        # Clamp to reasonable range (0.10-1.00)
        return max(0.10, min(1.00, dynamic_risk))
        
    # Method for handling dynamic position sizing from strategies
    def update_position_sizing(self, position_size=None):
        """
        Update position sizing based on market conditions provided by strategy
        
        Args:
            position_size: A position size multiplier (e.g., 0.8 means 80% of base position)
        """
        if position_size is None:
            return
        
        try:
            # Ensure position_size is a valid float
            position_size = float(position_size)
            
            # Clamp the position size multiplier to reasonable values (0.1 to 2.0)
            position_size = max(0.1, min(2.0, position_size))
            
            self.position_size_multiplier = position_size
            logger.debug(f"Position size multiplier updated to {position_size:.2f}")
        except (ValueError, TypeError) as e:
            logger.error(f"Error updating position size multiplier: {e}")
            # Keep the current multiplier
        return

    def test_position_sizing(self, symbol='SUIUSDT'):
        """
        Test method to verify position sizing and risk management are working correctly
        Returns details about current risk settings
        
        Args:
            symbol: Trading symbol to test with
        
        Returns:
            dict: Information about current risk settings
        """
        current_price = self.binance_client.get_symbol_price(symbol)
        balance = self.binance_client.get_account_balance()
        
        # Calculate trade amount (75% of balance)
        trade_amount = balance * FIXED_TRADE_PERCENTAGE
        
        # Calculate adjusted amount with position sizing
        adjusted_trade_amount = trade_amount * self.position_size_multiplier
        
        # Calculate theoretical position sizes with 10x leverage
        base_position_size = (trade_amount * LEVERAGE) / current_price
        adjusted_position_size = (adjusted_trade_amount * LEVERAGE) / current_price
        
        # Calculate margin requirements
        base_margin_required = (base_position_size * current_price) / LEVERAGE
        adjusted_margin_required = (adjusted_position_size * current_price) / LEVERAGE
        
        # Calculate maximum position size based on available margin
        max_safe_margin = balance * MARGIN_SAFETY_FACTOR
        # Ensure we keep minimum free balance
        min_free_balance = balance * MIN_FREE_BALANCE_PCT
        max_safe_margin = min(max_safe_margin, balance - min_free_balance)
        max_margin_position_size = (max_safe_margin * LEVERAGE) / current_price
        
        # Check if margin is sufficient
        margin_sufficient = adjusted_margin_required <= max_safe_margin
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'account_balance': balance,
            'trade_pct_of_balance': FIXED_TRADE_PERCENTAGE,
            'base_trade_amount': trade_amount,
            'position_size_multiplier': self.position_size_multiplier,
            'adjusted_trade_amount': adjusted_trade_amount,
            'base_position_size': base_position_size,
            'adjusted_position_size': adjusted_position_size,
            'leverage': LEVERAGE,
            'base_margin_required': base_margin_required,
            'adjusted_margin_required': adjusted_margin_required,
            'max_safe_margin': max_safe_margin,
            'max_position_size_by_margin': max_margin_position_size,
            'margin_sufficient': margin_sufficient,
            'market_condition': self.current_market_condition
        }

    def check_margin_sufficient(self, symbol, price, quantity):
        """
        Check if there's sufficient margin available for the requested position size
        
        Args:
            symbol: Trading pair symbol
            price: Current market price
            quantity: Position size to check
            
        Returns:
            bool: True if there's sufficient margin, False otherwise
        """
        # Get account balance
        balance = self.binance_client.get_account_balance()
        
        # Calculate required margin
        required_margin = (quantity * price) / LEVERAGE
        
        # Use margin safety factor from config
        max_safe_margin = balance * MARGIN_SAFETY_FACTOR
        
        # Always keep a minimum free balance
        min_free_balance = balance * MIN_FREE_BALANCE_PCT
        max_safe_margin = min(max_safe_margin, balance - min_free_balance)
        
        if required_margin > max_safe_margin:
            logger.warning(f"Insufficient margin: Required {required_margin:.4f} USDT, Available {max_safe_margin:.4f} USDT")
            return False
        
        logger.debug(f"Margin check passed: Required {required_margin:.4f} USDT, Available {max_safe_margin:.4f} USDT")
        return True


# Helper functions
def get_step_size(min_qty_str):
    """Extract step size from min quantity string"""
    step_size = min_qty_str
    if isinstance(step_size, str):
        try:
            step_size = float(step_size)
        except ValueError:
            return 0.001  # Default step size
    
    if step_size == 0:
        return 0.001  # Default step size
    
    return step_size

def round_step_size(quantity, step_size):
    """Round quantity to valid step size"""
    if step_size == 0:
        return quantity
        
    precision = int(round(-math.log10(step_size)))
    if precision < 0:
        precision = 0
    rounded = math.floor(quantity * 10**precision) / 10**precision
    
    # Ensure it's at least the step size
    if rounded < step_size:
        rounded = step_size
        
    return rounded