import logging
import math
import time
from datetime import datetime, timedelta
from modules.config import (
    RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, 
    TRAILING_STOP, TRAILING_STOP_PCT,
    USE_TAKE_PROFIT, TAKE_PROFIT_PCT,
    TRAILING_TAKE_PROFIT, TRAILING_TAKE_PROFIT_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT, COMPOUND_INTERVAL,
    # Multi-instance mode settings
    MULTI_INSTANCE_MODE, MAX_POSITIONS_PER_SYMBOL
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
            
        # Calculate risk amount
        risk_amount = balance * RISK_PER_TRADE
        
        # Apply position size multiplier from strategy
        adjusted_risk_amount = risk_amount * self.position_size_multiplier
        logger.debug(f"Risk amount adjusted from {risk_amount:.4f} to {adjusted_risk_amount:.4f} (multiplier: {self.position_size_multiplier:.2f})")
        risk_amount = adjusted_risk_amount
        
        # Calculate position size based on risk and stop loss
        if stop_loss_price and USE_STOP_LOSS:
            # If stop loss is provided, calculate size based on it
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error("Stop loss too close to entry price")
                return 0
                
            # Calculate max quantity based on risk
            max_quantity = risk_amount / risk_per_unit
        else:
            # If no stop loss, use a percentage of balance
            max_quantity = (balance * RISK_PER_TRADE) / price
        
        # Apply precision to quantity
        quantity_precision = symbol_info['quantity_precision']
        quantity = round_step_size(max_quantity, get_step_size(symbol_info['min_qty']))
        
        # Check minimum notional
        min_notional = symbol_info['min_notional']
        if quantity * price < min_notional:
            logger.warning(f"Position size too small - below minimum notional of {min_notional}")
            
            # Try to adjust to meet minimum notional
            min_quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
            
            # Make sure we don't use more than 50% of balance
            max_safe_quantity = (balance * 0.5) / price
            max_safe_quantity = math.floor(max_safe_quantity * 10**quantity_precision) / 10**quantity_precision
            
            quantity = min(min_quantity, max_safe_quantity)
            
            if quantity * price > balance * 0.5:
                logger.warning("Position would use more than 50% of balance - reducing size")
                quantity = math.floor((balance * 0.5 / price) * 10**quantity_precision) / 10**quantity_precision
            
            if quantity <= 0:
                logger.error("Balance too low to open even minimum position")
                return 0
                
        logger.info(f"Calculated position size: {quantity} units at {price} per unit")
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
        """Adjust take profit for trailing take profit if needed"""
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
            new_tp = current_price * (1 - TRAILING_TAKE_PROFIT_PCT)
            # Only move take profit up, never down
            current_tp = self.calculate_take_profit(symbol, side, entry_price)
            if current_tp and new_tp <= current_tp:
                logger.debug(f"Not adjusting trailing take profit: current ({current_tp}) > calculated ({new_tp})")
                return None
        else:  # Short position
            new_tp = current_price * (1 + TRAILING_TAKE_PROFIT_PCT)
            # Only move take profit down, never up
            current_tp = self.calculate_take_profit(symbol, side, entry_price)
            if current_tp and new_tp >= current_tp:
                logger.debug(f"Not adjusting trailing take profit: current ({current_tp}) < calculated ({new_tp})")
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_tp = round(new_tp, price_precision)
            
        logger.info(f"Adjusted trailing take profit to {new_tp} ({TRAILING_TAKE_PROFIT_PCT*100}%)")
        logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {current_tp} -> {new_tp}")
        return new_tp
        
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
        # Base risk level from config
        base_risk = RISK_PER_TRADE
        
        # Apply position size multiplier
        dynamic_risk = base_risk * self.position_size_multiplier
        
        # Clamp to reasonable range (0.01-0.10)
        return max(0.01, min(0.10, dynamic_risk))
        
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

    def test_position_sizing(self, symbol='RAYSOLUSDT'):
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
        
        # Calculate base risk amount without position sizing
        base_risk_amount = balance * RISK_PER_TRADE
        
        # Calculate adjusted risk with position sizing
        adjusted_risk_amount = base_risk_amount * self.position_size_multiplier
        
        # Calculate theoretical position sizes
        base_position_size = base_risk_amount / current_price
        adjusted_position_size = adjusted_risk_amount / current_price
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'account_balance': balance,
            'risk_per_trade': RISK_PER_TRADE,
            'base_risk_amount': base_risk_amount,
            'position_size_multiplier': self.position_size_multiplier,
            'adjusted_risk_amount': adjusted_risk_amount,
            'base_position_size': base_position_size,
            'adjusted_position_size': adjusted_position_size,
            'market_condition': self.current_market_condition
        }


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