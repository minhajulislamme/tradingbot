import logging
import numpy as np
import pandas as pd
import ta
import math
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SupertrendIndicator:
    """Supertrend indicator implementation for faster trend detection"""
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        
    def calculate(self, df):
        """Calculate Supertrend indicator"""
        # Calculate ATR
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=self.period
        )
        
        # Calculate basic upper and lower bands
        df['basic_upper'] = (df['high'] + df['low']) / 2 + (self.multiplier * df['atr'])
        df['basic_lower'] = (df['high'] + df['low']) / 2 - (self.multiplier * df['atr'])
        
        # Initialize Supertrend columns
        df['supertrend'] = np.nan
        df['supertrend_direction'] = np.nan
        df['final_upper'] = np.nan
        df['final_lower'] = np.nan
        
        # Calculate final upper and lower bands
        for i in range(self.period, len(df)):
            if i == self.period:
                # Using .loc to properly set values
                df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                
                # Initial trend direction
                if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                else:
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
            else:
                # Calculate upper band
                if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
                    df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                    df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                
                # Calculate lower band
                if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
                    df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                    df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                else:
                    df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                
                # Calculate Supertrend value
                if (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                    df['close'].iloc[i] <= df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                elif (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                      df['close'].iloc[i] > df['final_upper'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] >= df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                      df['close'].iloc[i] < df['final_lower'].iloc[i]):
                    df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                    df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
        
        return df

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.risk_manager = None
        # Add cache related attributes
        self._cache = {}
        self._max_cache_entries = 10  # Limit cache size
        self._cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        self._last_kline_time = None
        self._cached_dataframe = None
        
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Ensure dataframe is sorted by time
        df = df.sort_values('open_time', ascending=True).reset_index(drop=True)
        
        return df
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for the strategy"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for {self.strategy_name} strategy")
    
    def get_signal(self, klines):
        """
        Should be implemented by subclasses.
        Returns 'BUY', 'SELL', or None.
        """
        raise NotImplementedError("Each strategy must implement get_signal method")


class RaysolDynamicGridStrategy(TradingStrategy):
    """
    Enhanced Dynamic RAYSOL Grid Trading Strategy that adapts to market trends
    and different market conditions (bullish, bearish, and sideways).
    
    Features:
    - Dynamic position sizing based on volatility and account equity
    - Adaptive grid spacing based on market volatility
    - Asymmetric grids biased toward the trend direction
    - Automatic grid reset when price moves outside range
    - Cool-off period after consecutive losses
    - Supertrend indicator for faster trend detection
    - VWAP for sideways markets
    - Volume-weighted RSI for better signals
    - Bollinger Band squeeze detection for breakouts
    - Fibonacci level integration for support/resistance
    - Enhanced momentum filtering and multi-indicator confirmation
    - Sophisticated reversal detection
    """
    def __init__(self, 
                 grid_levels=5, 
                 grid_spacing_pct=1.2,
                 trend_ema_fast=8,
                 trend_ema_slow=21,
                 volatility_lookback=20,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30,
                 volume_ma_period=20,
                 adx_period=14,
                 adx_threshold=25,
                 sideways_threshold=15,
                 # RAYSOL-specific parameters
                 volatility_multiplier=1.1,
                 trend_condition_multiplier=1.3,
                 min_grid_spacing=0.6,
                 max_grid_spacing=3.5,
                 # New parameters for enhanced features
                 supertrend_period=10,
                 supertrend_multiplier=3.0,
                 fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786],
                 squeeze_threshold=0.5,
                 cooloff_period=3,
                 max_consecutive_losses=2):
        
        super().__init__('RaysolDynamicGridStrategy')
        # Base parameters
        self.grid_levels = grid_levels
        self.grid_spacing_pct = grid_spacing_pct
        self.trend_ema_fast = trend_ema_fast
        self.trend_ema_slow = trend_ema_slow
        self.volatility_lookback = volatility_lookback
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.volume_ma_period = volume_ma_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.sideways_threshold = sideways_threshold
        
        # RAYSOL-specific parameters
        self.volatility_multiplier = volatility_multiplier
        self.trend_condition_multiplier = trend_condition_multiplier
        self.min_grid_spacing = min_grid_spacing
        self.max_grid_spacing = max_grid_spacing
        
        # Enhanced feature parameters
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.fibonacci_levels = fibonacci_levels
        self.squeeze_threshold = squeeze_threshold
        self.cooloff_period = cooloff_period
        self.max_consecutive_losses = max_consecutive_losses
        
        # State variables
        self.grids = None
        self.current_trend = None
        self.current_market_condition = None
        self.last_grid_update = None
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        self.position_size_pct = 1.0  # Default position size percentage
        
        # Cached indicators to avoid recalculation
        self._last_kline_time = None
        self._cached_dataframe = None
        self.supertrend_indicator = SupertrendIndicator(
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier
        )
        
    def prepare_data(self, klines):
        """
        Convert raw klines to a DataFrame with OHLCV data
        Overrides base method to implement enhanced caching for performance
        """
        # Generate a cache key based on first and last kline timestamps
        cache_key = None
        if len(klines) > 0:
            cache_key = f"{klines[0][0]}_{klines[-1][0]}"
        
        # Check if we can use cached data
        if cache_key:
            current_time = int(datetime.now().timestamp())
            
            # Clean up expired cache entries periodically
            if random.random() < 0.05:  # 5% chance to clean on each call
                expired_keys = []
                for k, v in self._cache.items():
                    if current_time - v.get('time', 0) > self._cache_expiry:
                        expired_keys.append(k)
                for k in expired_keys:
                    del self._cache[k]
                    logger.debug(f"Removed expired cache entry: {k}")
            
            # Look for cache entry
            if cache_key in self._cache:
                cache_entry = self._cache[cache_key]
                cache_time = cache_entry.get('time', 0)
                
                # Check if cache is still valid (not expired)
                if current_time - cache_time < self._cache_expiry:
                    logger.debug(f"Using cached data for {cache_key}")
                    return cache_entry['data']
        
        # Fall back to simple cache check if complex caching fails
        if len(klines) > 0 and self._last_kline_time == klines[-1][0]:
            if self._cached_dataframe is not None:
                return self._cached_dataframe
            
        # Otherwise prepare data normally
        df = super().prepare_data(klines)
        
        # Cache the result
        if len(klines) > 0:
            # Simple caching (backward compatible)
            self._last_kline_time = klines[-1][0]
            self._cached_dataframe = df
            
            # Enhanced caching with expiry and size management
            if cache_key:
                # Manage cache size - remove oldest entry if needed
                if len(self._cache) >= self._max_cache_entries:
                    oldest_key = min(self._cache.keys(), 
                                    key=lambda k: self._cache[k].get('time', 0))
                    del self._cache[oldest_key]
                    logger.debug(f"Cache full, removed oldest entry {oldest_key}")
                
                # Store in cache with timestamp
                self._cache[cache_key] = {
                    'data': df,
                    'time': current_time
                }
                logger.debug(f"Cached data for {cache_key}")
            
        return df
    
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame with enhanced features"""
        # Trend indicators
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], 
                                               window=self.trend_ema_slow)
        
        # Add Supertrend indicator for faster trend detection
        df = self.supertrend_indicator.calculate(df)
        df['trend'] = np.where(df['supertrend_direction'] == 1, 'UPTREND', 'DOWNTREND')
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # Volume indicators first to avoid duplicate calculation
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], 
                                                window=self.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Volume-weighted RSI (using the volume_ratio calculated above)
        df['volume_weighted_rsi'] = df['rsi'] * df['volume_ratio']
        
        # Volatility indicators
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], 
                                                    df['close'], 
                                                    window=self.volatility_lookback)
        df['atr_pct'] = df['atr'] / df['close'] * 100
        
        # ADX for trend strength
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], 
                                             window=self.adx_period)
        df['adx'] = adx_indicator.adx()
        df['di_plus'] = adx_indicator.adx_pos()
        df['di_minus'] = adx_indicator.adx_neg()
        
        # Bollinger Bands
        indicator_bb = ta.volatility.BollingerBands(df['close'], 
                                                   window=20, 
                                                   window_dev=2)
        df['bb_upper'] = indicator_bb.bollinger_hband()
        df['bb_middle'] = indicator_bb.bollinger_mavg()
        df['bb_lower'] = indicator_bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Bollinger Band Squeeze detection
        df['bb_squeeze'] = df['bb_width'] < self.squeeze_threshold
        
        # MACD for additional trend confirmation
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        df['macd_crossover'] = np.where(
            (df['macd'].shift(1) < df['macd_signal'].shift(1)) & 
            (df['macd'] > df['macd_signal']), 
            1, np.where(
                (df['macd'].shift(1) > df['macd_signal'].shift(1)) & 
                (df['macd'] < df['macd_signal']), 
                -1, 0
            )
        )
        
        # VWAP (Volume Weighted Average Price) - calculated per day
        df['vwap'] = self.calculate_vwap(df)
        
        # Calculate Fibonacci levels based on recent swing highs and lows
        self.calculate_fibonacci_levels(df)
        
        # Market condition classification with improved detection
        df['market_condition'] = self.classify_market_condition(df)
        
        # Reversal detection indicators
        df['potential_reversal'] = self.detect_reversal_patterns(df)
        
        return df
    
    def calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        # Get date component of timestamp for grouping
        df['date'] = df['open_time'].dt.date
        
        # Calculate VWAP for each day
        vwap = pd.Series(index=df.index)
        for date, group in df.groupby('date'):
            # Calculate cumulative sum of price * volume
            cum_vol_price = (group['close'] * group['volume']).cumsum()
            # Calculate cumulative sum of volume
            cum_vol = group['volume'].cumsum()
            # Calculate VWAP
            daily_vwap = cum_vol_price / cum_vol
            # Add to result series
            vwap.loc[group.index] = daily_vwap
            
        return vwap
    
    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement/extension levels for support and resistance"""
        if len(df) < 20:  # Need sufficient data
            return
        
        # Find recent swing high and low points
        window = min(100, len(df) - 1)  # Look back window
        price_data = df['close'].iloc[-window:]
        
        # Identify swing high and low
        swing_high = price_data.max()
        swing_low = price_data.min()
        
        # Reset fibonacci levels
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # Calculate levels based on trend
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        
        if current_trend == 'UPTREND':
            # In uptrend, calculate fib retracements from low to high for support
            for fib in self.fibonacci_levels:
                level = swing_low + (swing_high - swing_low) * (1 - fib)
                if level < current_price:
                    self.fib_support_levels.append(level)
                else:
                    self.fib_resistance_levels.append(level)
                    
            # Add extension levels for resistance
            for ext in [1.272, 1.618, 2.0]:
                level = swing_low + (swing_high - swing_low) * ext
                self.fib_resistance_levels.append(level)
                
        else:  # DOWNTREND
            # In downtrend, calculate fib retracements from high to low for resistance
            for fib in self.fibonacci_levels:
                level = swing_high - (swing_high - swing_low) * fib
                if level > current_price:
                    self.fib_resistance_levels.append(level)
                else:
                    self.fib_support_levels.append(level)
                    
            # Add extension levels for support
            for ext in [1.272, 1.618, 2.0]:
                level = swing_high - (swing_high - swing_low) * ext
                self.fib_support_levels.append(level)
        
        # Sort the levels
        self.fib_support_levels.sort(reverse=True)  # Descending
        self.fib_resistance_levels.sort()  # Ascending
    
    def detect_reversal_patterns(self, df):
        """
        Enhanced reversal pattern detection
        Returns 1 for potential bullish reversal, -1 for bearish reversal, 0 for no reversal
        """
        if len(df) < 5:
            return pd.Series(0, index=df.index)
            
        # Initialize result series
        reversal = pd.Series(0, index=df.index)
        
        for i in range(4, len(df)):
            # Get relevant rows for pattern detection
            curr = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            prev3 = df.iloc[i-3]
            
            # Check for bullish reversal patterns
            bullish_reversal = False
            
            # Bullish engulfing
            if (curr['close'] > curr['open'] and  # Current candle is bullish
                prev1['close'] < prev1['open'] and  # Previous candle is bearish
                curr['close'] > prev1['open'] and  # Current close above prev open
                curr['open'] < prev1['close']):  # Current open below prev close
                bullish_reversal = True
                
            # Hammer pattern (bullish)
            elif (curr['low'] < curr['open'] and
                  curr['low'] < curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) < 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (min(curr['open'], curr['close']) - curr['low']) > 
                  (curr['high'] - max(curr['open'], curr['close'])) * 3):
                bullish_reversal = True
                
            # RSI divergence (bullish)
            elif (prev2['low'] > prev1['low'] and  # Price making lower low
                  prev2['rsi'] < prev1['rsi'] and  # RSI making higher low
                  curr['supertrend_direction'] == 1):  # Confirmed by Supertrend
                bullish_reversal = True
                
            # Check for bearish reversal patterns
            bearish_reversal = False
            
            # Bearish engulfing
            if (curr['close'] < curr['open'] and  # Current candle is bearish
                prev1['close'] > prev1['open'] and  # Previous candle is bullish
                curr['close'] < prev1['open'] and  # Current close below prev open
                curr['open'] > prev1['close']):  # Current open above prev close
                bearish_reversal = True
                
            # Shooting star (bearish)
            elif (curr['high'] > curr['open'] and
                  curr['high'] > curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 3):
                bearish_reversal = True
                
            # RSI divergence (bearish)
            elif (prev2['high'] < prev1['high'] and  # Price making higher high
                  prev2['rsi'] > prev1['rsi'] and  # RSI making lower high
                  curr['supertrend_direction'] == -1):  # Confirmed by Supertrend
                bearish_reversal = True
            
            # Set reversal value
            if bullish_reversal:
                reversal.iloc[i] = 1
            elif bearish_reversal:
                reversal.iloc[i] = -1
                
        return reversal
    
    def classify_market_condition(self, df):
        """
        Enhanced market condition classification with better state transitions and stability
        """
        conditions = []
        lookback_period = 10  # Use more historical data for smoother transitions
        current_condition = None
        condition_streak = 0  # Track how long we've been in a condition
        
        for i in range(len(df)):
            if i < self.adx_period:
                conditions.append('SIDEWAYS')  # Default for initial rows
                continue
                
            # Get relevant indicators
            adx = df['adx'].iloc[i]
            di_plus = df['di_plus'].iloc[i]
            di_minus = df['di_minus'].iloc[i]
            rsi = df['rsi'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            supertrend_dir = df['supertrend_direction'].iloc[i] if i >= self.supertrend_period else 0
            macd_crossover = df['macd_crossover'].iloc[i] if 'macd_crossover' in df else 0
            
            # Get average ADX for more stability
            lookback = min(lookback_period, i)
            avg_adx = df['adx'].iloc[i-lookback:i+1].mean() if i >= lookback else adx
            
            # Check for squeeze condition (low volatility, potential breakout)
            is_squeeze = bb_width < self.squeeze_threshold
            
            # Calculate the strength of each potential market condition
            bullish_strength = 0
            bearish_strength = 0
            sideways_strength = 0
            squeeze_strength = 0
            
            # ADX trending strength (0-100)
            trend_strength = min(100, adx * 2)  # Normalize to 0-100 scale
            
            # Directional bias (-100 to +100, negative = bearish, positive = bullish)
            if di_plus + di_minus > 0:  # Avoid division by zero
                directional_bias = 100 * (di_plus - di_minus) / (di_plus + di_minus)
            else:
                directional_bias = 0
                
            # Supertrend confirmation
            supertrend_bias = 100 if supertrend_dir > 0 else -100 if supertrend_dir < 0 else 0
            
            # RSI bias (normalized to -100 to +100)
            rsi_bias = (rsi - 50) * 2  # 0 = neutral, +100 = extremely bullish, -100 = extremely bearish
            
            # Combine for final bias score
            bias_score = (directional_bias + supertrend_bias + rsi_bias) / 3
            
            # Calculate condition strengths
            if bias_score > 30 and trend_strength > 50:  # Strong bullish
                bullish_strength = trend_strength
                
                # Determine if extreme bullish
                if bias_score > 60 and trend_strength > 70 and adx > self.adx_threshold * 1.5:
                    bullish_strength += 30  # Extra boost for extreme bullish
            
            if bias_score < -30 and trend_strength > 50:  # Strong bearish
                bearish_strength = trend_strength
                
                # Determine if extreme bearish
                if bias_score < -60 and trend_strength > 70 and adx > self.adx_threshold * 1.5:
                    bearish_strength += 30  # Extra boost for extreme bearish
            
            if trend_strength < 40:  # Low trend strength = sideways
                sideways_strength = 100 - trend_strength
                
            if is_squeeze:
                squeeze_strength = 100 if bb_width < self.squeeze_threshold * 0.7 else 70
            
            # Prevent rapid condition changes by requiring larger threshold to change states
            new_condition = None
            
            # Determine the new condition based on highest strength
            if max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == bullish_strength:
                if bullish_strength > 80:
                    new_condition = 'EXTREME_BULLISH'
                else:
                    new_condition = 'BULLISH'
            elif max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == bearish_strength:
                if bearish_strength > 80:
                    new_condition = 'EXTREME_BEARISH'
                else:
                    new_condition = 'BEARISH'
            elif max(bullish_strength, bearish_strength, sideways_strength, squeeze_strength) == squeeze_strength:
                new_condition = 'SQUEEZE'
            else:
                new_condition = 'SIDEWAYS'
            
            # Apply hysteresis to avoid rapid condition changes
            # Only change condition if new one is persistent or very strong
            if i > 0 and current_condition:
                previous_condition = conditions[i-1]
                
                # Keep current condition unless we have a strong signal to change
                # This prevents whipsaws between market states
                if previous_condition != new_condition:
                    # For a change to occur, the new condition needs to be significantly stronger
                    change_threshold = 20 if condition_streak < 3 else 10
                    
                    max_current_strength = 0
                    if previous_condition == 'BULLISH' or previous_condition == 'EXTREME_BULLISH':
                        max_current_strength = bullish_strength
                    elif previous_condition == 'BEARISH' or previous_condition == 'EXTREME_BEARISH':
                        max_current_strength = bearish_strength
                    elif previous_condition == 'SQUEEZE':
                        max_current_strength = squeeze_strength
                    else:  # SIDEWAYS
                        max_current_strength = sideways_strength
                    
                    max_new_strength = 0
                    if new_condition == 'BULLISH' or new_condition == 'EXTREME_BULLISH':
                        max_new_strength = bullish_strength
                    elif new_condition == 'BEARISH' or new_condition == 'EXTREME_BEARISH':
                        max_new_strength = bearish_strength
                    elif new_condition == 'SQUEEZE':
                        max_new_strength = squeeze_strength
                    else:  # SIDEWAYS
                        max_new_strength = sideways_strength
                    
                    # Only change if the new condition is significantly stronger
                    if max_new_strength > max_current_strength + change_threshold:
                        conditions.append(new_condition)
                        current_condition = new_condition
                        condition_streak = 1
                    else:
                        conditions.append(previous_condition)
                        current_condition = previous_condition
                        condition_streak += 1
                else:
                    conditions.append(previous_condition)
                    current_condition = previous_condition
                    condition_streak += 1
            else:
                conditions.append(new_condition)
                current_condition = new_condition
                condition_streak = 1
        
        return pd.Series(conditions, index=df.index)
    
    def calculate_dynamic_position_size(self, df, base_position=1.0):
        """
        Calculate dynamic position size based on volatility and market condition
        
        Args:
            df: DataFrame with indicators
            base_position: Base position size (1.0 = 100% of allowed position)
            
        Returns:
            float: Position size multiplier (e.g., 0.8 means 80% of base position)
        """
        latest = df.iloc[-1]
        
        # Get ATR as volatility measure
        atr_pct = latest['atr_pct']
        avg_atr_pct = df['atr_pct'].tail(20).mean()
        
        # Base position sizing on volatility relative to average
        if atr_pct > avg_atr_pct * 1.5:
            # High volatility - reduce position size
            volatility_factor = 0.7
        elif atr_pct < avg_atr_pct * 0.7:
            # Low volatility - increase position size slightly
            volatility_factor = 1.1
        else:
            # Normal volatility
            volatility_factor = 1.0
            
        # Adjust based on market condition
        market_condition = latest['market_condition']
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            # Extreme trend - reduce position size for safety
            condition_factor = 0.75
        elif market_condition in ['BULLISH', 'BEARISH']:
            # Clear trend - standard position
            condition_factor = 1.0
        elif market_condition == 'SQUEEZE':
            # Squeeze condition - smaller position for breakout
            condition_factor = 0.8
        else:  # SIDEWAYS
            # Sideways market - slightly smaller position
            condition_factor = 0.9
            
        # Calculate final position size
        position_size = base_position * volatility_factor * condition_factor
        
        # Cap the position size
        return min(max(position_size, 0.5), 1.2)  # Between 50% and 120%
    
    def calculate_dynamic_grid_levels(self, df):
        """
        Adjust grid levels count based on volatility
        """
        latest = df.iloc[-1]
        
        # Base decision on ATR percentage and BB width
        atr_pct = latest['atr_pct']
        bb_width = latest['bb_width']
        
        # Fewer levels in high volatility
        if atr_pct > 3.0 or bb_width > 0.08:
            return max(3, self.grid_levels - 2)
        # More levels in low volatility
        elif atr_pct < 1.0 or bb_width < 0.03:
            return min(7, self.grid_levels + 2)
        # Default to configured level
        else:
            return self.grid_levels
            
    def calculate_grid_spacing(self, df):
        """
        Enhanced dynamic grid spacing calculation based on multiple factors
        """
        try:
            # Get the latest row
            latest = df.iloc[-1]
            
            # Base grid spacing on ATR percentage
            base_spacing = latest['atr_pct'] * self.volatility_multiplier
            
            # Adjust based on Bollinger Band width (volatility scaling)
            bb_multiplier = min(max(latest['bb_width'] * 6, 0.5), 3.5)
            
            # Adjust based on market condition
            market_condition = latest['market_condition']
            if market_condition == 'SIDEWAYS':
                # Tighter grid spacing in sideways markets
                condition_multiplier = 0.8
            elif market_condition in ['BULLISH', 'BEARISH']:
                # Wider grid spacing in trending markets
                condition_multiplier = self.trend_condition_multiplier
            elif market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
                # Even wider spacing in extreme trends
                condition_multiplier = self.trend_condition_multiplier * 1.5
            elif market_condition == 'SQUEEZE':
                # Prepare for potential breakout with wider spacing
                condition_multiplier = self.trend_condition_multiplier * 1.2
            else:
                condition_multiplier = 1.0
            
            # Calculate final grid spacing
            dynamic_spacing = base_spacing * bb_multiplier * condition_multiplier
            
            # Ensure minimum and maximum spacing
            return min(max(dynamic_spacing, self.min_grid_spacing), self.max_grid_spacing)
            
        except Exception as e:
            logger.error(f"Error calculating grid spacing: {e}")
            # Return default spacing in case of error
            return self.grid_spacing_pct
    
    def calculate_grid_bias(self, df):
        """
        Calculate asymmetric grid bias based on market conditions
        Returns percentage of grid levels that should be above current price
        """
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Extreme bias in extreme market conditions
        if market_condition == 'EXTREME_BULLISH':
            return 0.8  # 80% of levels above (strong buy bias)
        elif market_condition == 'EXTREME_BEARISH':
            return 0.2  # 20% of levels above (strong sell bias)
        # Strong bias in trending markets
        elif market_condition == 'BULLISH':
            return 0.7  # 70% of levels above
        elif market_condition == 'BEARISH':
            return 0.3  # 30% of levels above
        # Neutral bias in sideways or squeeze markets
        else:
            return 0.5  # 50% of levels above/below (neutral)
    
    def generate_grids(self, df):
        """
        Generate enhanced dynamic grid levels with asymmetric distribution
        and Fibonacci integration
        """
        # Get latest price and indicators
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        market_condition = latest['market_condition']
        
        # Update risk manager with market condition if available
        if self.risk_manager and market_condition:
            self.risk_manager.set_market_condition(market_condition)
            self.risk_manager.update_position_sizing(self.calculate_dynamic_position_size(df))
            logger.info(f"Updated risk manager with market condition: {market_condition}")
        
        # Determine asymmetric grid bias based on market condition
        grid_bias = self.calculate_grid_bias(df)
        
        # Calculate dynamic grid spacing based on volatility
        dynamic_spacing = self.calculate_grid_spacing(df)
        
        # Adjust grid levels based on volatility
        dynamic_grid_levels = self.calculate_dynamic_grid_levels(df)
        
        # Generate grid levels
        grid_levels = []
        
        # Calculate number of levels above and below current price
        levels_above = int(dynamic_grid_levels * grid_bias)
        levels_below = dynamic_grid_levels - levels_above
        
        # Generate grid levels below current price
        for i in range(1, levels_below + 1):
            # Base grid price
            base_grid_price = current_price * (1 - (dynamic_spacing / 100) * i)
            
            # Check if any Fibonacci support level is nearby
            grid_price = base_grid_price
            for support_level in self.fib_support_levels:
                # If within 1% of a Fibonacci level, snap to it
                if abs(support_level - base_grid_price) / base_grid_price < 0.01:
                    grid_price = support_level
                    break
            
            # Grid levels below current price are buy levels
            grid_levels.append({
                'price': grid_price,
                'type': 'BUY',
                'status': 'ACTIVE',
                'created_at': latest['open_time']
            })
        
        # Generate grid levels above current price
        for i in range(1, levels_above + 1):
            # Base grid price
            base_grid_price = current_price * (1 + (dynamic_spacing / 100) * i)
            
            # Check if any Fibonacci resistance level is nearby
            grid_price = base_grid_price
            for resistance_level in self.fib_resistance_levels:
                # If within 1% of a Fibonacci level, snap to it
                if abs(resistance_level - base_grid_price) / base_grid_price < 0.01:
                    grid_price = resistance_level
                    break
            
            # Grid levels above current price are sell levels
            grid_levels.append({
                'price': grid_price,
                'type': 'SELL',
                'status': 'ACTIVE',
                'created_at': latest['open_time']
            })
        
        # Sort grid levels by price
        grid_levels.sort(key=lambda x: x['price'])
        
        return grid_levels
    
    def should_update_grids(self, df):
        """Enhanced grid reset logic"""
        if self.grids is None or len(self.grids) == 0:
            return True
            
        latest = df.iloc[-1]
        current_trend = latest['trend']
        current_market_condition = latest['market_condition']
        
        # Update risk manager with new market condition if it changed
        if self.risk_manager and self.current_market_condition != current_market_condition:
            self.risk_manager.set_market_condition(current_market_condition)
            logger.info(f"Updated risk manager with market condition: {current_market_condition}")
        
        # Update grids if trend or market condition changed significantly
        trend_change = (self.current_trend != current_trend)
        condition_change = (
            (self.current_market_condition in ['BULLISH', 'BEARISH'] and 
             current_market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH', 'SQUEEZE']) or
            (self.current_market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH'] and 
             current_market_condition in ['BULLISH', 'BEARISH', 'SQUEEZE']) or
            (self.current_market_condition == 'SQUEEZE' and 
             current_market_condition in ['BULLISH', 'BEARISH', 'EXTREME_BULLISH', 'EXTREME_BEARISH'])
        )
        
        if trend_change or condition_change:
            logger.info(f"Market conditions changed. Trend: {self.current_trend}->{current_trend}, "
                       f"Condition: {self.current_market_condition}->{current_market_condition}. "
                       f"Updating grids.")
            return True
            
        # Check if price moved significantly outside grid range (auto-reset)
        current_price = latest['close']
        min_grid = min(grid['price'] for grid in self.grids)
        max_grid = max(grid['price'] for grid in self.grids)
        
        # If price is outside grid range by more than 2%, update grids
        if current_price < min_grid * 0.98 or current_price > max_grid * 1.02:
            logger.info(f"Price moved outside grid range. Updating grids.")
            return True
            
        # Check if many grid levels have been triggered
        active_grids = [grid for grid in self.grids if grid['status'] == 'ACTIVE']
        if len(active_grids) < len(self.grids) * 0.3:  # Less than 30% active
            logger.info(f"Too many grid levels have been triggered. Refreshing grid.")
            return True
            
        return False
    
    def in_cooloff_period(self, current_time):
        """Check if we're in a cool-off period after losses"""
        if self.consecutive_losses >= self.max_consecutive_losses and self.last_loss_time:
            try:
                # Convert from pandas timestamp if needed
                if hasattr(self.last_loss_time, 'to_pydatetime'):
                    last_loss_time = self.last_loss_time.to_pydatetime()
                else:
                    last_loss_time = self.last_loss_time
                    
                # Convert current_time from pandas timestamp if needed
                if hasattr(current_time, 'to_pydatetime'):
                    current_time = current_time.to_pydatetime()
                
                # Check if we're still in the cool-off period
                cooloff_end_time = last_loss_time + timedelta(minutes=self.cooloff_period)
                return current_time < cooloff_end_time
            except Exception as e:
                logger.error(f"Error in cooloff period calculation: {e}")
                # Default to False if there's an error in comparison
                return False
            
        return False
    
    def get_v_reversal_signal(self, df):
        """
        Detect V-shaped reversals in extreme market conditions
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        potential_reversal = latest['potential_reversal']
        
        # Only look for reversals in extreme conditions
        if market_condition not in ['EXTREME_BEARISH', 'EXTREME_BULLISH']:
            return None
            
        # Return reversal signal if detected
        if potential_reversal == 1 and market_condition == 'EXTREME_BEARISH':
            logger.info("V-shaped bullish reversal detected in extreme bearish market")
            return 'BUY'
        elif potential_reversal == -1 and market_condition == 'EXTREME_BULLISH':
            logger.info("V-shaped bearish reversal detected in extreme bullish market")
            return 'SELL'
            
        return None
    
    def get_squeeze_breakout_signal(self, df):
        """
        Detect breakouts from low-volatility squeeze conditions
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        market_condition = latest['market_condition']
        
        # Only look for breakout if we're in or just exited a squeeze
        if market_condition != 'SQUEEZE' and prev['market_condition'] != 'SQUEEZE':
            return None
            
        # Volume spike indicates breakout
        if latest['volume_ratio'] > 1.5:
            # Direction of breakout
            if latest['close'] > latest['bb_upper']:
                return 'BUY'
            elif latest['close'] < latest['bb_lower']:
                return 'SELL'
                
        return None
    
    def get_multi_indicator_signal(self, df):
        """
        Get signals based on multi-indicator confirmation with stronger consolidated validation
        and enhanced filtering to reduce false positives
        """
        if len(df) < 10:  # Increased minimum data requirement for better context
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        prev3 = df.iloc[-3] if len(df) > 3 else None
        prev5 = df.iloc[-5] if len(df) > 5 else None
        
        # Count bullish and bearish signals with weighting for stronger indicators
        bullish_signals = 0
        bearish_signals = 0
        
        # Check for trend consistency over multiple candles (reduces false signals)
        is_consistent_trend = False
        if len(df) >= 5:
            # Check if the last 5 candles show consistent trend direction
            last_5_supertrend = [df.iloc[-i]['supertrend_direction'] for i in range(1, 6)]
            if all(direction == 1 for direction in last_5_supertrend):
                is_consistent_trend = True
                bullish_signals += 1  # Bonus for trend consistency
            elif all(direction == -1 for direction in last_5_supertrend):
                is_consistent_trend = True
                bearish_signals += 1  # Bonus for trend consistency
        
        # === PRIMARY INDICATORS (Higher weight) ===
        
        # Supertrend (stronger weight x2.5 for consistent trends)
        if latest['supertrend_direction'] == 1:
            bullish_signals += 2.5 if is_consistent_trend else 2.0
        else:
            bearish_signals += 2.5 if is_consistent_trend else 2.0
            
        # Price action relative to key levels with stronger confirmation
        if (latest['close'] > latest['ema_slow'] and 
            latest['close'] > latest['vwap'] and
            latest['close'] > latest['ema_fast']):  # Added requirement for all EMAs
            bullish_signals += 2.0  # Increased weight for stronger price action
        elif (latest['close'] < latest['ema_slow'] and 
              latest['close'] < latest['vwap'] and
              latest['close'] < latest['ema_fast']):  # Added requirement for all EMAs
            bearish_signals += 2.0  # Increased weight for stronger price action
        
        # Trend direction change (stronger weight)
        if prev['supertrend_direction'] == -1 and latest['supertrend_direction'] == 1:
            bullish_signals += 2  # Fresh bullish trend
        elif prev['supertrend_direction'] == 1 and latest['supertrend_direction'] == -1:
            bearish_signals += 2  # Fresh bearish trend
            
        # === SECONDARY INDICATORS ===
        
        # RSI & Volume-weighted RSI
        if latest['rsi'] < 30:
            bullish_signals += 1
        elif latest['rsi'] > 70:
            bearish_signals += 1
            
        # More extreme RSI values (stronger weight)
        if latest['rsi'] < 20:
            bullish_signals += 0.5  # Very oversold
        elif latest['rsi'] > 80:
            bearish_signals += 0.5  # Very overbought
            
        # Volume-weighted RSI
        if latest['volume_weighted_rsi'] < 25:
            bullish_signals += 1
        elif latest['volume_weighted_rsi'] > 75:
            bearish_signals += 1
            
        # MACD
        if latest['macd_crossover'] == 1:
            bullish_signals += 1
        elif latest['macd_crossover'] == -1:
            bearish_signals += 1
            
        # Volume confirmation
        if latest['volume_ratio'] > 1.5:
            # High volume adds weight to the dominant direction
            if bullish_signals > bearish_signals:
                bullish_signals += 1
            elif bearish_signals > bullish_signals:
                bearish_signals += 1
            
        # ADX trend strength
        if latest['adx'] > self.adx_threshold:
            if latest['di_plus'] > latest['di_minus']:
                bullish_signals += 1
            else:
                bearish_signals += 1
                
        # Strong ADX (higher weight)
        if latest['adx'] > self.adx_threshold * 1.5:
            if latest['di_plus'] > latest['di_minus']:
                bullish_signals += 0.5  # Strong trend confirmation
            else:
                bearish_signals += 0.5  # Strong trend confirmation
                
        # === SUPPORT/RESISTANCE CONFIRMATIONS ===
                
        # Price relative to VWAP
        if latest['close'] < latest['vwap'] * 0.98:
            bullish_signals += 0.5  # Potential support
        elif latest['close'] > latest['vwap'] * 1.02:
            bearish_signals += 0.5  # Potential resistance
            
        # Bollinger Bands
        if latest['close'] < latest['bb_lower'] * 1.01:
            bullish_signals += 1  # At support
        elif latest['close'] > latest['bb_upper'] * 0.99:
            bearish_signals += 1  # At resistance
        
        # Fibonacci levels (higher weight)
        close_to_fib_support = any(abs(latest['close'] - level) / latest['close'] < 0.005 for level in self.fib_support_levels)
        close_to_fib_resistance = any(abs(latest['close'] - level) / latest['close'] < 0.005 for level in self.fib_resistance_levels)
        
        if close_to_fib_support:
            bullish_signals += 1.5  # Strong support
        if close_to_fib_resistance:
            bearish_signals += 1.5  # Strong resistance
            
        # === MARKET CONDITION ADJUSTMENT ===
        
        # Adjust signal thresholds based on market condition - higher thresholds to reduce false signals
        market_condition = latest['market_condition']
        bull_threshold = 7.0  # Increased base threshold for stronger confirmation
        bear_threshold = 7.0  # Increased base threshold for stronger confirmation
        
        # Volume requirement - ensure we have enough volume to validate signals
        min_volume_ratio = 1.2  # Minimum volume needed relative to average
        has_sufficient_volume = latest['volume_ratio'] >= min_volume_ratio
        
        # Add requirement for sufficient volume in trending markets
        if not has_sufficient_volume:
            bull_threshold += 1.0  # Make it harder to trigger without enough volume
            bear_threshold += 1.0  # Make it harder to trigger without enough volume
        
        if market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            bull_threshold -= 1.0  # Easier to trigger buy in bullish market, but still higher than before
            bear_threshold += 1.5  # Much harder to trigger sell in bullish market
        elif market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            bull_threshold += 1.5  # Much harder to trigger buy in bearish market
            bear_threshold -= 1.0  # Easier to trigger sell in bearish market, but still higher than before
        elif market_condition == 'SQUEEZE':
            # In squeeze, wait for much stronger confirmation
            bull_threshold += 1.5
            bear_threshold += 1.5
            
        # === FINAL SIGNAL DETERMINATION ===
        
        # Calculate signal strength as a percentage of max possible
        max_possible_score = 20  # Increased from 15 to account for new signal weights
        bull_strength = bullish_signals / max_possible_score * 100
        bear_strength = bearish_signals / max_possible_score * 100
        
        # Check for trend continuation or reversal - add strength for continuation
        has_momentum = False
        if len(df) >= 3:
            # Check previous signal direction to add momentum confirmation
            # This helps filter out minor counter-trend moves
            prev_candle_momentum = prev['close'] - prev['open']
            current_candle_momentum = latest['close'] - latest['open']
            
            # If momentum is in the same direction for 2+ candles, it's stronger
            if (prev_candle_momentum > 0 and current_candle_momentum > 0 and bullish_signals > bearish_signals):
                bullish_signals += 1.0  # Bonus for momentum continuation
                has_momentum = True
            elif (prev_candle_momentum < 0 and current_candle_momentum < 0 and bearish_signals > bullish_signals):
                bearish_signals += 1.0  # Bonus for momentum continuation
                has_momentum = True
        
        logger.debug(f"Multi-indicator signals - Bullish: {bullish_signals:.1f} ({bull_strength:.1f}%), " +
                    f"Bearish: {bearish_signals:.1f} ({bear_strength:.1f}%)")
        
        # Return signal based on very strong confirmation from multiple indicators
        # Added more stringent requirements with 2x difference instead of 1.5x
        if (bullish_signals >= bull_threshold and 
            bullish_signals > bearish_signals * 2.0 and  # Stronger dominance required
            (has_momentum or bull_strength > 60)):  # Either has momentum or strong overall score
            logger.info(f"Strong bullish confirmation: {bullish_signals:.1f} signals ({bull_strength:.1f}%)")
            return 'BUY'
            
        if (bearish_signals >= bear_threshold and 
            bearish_signals > bullish_signals * 2.0 and  # Stronger dominance required
            (has_momentum or bear_strength > 60)):  # Either has momentum or strong overall score
            logger.info(f"Strong bearish confirmation: {bearish_signals:.1f} signals ({bear_strength:.1f}%)")
            return 'SELL'
            
        return None
    
    def get_grid_signal(self, df):
        """Enhanced grid signal with position sizing"""
        latest = df.iloc[-1]
        current_price = latest['close']
        current_time = latest['open_time']
        
        # Check for cool-off period after consecutive losses
        if self.in_cooloff_period(current_time):
            logger.info(f"In cool-off period after {self.consecutive_losses} consecutive losses. No grid signals.")
            return None
        
        # If no grids, generate them first
        if self.grids is None or len(self.grids) == 0 or self.should_update_grids(df):
            self.grids = self.generate_grids(df)
            self.current_trend = latest['trend']
            self.current_market_condition = latest['market_condition']
            self.last_grid_update = latest['open_time']
            logger.info(f"Generated new grids for {self.current_market_condition} market condition")
            return None  # No signal on grid initialization
        
        # Find the nearest grid levels
        buy_grids = [grid for grid in self.grids if grid['type'] == 'BUY' and grid['status'] == 'ACTIVE']
        sell_grids = [grid for grid in self.grids if grid['type'] == 'SELL' and grid['status'] == 'ACTIVE']
        
        # Find closest buy and sell grids
        closest_buy = None
        closest_sell = None
        
        if buy_grids:
            closest_buy = max(buy_grids, key=lambda x: x['price'])
            
        if sell_grids:
            closest_sell = min(sell_grids, key=lambda x: x['price'])
        
        # Determine signal based on price position relative to grids
        if closest_buy and current_price <= closest_buy['price'] * 1.001:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_buy['price']:
                    grid['status'] = 'TRIGGERED'
                    
            # Update position size for risk manager
            if self.risk_manager:
                position_size = self.calculate_dynamic_position_size(df)
                self.risk_manager.update_position_sizing(position_size)
                self.position_size_pct = position_size
            
            # Return BUY signal
            return 'BUY'
            
        elif closest_sell and current_price >= closest_sell['price'] * 0.999:
            # Mark this grid as triggered
            for grid in self.grids:
                if grid['price'] == closest_sell['price']:
                    grid['status'] = 'TRIGGERED'
                    
            # Update position size for risk manager
            if self.risk_manager:
                position_size = self.calculate_dynamic_position_size(df)
                self.risk_manager.update_position_sizing(position_size)
                self.position_size_pct = position_size
            
            # Return SELL signal
            return 'SELL'
            
        return None
    
    def get_sideways_signal(self, df):
        """Enhanced sideways market signal with VWAP integration"""
        latest = df.iloc[-1]
        
        # In sideways markets, use VWAP as a dynamic grid anchor
        
        # Buy near lower Bollinger Band with VWAP confirmation
        if latest['close'] < latest['bb_lower'] * 1.01 and latest['close'] < latest['vwap']:
            return 'BUY'
            
        # Sell near upper Bollinger Band with VWAP confirmation
        elif latest['close'] > latest['bb_upper'] * 0.99 and latest['close'] > latest['vwap']:
            return 'SELL'
            
        # Volume-weighted RSI signals in sideways markets
        elif latest['volume_weighted_rsi'] < 30:
            return 'BUY'
        elif latest['volume_weighted_rsi'] > 70:
            return 'SELL'
            
        return None
    
    def get_bullish_signal(self, df):
        """
        Enhanced signal for bullish market with aggressive trend thresholds
        """
        if len(df) < 3:
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_oversold = 25 if market_condition == 'EXTREME_BULLISH' else 35
            
            # More aggressive oversold conditions for BUY signals in bullish markets
            if latest['rsi'] < rsi_oversold:
                return 'BUY'
                
            # BUY on MACD crossover with volume confirmation
            if (prev['macd'] < prev['macd_signal'] and 
                latest['macd'] > latest['macd_signal'] and 
                latest['volume_ratio'] > 1.2):
                return 'BUY'
                
            # BUY on Supertrend direction change
            if prev['supertrend_direction'] == -1 and latest['supertrend_direction'] == 1:
                return 'BUY'
                
            # Sell only on extreme overbought conditions in bullish markets
            if (latest['rsi'] > 80 and 
                latest['close'] > latest['bb_upper'] * 1.01 and
                latest['close'] > latest['vwap'] * 1.03):
                return 'SELL'
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bullish_signal: {e}")
            return None
    
    def get_bearish_signal(self, df):
        """
        Enhanced signal for bearish market with aggressive trend thresholds
        """
        if len(df) < 3:
            return None
            
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            market_condition = latest['market_condition']
            
            # Adjust RSI thresholds based on market condition
            rsi_overbought = 75 if market_condition == 'EXTREME_BEARISH' else 65
            
            # More aggressive overbought conditions for SELL signals in bearish markets
            if latest['rsi'] > rsi_overbought:
                return 'SELL'
                
            # SELL on MACD crossover with volume confirmation
            if (prev['macd'] > prev['macd_signal'] and 
                latest['macd'] < latest['macd_signal'] and 
                latest['volume_ratio'] > 1.2):
                return 'SELL'
                
            # SELL on Supertrend direction change
            if prev['supertrend_direction'] == 1 and latest['supertrend_direction'] == -1:
                return 'SELL'
                
            # Buy only on extreme oversold conditions in bearish markets
            if (latest['rsi'] < 20 and 
                latest['close'] < latest['bb_lower'] * 0.99 and
                latest['close'] < latest['vwap'] * 0.97):
                return 'BUY'
                
            return None
            
        except Exception as e:
            logger.error(f"Error in get_bearish_signal: {e}")
            return None
            
    def update_trade_result(self, was_profitable):
        """
        Update consecutive losses counter for cool-off period calculation
        
        Args:
            was_profitable: Boolean indicating if the last trade was profitable
        """
        if was_profitable:
            # Reset consecutive losses on profitable trade
            self.consecutive_losses = 0
            self.last_loss_time = None
            logger.info("Profitable trade - reset consecutive losses counter")
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            logger.info(f"Loss recorded - consecutive losses: {self.consecutive_losses}")
            
            # Check if we need to enter cool-off period
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.info(f"Entering cool-off period for {self.cooloff_period} candles")
    
    def get_extreme_market_signal(self, df):
        """
        Specialized signal generation for extreme market conditions
        """
        if len(df) < 3:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # Only process if we're in extreme market conditions
        if market_condition not in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            return None
            
        # In extreme bullish market
        if market_condition == 'EXTREME_BULLISH':
            # Look for BUYING opportunities on small dips
            if (latest['close'] < latest['vwap'] and 
                latest['supertrend_direction'] == 1 and
                latest['rsi'] < 40):
                return 'BUY'
                
        # In extreme bearish market
        elif market_condition == 'EXTREME_BEARISH':
            # Look for SELLING opportunities on small rallies
            if (latest['close'] > latest['vwap'] and 
                latest['supertrend_direction'] == -1 and
                latest['rsi'] > 60):
                return 'SELL'
                
        return None
    
    def get_signal(self, klines):
        """
        Enhanced signal generation integrating all the new features
        """
        # Prepare and add indicators to the data
        df = self.prepare_data(klines)
        df = self.add_indicators(df)
        
        if len(df) < self.trend_ema_slow + 5:
            # Not enough data to generate reliable signals
            return None
        
        # Get latest data
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        current_time = latest['open_time']
        
        # Update risk manager with current market condition
        if self.risk_manager:
            self.risk_manager.set_market_condition(market_condition)
        
        # Check for cool-off period first
        if self.in_cooloff_period(current_time):
            logger.info(f"In cool-off period after {self.consecutive_losses} consecutive losses. No trading signals.")
            return None
        
        # 1. Check for V-shaped reversals in extreme market conditions
        reversal_signal = self.get_v_reversal_signal(df)
        if reversal_signal:
            logger.info(f"V-reversal detected in {market_condition} market. Signal: {reversal_signal}")
            return reversal_signal
        
        # 2. Check for breakouts from squeeze conditions
        squeeze_signal = self.get_squeeze_breakout_signal(df)
        if squeeze_signal:
            logger.info(f"Squeeze breakout detected. Signal: {squeeze_signal}")
            return squeeze_signal
        
        # 3. Check for multi-indicator confirmation signals
        multi_signal = self.get_multi_indicator_signal(df)
        if multi_signal:
            logger.info(f"Multi-indicator confirmation. Signal: {multi_signal}")
            return multi_signal
        
        # 4. Get grid signal (works in all market conditions)
        grid_signal = self.get_grid_signal(df)
        
        # 5. Get specific signals based on market condition
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            condition_signal = self.get_extreme_market_signal(df)
            logger.debug(f"EXTREME market detected. Grid signal: {grid_signal}, Extreme signal: {condition_signal}")
            
            # In extreme market conditions, prefer the condition-specific signal
            if condition_signal:
                return condition_signal
                
        elif market_condition in ['BULLISH', 'BEARISH']:
            if market_condition == 'BULLISH':
                condition_signal = self.get_bullish_signal(df)
            else:
                condition_signal = self.get_bearish_signal(df)
                
            logger.debug(f"{market_condition} market detected. Grid signal: {grid_signal}, Condition signal: {condition_signal}")
            
            # In trending markets, prefer the trending signal
            if condition_signal:
                return condition_signal
                
        elif market_condition == 'SIDEWAYS':
            condition_signal = self.get_sideways_signal(df)
            logger.debug(f"SIDEWAYS market detected. Grid signal: {grid_signal}, Sideways signal: {condition_signal}")
            
            # In sideways markets, prioritize mean reversion signals
            if condition_signal:
                return condition_signal
                
        # 6. Default to grid signal if no specialized signal was returned
        return grid_signal


# Update the factory function to include only RAYSOL strategy
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    from modules.config import (
        # RAYSOL parameters
        RAYSOL_GRID_LEVELS, RAYSOL_GRID_SPACING_PCT, RAYSOL_TREND_EMA_FAST, RAYSOL_TREND_EMA_SLOW,
        RAYSOL_VOLATILITY_LOOKBACK, RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD,
        RAYSOL_VOLUME_MA_PERIOD, RAYSOL_ADX_PERIOD, RAYSOL_ADX_THRESHOLD, RAYSOL_SIDEWAYS_THRESHOLD,
        RAYSOL_VOLATILITY_MULTIPLIER, RAYSOL_TREND_CONDITION_MULTIPLIER,
        RAYSOL_MIN_GRID_SPACING, RAYSOL_MAX_GRID_SPACING
    )
    
    strategies = {
        'RaysolDynamicGridStrategy': RaysolDynamicGridStrategy(
            grid_levels=RAYSOL_GRID_LEVELS,
            grid_spacing_pct=RAYSOL_GRID_SPACING_PCT,
            trend_ema_fast=RAYSOL_TREND_EMA_FAST,
            trend_ema_slow=RAYSOL_TREND_EMA_SLOW,
            volatility_lookback=RAYSOL_VOLATILITY_LOOKBACK,
            rsi_period=RSI_PERIOD,
            rsi_overbought=RSI_OVERBOUGHT,
            rsi_oversold=RSI_OVERSOLD,
            volume_ma_period=RAYSOL_VOLUME_MA_PERIOD,
            adx_period=RAYSOL_ADX_PERIOD,
            adx_threshold=RAYSOL_ADX_THRESHOLD,
            sideways_threshold=RAYSOL_SIDEWAYS_THRESHOLD,
            # Pass RAYSOL specific parameters
            volatility_multiplier=RAYSOL_VOLATILITY_MULTIPLIER,
            trend_condition_multiplier=RAYSOL_TREND_CONDITION_MULTIPLIER,
            min_grid_spacing=RAYSOL_MIN_GRID_SPACING,
            max_grid_spacing=RAYSOL_MAX_GRID_SPACING
        )
    }
    
    if strategy_name in strategies:
        return strategies[strategy_name]
    
    logger.warning(f"Strategy {strategy_name} not found. Defaulting to base trading strategy.")
    return TradingStrategy(strategy_name)


def get_strategy_for_symbol(symbol, strategy_name=None):
    """Get the appropriate strategy based on the trading symbol"""
    # If a specific strategy is requested, use it
    if strategy_name:
        return get_strategy(strategy_name)
    
    # Default to RAYSOLUSDT strategy for any symbol
    return RaysolDynamicGridStrategy()
    
    # Default to base strategy if needed
    # return TradingStrategy(symbol)

# End of file