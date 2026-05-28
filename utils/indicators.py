import numpy as np
import pandas as pd
import logging
from utils.config import setup_logger

logger = setup_logger("indicators", "system.log")

# Try to import pandas_ta
PANDAS_TA_AVAILABLE = False
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
    logger.info("pandas-ta library detected and imported successfully.")
except ImportError:
    logger.warning("pandas-ta library not found. Falling back to custom vectorized technical indicators.")

# Try to import talib
TALIB_AVAILABLE = False
try:
    import talib
    TALIB_AVAILABLE = True
    logger.info("TA-Lib library detected and imported successfully.")
except ImportError:
    logger.warning("TA-Lib library not found. Falling back to custom/pandas-ta indicators.")


def calculate_rsi(df, period=14):
    """Calculates the Relative Strength Index."""
    if PANDAS_TA_AVAILABLE:
        try:
            return df.ta.rsi(length=period)
        except Exception:
            pass
            
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculates MACD, Signal line, and Histograms."""
    if PANDAS_TA_AVAILABLE:
        try:
            macd_df = df.ta.macd(fast=fast, slow=slow, signal=signal)
            if macd_df is not None:
                return macd_df.iloc[:, 0], macd_df.iloc[:, 1], macd_df.iloc[:, 2]
        except Exception:
            pass
            
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_vwap(df):
    """Calculates Volume Weighted Average Price (VWAP)."""
    # Try to group by day if timestamp exists, otherwise roll
    price = (df['high'] + df['low'] + df['close']) / 3
    volume = df['volume']
    
    if 'datetime' in df.columns or isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.date if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['datetime']).dt.date
        cum_pv = (price * volume).groupby(dates).cumsum()
        cum_v = volume.groupby(dates).cumsum()
        vwap = cum_pv / (cum_v + 1e-10)
    else:
        # Fallback to rolling cumulative window of 100
        cum_pv = (price * volume).rolling(window=100, min_periods=1).sum()
        cum_v = volume.rolling(window=100, min_periods=1).sum()
        vwap = cum_pv / (cum_v + 1e-10)
        
    return vwap


def calculate_atr(df, period=14):
    """Calculates Average True Range."""
    if PANDAS_TA_AVAILABLE:
        try:
            return df.ta.atr(length=period)
        except Exception:
            pass
            
    high_low = df['high'] - df['low']
    high_close_prev = (df['high'] - df['close'].shift(1)).abs()
    low_close_prev = (df['low'] - df['close'].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Calculates Bollinger Bands (Upper, Middle, Lower)."""
    if PANDAS_TA_AVAILABLE:
        try:
            bb = df.ta.bbands(length=period, std=std_dev)
            if bb is not None:
                return bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        except Exception:
            pass
            
    mid = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = mid + (std_dev * std)
    lower = mid - (std_dev * std)
    return upper, mid, lower


def calculate_all_indicators(df):
    """Generates all requested technical indicators and adds them to DataFrame."""
    res_df = df.copy()
    
    # EMAs & SMA
    res_df['ema_8'] = res_df['close'].ewm(span=8, adjust=False).mean()
    res_df['ema_21'] = res_df['close'].ewm(span=21, adjust=False).mean()
    res_df['ema_50'] = res_df['close'].ewm(span=50, adjust=False).mean()
    res_df['ema_200'] = res_df['close'].ewm(span=200, adjust=False).mean()
    res_df['sma_20'] = res_df['close'].rolling(window=20).mean()
    
    # RSI, MACD, VWAP, ATR, Bollinger
    res_df['rsi'] = calculate_rsi(res_df)
    macd, signal, hist = calculate_macd(res_df)
    res_df['macd'] = macd
    res_df['macd_signal'] = signal
    res_df['macd_hist'] = hist
    res_df['vwap'] = calculate_vwap(res_df)
    res_df['atr'] = calculate_atr(res_df)
    
    upper, mid, lower = calculate_bollinger_bands(res_df)
    res_df['bb_upper'] = upper
    res_df['bb_middle'] = mid
    res_df['bb_lower'] = lower
    
    # Advanced Indicators (ADX, CCI, WILLR, Stochastic, Aroon, MOM, ROC, MFI, OBV, SAR)
    if TALIB_AVAILABLE:
        try:
            hi = res_df['high'].values
            lo = res_df['low'].values
            cl = res_df['close'].values
            vol = res_df['volume'].values.astype(float)
            
            res_df['adx'] = talib.ADX(hi, lo, cl, timeperiod=14)
            res_df['cci'] = talib.CCI(hi, lo, cl, timeperiod=14)
            res_df['willr'] = talib.WILLR(hi, lo, cl, timeperiod=14)
            slowk, slowd = talib.STOCH(hi, lo, cl, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            res_df['stoch_k'] = slowk
            res_df['stoch_d'] = slowd
            res_df['aroon'] = talib.AROONOSC(hi, lo, timeperiod=14)
            res_df['mom'] = talib.MOM(cl, timeperiod=10)
            res_df['roc'] = talib.ROC(cl, timeperiod=10)
            res_df['mfi'] = talib.MFI(hi, lo, cl, vol, timeperiod=14)
            res_df['obv'] = talib.OBV(cl, vol)
            res_df['sar'] = talib.SAR(hi, lo, acceleration=0.02, maximum=0.2)
        except Exception as e:
            logger.error(f"Error calculating TA-Lib indicators: {e}. Trying pandas-ta fallbacks.")
            
    # Fallback to pandas_ta for columns not added yet
    for col in ['adx', 'cci', 'willr', 'stoch_k', 'stoch_d', 'aroon', 'mom', 'roc', 'mfi', 'obv', 'sar']:
        if col not in res_df.columns:
            try:
                if col == 'adx' and PANDAS_TA_AVAILABLE:
                    res_df['adx'] = res_df.ta.adx(length=14).iloc[:, 0]
                elif col == 'cci' and PANDAS_TA_AVAILABLE:
                    res_df['cci'] = res_df.ta.cci(length=14)
                elif col == 'willr' and PANDAS_TA_AVAILABLE:
                    res_df['willr'] = res_df.ta.willr(length=14)
                elif col in ['stoch_k', 'stoch_d'] and PANDAS_TA_AVAILABLE:
                    stoch_df = res_df.ta.stoch(fastk_period=5, slowk_period=3)
                    res_df['stoch_k'] = stoch_df.iloc[:, 0]
                    res_df['stoch_d'] = stoch_df.iloc[:, 1]
                elif col == 'aroon' and PANDAS_TA_AVAILABLE:
                    res_df['aroon'] = res_df.ta.aroon(length=14).iloc[:, 2]
                elif col == 'mom' and PANDAS_TA_AVAILABLE:
                    res_df['mom'] = res_df.ta.mom(length=10)
                elif col == 'roc' and PANDAS_TA_AVAILABLE:
                    res_df['roc'] = res_df.ta.roc(length=10)
                elif col == 'mfi' and PANDAS_TA_AVAILABLE:
                    res_df['mfi'] = res_df.ta.mfi(length=14)
                elif col == 'obv' and PANDAS_TA_AVAILABLE:
                    res_df['obv'] = res_df.ta.obv()
                elif col == 'sar' and PANDAS_TA_AVAILABLE:
                    res_df['sar'] = res_df.ta.psar().iloc[:, 0].fillna(res_df['close'])
                else:
                    res_df[col] = 0.0
            except Exception as e:
                logger.warning(f"Failed to calculate fallback for {col}: {e}. Zero-filling.")
                res_df[col] = 0.0
                
    # Drop rows with NaNs to clean up features
    res_df.bfill(inplace=True)
    return res_df


def detect_candlestick_patterns(df):
    """
    Detects candlestick patterns using TA-Lib or manual fallback rules.
    Returns a DataFrame with boolean/integer indicators for each pattern.
    1 = Bullish pattern, -1 = Bearish pattern, 0 = No pattern
    """
    pat_df = pd.DataFrame(index=df.index)
    
    if TALIB_AVAILABLE:
        try:
            op = df['open'].values
            hi = df['high'].values
            lo = df['low'].values
            cl = df['close'].values
            
            # Find all CDL functions in talib
            cdl_funcs = [func for func in dir(talib) if func.startswith('CDL')]
            for func_name in cdl_funcs:
                col_name = f"pattern_{func_name.lower()}"
                func = getattr(talib, func_name)
                res = func(op, hi, lo, cl)
                pat_df[col_name] = (res / 100).astype(int)
            return pat_df
        except Exception as e:
            logger.error(f"Error using TA-Lib for pattern detection: {e}. Falling back to manual patterns.")
            pat_df = pd.DataFrame(index=df.index)
            
    # Basic candle definitions
    body = (df['close'] - df['open']).abs()
    direction = np.sign(df['close'] - df['open'])  # 1 for green, -1 for red, 0 for flat
    candle_range = df['high'] - df['low']
    candle_range = candle_range.replace(0, 1e-10) # Avoid division by zero
    
    upper_wick = df['high'] - np.maximum(df['open'], df['close'])
    lower_wick = np.minimum(df['open'], df['close']) - df['low']
    
    avg_body = body.rolling(window=10).mean()
    avg_range = candle_range.rolling(window=10).mean()
    
    # 1. Doji (Very small body relative to range)
    pat_df['pattern_doji'] = (body <= 0.1 * candle_range).astype(int)
    
    # 2. Hammer (Small body near high, long lower wick)
    pat_df['pattern_hammer'] = (
        (lower_wick >= 2 * body) & 
        (upper_wick <= 0.1 * candle_range) & 
        (body > 0.05 * candle_range)
    ).astype(int)
    
    # 3. Inverted Hammer (Small body near low, long upper wick)
    pat_df['pattern_inverted_hammer'] = (
        (upper_wick >= 2 * body) & 
        (lower_wick <= 0.1 * candle_range) & 
        (body > 0.05 * candle_range)
    ).astype(int)
    
    # 4. Shooting Star (Bearish version of inverted hammer, occurring in uptrend)
    pat_df['pattern_shooting_star'] = (
        (upper_wick >= 2 * body) & 
        (lower_wick <= 0.1 * candle_range) & 
        (direction == -1) & 
        (body > 0.05 * candle_range)
    ).astype(int)
    
    # 5. Bullish Engulfing (Red candle followed by larger Green candle)
    pat_df['pattern_bullish_engulfing'] = (
        (direction.shift(1) == -1) & 
        (direction == 1) & 
        (df['close'] > df['open'].shift(1)) & 
        (df['open'] < df['close'].shift(1))
    ).astype(int)
    
    # 6. Bearish Engulfing (Green candle followed by larger Red candle)
    pat_df['pattern_bearish_engulfing'] = (
        (direction.shift(1) == 1) & 
        (direction == -1) & 
        (df['close'] < df['open'].shift(1)) & 
        (df['open'] > df['close'].shift(1))
    ).astype(int)
    
    # 7. Morning Star (Three-candle bullish pattern)
    pat_df['pattern_morning_star'] = (
        (direction.shift(2) == -1) &                         # Candle 1: Red
        (body.shift(2) > avg_body.shift(2)) & 
        (body.shift(1) < 0.3 * avg_body.shift(2)) &           # Candle 2: Small body (star)
        (direction == 1) &                                  # Candle 3: Green
        (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2) # Closes > 50% of Candle 1
    ).astype(int)
    
    # 8. Evening Star (Three-candle bearish pattern)
    pat_df['pattern_evening_star'] = (
        (direction.shift(2) == 1) &                          # Candle 1: Green
        (body.shift(2) > avg_body.shift(2)) & 
        (body.shift(1) < 0.3 * avg_body.shift(2)) &           # Candle 2: Small body
        (direction == -1) &                                 # Candle 3: Red
        (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2) # Closes < 50% of Candle 1
    ).astype(int)
    
    # 9. Harami (Inside bar: small body engulfed by previous large body)
    pat_df['pattern_harami'] = (
        (body.shift(1) > avg_body.shift(1)) & 
        (df['high'] < df['high'].shift(1)) & 
        (df['low'] > df['low'].shift(1)) & 
        (body < 0.5 * body.shift(1))
    ).astype(int)
    # Refine direction for Bullish vs Bearish Harami
    pat_df['pattern_harami'] = pat_df['pattern_harami'] * direction.shift(1) * -1 # If prev was red, harami is bullish (+1), else bearish (-1)
    
    # 10. Marubozu (Large body, little to no wicks)
    marubozu_bullish = (body >= 0.85 * candle_range) & (direction == 1) & (body > avg_body)
    marubozu_bearish = (body >= 0.85 * candle_range) & (direction == -1) & (body > avg_body)
    pat_df['pattern_marubozu'] = marubozu_bullish.astype(int) - marubozu_bearish.astype(int)
    
    # 11. Tweezer Top (Two candles with similar highs in uptrend)
    pat_df['pattern_tweezer_top'] = (
        (np.abs(df['high'] - df['high'].shift(1)) / (df['high'] + 1e-10) < 0.001) & 
        (direction.shift(1) == 1) & 
        (direction == -1)
    ).astype(int)
    
    # 12. Tweezer Bottom (Two candles with similar lows in downtrend)
    pat_df['pattern_tweezer_bottom'] = (
        (np.abs(df['low'] - df['low'].shift(1)) / (df['low'] + 1e-10) < 0.001) & 
        (direction.shift(1) == -1) & 
        (direction == 1)
    ).astype(int)
    
    # 13. Three White Soldiers (Three consecutive green candles with increasing/close prices)
    pat_df['pattern_three_white_soldiers'] = (
        (direction.shift(2) == 1) & (direction.shift(1) == 1) & (direction == 1) &
        (df['close'] > df['close'].shift(1)) & (df['close'].shift(1) > df['close'].shift(2)) &
        (body.shift(2) > 0.5 * avg_body) & (body.shift(1) > 0.5 * avg_body) & (body > 0.5 * avg_body)
    ).astype(int)
    
    # 14. Three Black Crows (Three consecutive red candles with decreasing/close prices)
    pat_df['pattern_three_black_crows'] = (
        (direction.shift(2) == -1) & (direction.shift(1) == -1) & (direction == -1) &
        (df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2)) &
        (body.shift(2) > 0.5 * avg_body) & (body.shift(1) > 0.5 * avg_body) & (body > 0.5 * avg_body)
    ).astype(int)
    
    return pat_df


def identify_support_resistance(df, window=20):
    """
    Finds Support and Resistance levels using rolling local minima/maxima.
    Returns: Support levels, Resistance levels (lists of float prices)
    """
    highs = df['high'].values
    lows = df['low'].values
    
    supports = []
    resistances = []
    
    for i in range(window, len(df) - window):
        # Local Maxima (Resistance)
        if highs[i] == np.max(highs[i - window : i + window + 1]):
            # Filter close levels to prevent duplicates
            if not any(abs(r - highs[i]) / highs[i] < 0.01 for r in resistances):
                resistances.append(float(highs[i]))
        
        # Local Minima (Support)
        if lows[i] == np.min(lows[i - window : i + window + 1]):
            if not any(abs(s - lows[i]) / lows[i] < 0.01 for s in supports):
                supports.append(float(lows[i]))
                
    return sorted(supports), sorted(resistances)


def detect_breakouts_and_consolidation(df, window=20):
    """
    Detects Trendline breakouts, consolidation, volume breakout, and fake breakout.
    Returns a DataFrame with indicators:
    - breakout_volume: bool
    - consolidation: bool
    - breakout_trendline: int (-1 for down, 1 for up, 0 for none)
    - fake_breakout: int (-1 for fake down, 1 for fake up, 0 for none)
    """
    res_df = pd.DataFrame(index=df.index)
    
    # 1. Volume Breakout: Volume exceeds 1.5x of its rolling SMA
    vol_sma = df['volume'].rolling(window=window).mean()
    res_df['breakout_volume'] = (df['volume'] > 1.5 * vol_sma).astype(int)
    
    # 2. Consolidation Zone: ATR is less than 0.7x of its historical average ATR (20 window)
    atr = calculate_atr(df, period=14)
    atr_sma = atr.rolling(window=window).mean()
    res_df['consolidation'] = (atr < 0.7 * atr_sma).astype(int)
    
    # 3. Support/Resistance Breakouts
    # We will compute a simple rolling support (low of last N candles) and resistance (high of last N candles)
    roll_high = df['high'].shift(1).rolling(window=window).max()
    roll_low = df['low'].shift(1).rolling(window=window).min()
    
    breakout_up = (df['close'] > roll_high)
    breakout_down = (df['close'] < roll_low)
    
    res_df['breakout_trendline'] = breakout_up.astype(int) - breakout_down.astype(int)
    
    # 4. Fake Breakout:
    # A breakout occurred in the previous 2 candles, but the price closed back inside the range,
    # or the candle has a very long wick rejection on lower volume.
    prev_breakout_up = breakout_up.shift(1).fillna(False) | breakout_up.shift(2).fillna(False)
    prev_breakout_down = breakout_down.shift(1).fillna(False) | breakout_down.shift(2).fillna(False)
    
    fake_breakout_up = prev_breakout_up & (df['close'] <= roll_high) & (df['volume'] < vol_sma)
    fake_breakout_down = prev_breakout_down & (df['close'] >= roll_low) & (df['volume'] < vol_sma)
    
    res_df['fake_breakout'] = fake_breakout_up.astype(int) - fake_breakout_down.astype(int)
    
    return res_df


def engineer_all_features(df):
    """
    Combines basic OHLCV, all indicators, breakouts, and candlestick patterns.
    Returns complete feature DataFrame.
    """
    df_ind = calculate_all_indicators(df)
    df_pat = detect_candlestick_patterns(df_ind)
    df_brk = detect_breakouts_and_consolidation(df_ind)
    
    features_df = pd.concat([df_ind, df_pat, df_brk], axis=1)
    features_df.dropna(inplace=True)
    return features_df
