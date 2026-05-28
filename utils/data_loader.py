import os
import yfinance as yf
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from utils.config import setup_logger, load_config
from utils.indicators import engineer_all_features

logger = setup_logger("data_loader", "system.log")
config = load_config()

def generate_synthetic_data(timeframe="1d", length=1000):
    """Generates realistic synthetic trading data for offline testing."""
    logger.info(f"Generating synthetic NIFTY 50 data for timeframe {timeframe}...")
    np.random.seed(42)
    
    # Generate dates
    if timeframe == "1d":
        dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='D')
    elif timeframe in ["1h", "15m", "5m", "1m"]:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='h') # Approximate
    else:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=length, freq='D')
        
    # Geometric Brownian Motion simulation
    s0 = 18000.0  # Initial price
    mu = 0.0001
    sigma = 0.01
    
    returns = np.random.normal(mu, sigma, length)
    price_path = s0 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame(index=dates)
    df.index.name = "datetime"
    
    df['close'] = price_path
    df['open'] = df['close'].shift(1) * (1 + np.random.normal(0, 0.002, length))
    df.loc[df.index[0], 'open'] = s0 * 0.998
    
    # Ensure open-close boundaries
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + np.abs(np.random.normal(0, 0.004, length)))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - np.abs(np.random.normal(0, 0.004, length)))
    df['volume'] = np.random.randint(100000, 5000000, size=length)
    
    # Clean up NaNs
    df.bfill(inplace=True)
    return df


def download_nifty_data(timeframe="15m", force_download=False):
    """Downloads NIFTY 50 historical data based on timeframe configuration."""
    symbol = config['data']['symbol']
    cache_dir = Path(config['data']['cache_dir'])
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_file = cache_dir / f"nifty50_{timeframe}.csv"
    
    custom_dataset_name = "nifty_5year_complete_dataset.csv"
    custom_path = Path(__file__).resolve().parent.parent / custom_dataset_name
    use_custom = config['data'].get('use_custom_dataset', False)
    if use_custom and custom_path.exists() and timeframe == "1d" and not force_download:
        logger.info(f"Detected user-provided NIFTY 5-year dataset: {custom_dataset_name}. Loading custom data for daily timeframe.")
        try:
            df = pd.read_csv(custom_path)
            # Find the date column
            date_col = None
            for col in ['Date', 'datetime', 'date', 'DATE']:
                if col in df.columns:
                    date_col = col
                    break
            if date_col:
                df['datetime'] = pd.to_datetime(df[date_col])
                df.set_index('datetime', inplace=True)
            else:
                df.index = pd.to_datetime(df.index)
                df.index.name = "datetime"
            
            # Normalize column names to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            # Keep only standard columns to pass through indicators engineering
            standard_cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[standard_cols]
            
            # Cache it as the daily dataset
            df.to_csv(cache_file)
            logger.info(f"Cached {len(df)} records from user-supplied NIFTY 5-year dataset to {cache_file}.")
            return df
        except Exception as e:
            logger.error(f"Error loading custom user dataset {custom_dataset_name}: {e}. Reverting to standard pipeline.")
            
    if not force_download and cache_file.exists():
        logger.info(f"Loading NIFTY 50 {timeframe} data from local cache: {cache_file}")
        try:
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df
        except Exception as e:
            logger.error(f"Error loading cached file {cache_file}: {e}. Retrying download.")
            
    # Set yfinance parameters based on limits
    # Timeframe limits: 1m (max 7d), 5m/15m (max 60d), 1h (max 730d), 1d (max max)
    period_map = {
        "1m": "7d",
        "5m": "60d",
        "15m": "60d",
        "1h": "730d",
        "1d": "max"
    }
    
    period = period_map.get(timeframe, "60d")
    
    logger.info(f"Downloading NIFTY 50 {timeframe} data from yfinance (symbol={symbol}, period={period})...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(interval=timeframe, period=period)
        
        if df.empty:
            logger.warning("Empty DataFrame returned from yfinance. Fetching synthetic fallback.")
            df = generate_synthetic_data(timeframe)
        else:
            # Rename columns to lowercase standard
            df.columns = [col.lower() for col in df.columns]
            # Verify if index is datetime
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df.index.name = "datetime"
            # Keep only OHLCV
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Save cache
            df.to_csv(cache_file)
            logger.info(f"Cached {len(df)} records to {cache_file}.")
            
    except Exception as e:
        logger.error(f"Failed to download data from yfinance: {e}. Reverting to synthetic data.")
        df = generate_synthetic_data(timeframe)
        df.to_csv(cache_file)
        
    return df


def prepare_labeled_dataset(df, seq_len=30, predict_ahead=1):
    """
    Applies technical indicators, scales numerical features, and creates classification labels:
    - 0: Sideways
    - 1: Bullish
    - 2: Bearish
    
    Returns scaled features, labels, scaler instance, and raw engineered DataFrame.
    """
    logger.info("Engineering indicators and pattern features...")
    feat_df = engineer_all_features(df)
    
    # Set up labels: look ahead by predict_ahead candles
    future_close = feat_df['close'].shift(-predict_ahead)
    returns = (future_close - feat_df['close']) / feat_df['close']
    
    # Dynamic volatility threshold: 0.5 * ATR / Close
    # This represents a move exceeding half of average daily/interval range
    atr = feat_df['atr']
    threshold = 0.5 * (atr / feat_df['close'])
    
    labels = np.zeros(len(feat_df), dtype=int)
    labels[returns > threshold] = 1   # Bullish
    labels[returns < -threshold] = 2  # Bearish
    
    # Add label column and shift future returns to drop final values which have no targets
    feat_df['label'] = labels
    feat_df = feat_df.iloc[:-predict_ahead].copy()
    
    # Feature Selection: columns to feed into model
    # Drop raw OHLCV prices (which are non-stationary) but keep volume and scaled indicators
    # Note: Keep close in feat_df for plotting/reference, but drop from scaling
    exclude_cols = ['label', 'open', 'high', 'low', 'close']
    feature_cols = [col for col in feat_df.columns if col not in exclude_cols]
    
    # Standardize numerical features
    scaler = StandardScaler()
    scaled_features = feat_df[feature_cols].copy()
    
    # Identify numerical columns (excluding boolean pattern indicators which are 0/1/-1)
    num_cols = [c for c in feature_cols if not c.startswith('pattern_') 
                and c not in ['breakout_volume', 'consolidation', 'breakout_trendline', 'fake_breakout']]
    
    scaled_features[num_cols] = scaler.fit_transform(scaled_features[num_cols])
    
    return scaled_features, feat_df['label'].values, scaler, feat_df


def create_sequences(scaled_features, labels, seq_len=30):
    """Creates sliding window input arrays for the LSTM model."""
    X, y = [], []
    features_array = scaled_features.values
    
    for i in range(len(features_array) - seq_len):
        X.append(features_array[i : i + seq_len])
        y.append(labels[i + seq_len])
        
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
