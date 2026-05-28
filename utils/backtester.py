import os
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from utils.config import setup_logger, load_config
from utils.data_loader import download_nifty_data, prepare_labeled_dataset
from training.train_fusion import NiftyFusionDataset
from models.models import MarketFusionModel

logger = setup_logger("backtester", "system.log")
config = load_config()

def run_backtest(timeframe="15m", progress_callback=None):
    """
    Runs a historical backtest of the trading strategy using the trained Fusion model predictions
    on the validation/testing split.
    """
    save_dir = Path(config['model']['training']['save_dir'])
    model_path = save_dir / f"fusion_model_{timeframe}.pt"
    scaler_path = save_dir / f"scaler_{timeframe}.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error(f"Cannot run backtest: model or scaler for {timeframe} does not exist.")
        return {"error": "Model not trained yet."}
        
    # Load settings
    initial_capital = config['backtest']['initial_capital']
    rr_ratio = config['backtest']['risk_reward_ratio']
    sl_mult = config['backtest']['atr_multiplier_sl']
    target_mult = config['backtest']['atr_multiplier_target']
    seq_len = config['data']['seq_len']
    predict_ahead = config['data']['predict_ahead']
    
    # 1. Load data
    raw_df = download_nifty_data(timeframe)
    
    # Load Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Prepare features manually to ensure using saved scaler
    from utils.indicators import engineer_all_features
    feat_df = engineer_all_features(raw_df)
    
    future_close = feat_df['close'].shift(-predict_ahead)
    returns = (future_close - feat_df['close']) / feat_df['close']
    atr_series = feat_df['atr']
    threshold = 0.5 * (atr_series / feat_df['close'])
    
    labels = np.zeros(len(feat_df), dtype=int)
    labels[returns > threshold] = 1
    labels[returns < -threshold] = 2
    
    feat_df['label'] = labels
    feat_df_trimmed = feat_df.iloc[:-predict_ahead].copy()
    
    exclude_cols = ['label', 'open', 'high', 'low', 'close']
    feature_cols = [col for col in feat_df_trimmed.columns if col not in exclude_cols]
    
    scaled_feats = feat_df_trimmed[feature_cols].copy()
    num_cols = [c for c in feature_cols if not c.startswith('pattern_') 
                and c not in ['breakout_volume', 'consolidation', 'breakout_trendline', 'fake_breakout']]
    
    scaled_feats[num_cols] = scaler.transform(scaled_feats[num_cols])
    
    # 2. Form Dataset & Load Model
    full_dataset = NiftyFusionDataset(feat_df_trimmed, scaled_feats, labels, seq_len=seq_len)
    train_size = int(len(full_dataset) * config['data']['train_split'])
    
    # Perform inference on the validation segment
    val_indices = list(range(train_size, len(full_dataset)))
    
    device = torch.device("cuda" if torch.cuda.is_available() and config['system']['device'] == "auto" else "cpu")
    if config['system']['device'] in ["cuda", "cpu"]:
        device = torch.device(config['system']['device'])
        
    input_size = scaled_feats.shape[1]
    model = MarketFusionModel(
        lstm_input_size=input_size,
        lstm_hidden_dim=config['model']['lstm']['hidden_dim'],
        lstm_layers=config['model']['lstm']['num_layers'],
        cnn_backbone=config['model']['cnn']['backbone'],
        cnn_feature_dim=config['model']['cnn']['feature_dim'],
        fusion_hidden_dim=config['model']['fusion']['hidden_dim'],
        dropout=config['model']['fusion']['dropout']
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    logger.info("Generating predictions for backtest...")
    
    predictions = []
    probabilities = []
    
    # Process item by item (or in batch but save corresponding timestamps)
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_dataset, val_indices),
        batch_size=32,
        shuffle=False
    )
    
    with torch.no_grad():
        for seq_batch, img_batch, _ in val_loader:
            seq_batch, img_batch = seq_batch.to(device), img_batch.to(device)
            outputs = model(seq_batch, img_batch)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            predictions.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            
    # We correspond validation items to the raw DataFrame slice
    # Note that full_dataset index matching: index `idx` in dataset corresponds to raw `idx + seq_len` in df
    val_start_raw_idx = train_size + seq_len
    backtest_df = feat_df_trimmed.iloc[val_start_raw_idx:].copy()
    
    backtest_df['predicted_signal'] = predictions
    backtest_df['prob_sideways'] = [p[0] for p in probabilities]
    backtest_df['prob_bullish'] = [p[1] for p in probabilities]
    backtest_df['prob_bearish'] = [p[2] for p in probabilities]
    
    # 3. Simulate Trades
    capital = initial_capital
    position = 0  # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    sl_price = 0.0
    target_price = 0.0
    
    equity_curve = [capital]
    dates = [backtest_df.index[0]]
    
    trades = []
    
    # Trading Loop
    prices_close = backtest_df['close'].values
    prices_high = backtest_df['high'].values
    prices_low = backtest_df['low'].values
    atrs = backtest_df['atr'].values
    signals = backtest_df['predicted_signal'].values
    timestamps = backtest_df.index
    
    active_trade = None
    
    for i in range(len(backtest_df)):
        current_close = prices_close[i]
        current_high = prices_high[i]
        current_low = prices_low[i]
        current_atr = atrs[i]
        current_signal = signals[i]
        current_time = timestamps[i]
        
        # Check active trade exit
        if active_trade is not None:
            exit_triggered = False
            pnl_pct = 0.0
            
            if active_trade['type'] == 'long':
                # Check if SL or Target hit
                if current_low <= active_trade['sl']:
                    # SL hit (Loss)
                    exit_price = active_trade['sl']
                    pnl_pct = (exit_price - active_trade['entry']) / active_trade['entry']
                    exit_triggered = True
                    reason = "Stop Loss"
                elif current_high >= active_trade['target']:
                    # Target hit (Win)
                    exit_price = active_trade['target']
                    pnl_pct = (exit_price - active_trade['entry']) / active_trade['entry']
                    exit_triggered = True
                    reason = "Target"
            else: # Short trade
                if current_high >= active_trade['sl']:
                    exit_price = active_trade['sl']
                    pnl_pct = (active_trade['entry'] - exit_price) / active_trade['entry']
                    exit_triggered = True
                    reason = "Stop Loss"
                elif current_low <= active_trade['target']:
                    exit_price = active_trade['target']
                    pnl_pct = (active_trade['entry'] - exit_price) / active_trade['entry']
                    exit_triggered = True
                    reason = "Target"
                    
            if exit_triggered:
                # Update capital
                trade_profit = capital * pnl_pct * 5.0 # Leverage = 5x
                capital += trade_profit
                
                active_trade['exit_time'] = current_time
                active_trade['exit_price'] = exit_price
                active_trade['pnl'] = trade_profit
                active_trade['pnl_pct'] = pnl_pct * 100
                active_trade['reason'] = reason
                active_trade['capital_after'] = capital
                trades.append(active_trade)
                
                active_trade = None
                
        # If no active trade, check entry signals
        if active_trade is None:
            if current_signal == 1: # Bullish signal
                # Enter Long
                entry = current_close
                sl = entry - sl_mult * current_atr
                target = entry + target_mult * current_atr
                active_trade = {
                    'type': 'long',
                    'entry_time': current_time,
                    'entry': entry,
                    'sl': sl,
                    'target': target
                }
            elif current_signal == 2: # Bearish signal
                # Enter Short
                entry = current_close
                sl = entry + sl_mult * current_atr
                target = entry - target_mult * current_atr
                active_trade = {
                    'type': 'short',
                    'entry_time': current_time,
                    'entry': entry,
                    'sl': sl,
                    'target': target
                }
                
        equity_curve.append(capital)
        dates.append(current_time)
        
    # Calculate performance statistics
    equity_curve = np.array(equity_curve)
    total_return_pct = ((capital - initial_capital) / initial_capital) * 100
    
    # Trades calculations
    total_trades = len(trades)
    winning_trades = sum(1 for t in trades if t['pnl_pct'] > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    # Calculate Max Drawdown
    peak = equity_curve[0]
    drawdowns = []
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        drawdowns.append(dd)
    max_dd_pct = np.max(drawdowns) * 100 if drawdowns else 0.0
    
    # Calculate Sharpe Ratio
    daily_returns = np.diff(equity_curve) / equity_curve[:-1]
    mean_ret = np.mean(daily_returns) if len(daily_returns) > 0 else 0
    std_ret = np.std(daily_returns) if len(daily_returns) > 0 else 1
    # Annualized Sharpe (assuming 252 trading days a year or equivalent intervals)
    sharpe = (mean_ret / (std_ret + 1e-10)) * np.sqrt(252) if len(daily_returns) > 0 else 0
    
    # Plot equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(dates, equity_curve, label='Strategy Equity Curve', color='#1f77b4', linewidth=2)
    plt.title(f'NIFTY 50 Backtest Performance ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel('Capital (INR)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    plot_path = save_dir / f"backtest_equity_{timeframe}.png"
    plt.savefig(plot_path)
    plt.close()
    
    results = {
        "initial_capital": initial_capital,
        "final_capital": capital,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "max_drawdown_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
        "trades_list": trades,
        "equity_curve_plot": str(plot_path)
    }
    
    logger.info(f"Backtest completed: Return = {total_return_pct:.2f}%, Win Rate = {win_rate:.2f}%")
    return results

if __name__ == "__main__":
    run_backtest(timeframe="15m")
