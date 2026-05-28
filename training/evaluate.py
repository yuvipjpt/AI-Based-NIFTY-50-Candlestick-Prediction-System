import os
import pickle
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from utils.config import setup_logger, load_config
from utils.data_loader import download_nifty_data, prepare_labeled_dataset
from training.train_fusion import NiftyFusionDataset
from models.models import MarketFusionModel

logger = setup_logger("evaluate", "system.log")
config = load_config()

def evaluate_model(timeframe="15m"):
    """Evaluates the trained Fusion Model and generates metrics/graphs."""
    save_dir = Path(config['model']['training']['save_dir'])
    model_path = save_dir / f"fusion_model_{timeframe}.pt"
    scaler_path = save_dir / f"scaler_{timeframe}.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        logger.error(f"Cannot find trained model or scaler for {timeframe}. Train model first.")
        return None
        
    # Load raw data
    raw_df = download_nifty_data(timeframe)
    
    seq_len = config['data']['seq_len']
    predict_ahead = config['data']['predict_ahead']
    
    # Load Scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    # Prepare features and labels (force utilizing loaded scaler)
    # We will do this manually to ensure we use the saved scaler parameters
    from utils.indicators import engineer_all_features
    feat_df = engineer_all_features(raw_df)
    
    future_close = feat_df['close'].shift(-predict_ahead)
    returns = (future_close - feat_df['close']) / feat_df['close']
    atr = feat_df['atr']
    threshold = 0.5 * (atr / feat_df['close'])
    
    labels = np.zeros(len(feat_df), dtype=int)
    labels[returns > threshold] = 1   # Bullish
    labels[returns < -threshold] = 2  # Bearish
    
    feat_df['label'] = labels
    feat_df = feat_df.iloc[:-predict_ahead].copy()
    
    exclude_cols = ['label', 'open', 'high', 'low', 'close']
    feature_cols = [col for col in feat_df.columns if col not in exclude_cols]
    
    scaled_feats = feat_df[feature_cols].copy()
    num_cols = [c for c in feature_cols if not c.startswith('pattern_') 
                and c not in ['breakout_volume', 'consolidation', 'breakout_trendline', 'fake_breakout']]
    
    # Transform using pre-loaded scaler
    scaled_feats[num_cols] = scaler.transform(scaled_feats[num_cols])
    
    # Form Dataset
    full_dataset = NiftyFusionDataset(feat_df, scaled_feats, labels, seq_len=seq_len)
    
    # Take validation slice (last 20% of data)
    train_size = int(len(full_dataset) * config['data']['train_split'])
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    val_loader = DataLoader(val_dataset, batch_size=config['model']['training']['batch_size'], shuffle=False)
    
    # Initialize Model
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
    
    y_true = []
    y_pred = []
    y_probs = []
    
    logger.info("Running evaluation prediction loop...")
    with torch.no_grad():
        for seq_batch, img_batch, label_batch in val_loader:
            seq_batch, img_batch = seq_batch.to(device), img_batch.to(device)
            outputs = model(seq_batch, img_batch)
            probs = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(label_batch.numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    accuracy = accuracy_score(y_true, y_pred)
    conf_mat = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Sideways', 'Bullish', 'Bearish'], output_dict=True)
    
    logger.info(f"Evaluation Accuracy: {accuracy:.2%}")
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    classes = ['Sideways', 'Bullish', 'Bearish']
    plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'NIFTY 50 Predictor Confusion Matrix - {timeframe}')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text numbers to cells
    thresh = conf_mat.max() / 2.
    for i, j in np.ndindex(conf_mat.shape):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")
                 
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    plot_path = save_dir / f"confusion_matrix_{timeframe}.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {plot_path}")
    
    eval_results = {
        "accuracy": accuracy,
        "confusion_matrix": conf_mat.tolist(),
        "classification_report": report,
        "confusion_matrix_plot": str(plot_path)
    }
    
    # Save text report summary
    report_file = save_dir / f"evaluation_report_{timeframe}.txt"
    with open(report_file, 'w') as rf:
        rf.write("=== MODEL EVALUATION REPORT ===\n")
        rf.write(f"Timeframe: {timeframe}\n")
        rf.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        rf.write("Classification Report:\n")
        rf.write(classification_report(y_true, y_pred, target_names=['Sideways', 'Bullish', 'Bearish']))
    logger.info(f"Saved text report to {report_file}")
    
    return eval_results

if __name__ == "__main__":
    evaluate_model(timeframe="15m")
