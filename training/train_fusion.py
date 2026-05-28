import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import cv2
import logging
from pathlib import Path
from utils.config import setup_logger, load_config
from utils.data_loader import download_nifty_data, prepare_labeled_dataset, create_sequences
from models.models import MarketFusionModel

logger = setup_logger("train_fusion", "system.log")
config = load_config()

def render_chart_opencv(ohlc_values, width=224, height=224):
    """
    Rapidly renders a sequence of candles onto an OpenCV canvas in memory.
    ohlc_values shape: [seq_len, 4] (Open, High, Low, Close)
    Returns: BGR numpy array [height, width, 3] normalized [0, 1]
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8) # Dark background
    seq_len = len(ohlc_values)
    
    # Pad borders
    pad_h = 10
    pad_w = 10
    
    # Calculate price scale
    highs = ohlc_values[:, 1]
    lows = ohlc_values[:, 2]
    
    p_max = np.max(highs)
    p_min = np.min(lows)
    p_span = p_max - p_min + 1e-10
    
    # Calculate spacing
    candle_w = max(1, int((width - 2 * pad_w) / seq_len) - 1)
    step_w = (width - 2 * pad_w) / seq_len
    
    for i in range(seq_len):
        o, h, l, c = ohlc_values[i]
        
        # Calculate pixel coordinates (y is inverted in image coordinates)
        y_open = int(pad_h + (height - 2 * pad_h) * (1 - (o - p_min) / p_span))
        y_close = int(pad_h + (height - 2 * pad_h) * (1 - (c - p_min) / p_span))
        y_high = int(pad_h + (height - 2 * pad_h) * (1 - (h - p_min) / p_span))
        y_low = int(pad_h + (height - 2 * pad_h) * (1 - (l - p_min) / p_span))
        
        center_x = int(pad_w + i * step_w + step_w / 2)
        x_left = center_x - candle_w // 2
        x_right = center_x + candle_w // 2
        
        # Determine color (BGR)
        # Green: (0, 200, 0), Red: (0, 0, 220)
        color = (0, 220, 0) if c >= o else (0, 0, 220)
        
        # Draw wicks
        cv2.line(canvas, (center_x, y_high), (center_x, y_low), color, 1)
        
        # Draw body
        y_top = min(y_open, y_close)
        y_bottom = max(y_open, y_close)
        
        # Ensure at least 1 pixel height for bodies
        if y_top == y_bottom:
            y_bottom += 1
            
        cv2.rectangle(canvas, (x_left, y_top), (x_right, y_bottom), color, -1)
        
    return canvas.astype(np.float32) / 255.0


class NiftyFusionDataset(Dataset):
    """PyTorch Dataset that returns both numerical sequences and rendered images."""
    def __init__(self, raw_ohlc, scaled_features, labels, seq_len=30):
        self.raw_ohlc = raw_ohlc[['open', 'high', 'low', 'close']].values
        self.scaled_features = scaled_features.values
        self.labels = labels
        self.seq_len = seq_len
        
        # Pre-render all visual charts in memory for 100x training speedup (set to 32x32 for high CPU efficiency)
        backbone = config['model']['cnn']['backbone']
        self.img_size = 224 if backbone == "resnet18" else 32
        logger.info(f"Pre-rendering all {len(self.scaled_features) - seq_len} chart images in memory...")
        
        self.pre_rendered_images = []
        for i in range(len(self.scaled_features) - seq_len):
            ohlc_slice = self.raw_ohlc[i : i + seq_len]
            img = render_chart_opencv(ohlc_slice, width=self.img_size, height=self.img_size)
            img_tensor = torch.tensor(img).permute(2, 0, 1)
            self.pre_rendered_images.append(img_tensor)
            
        logger.info("Pre-rendering complete.")
        
    def __len__(self):
        return len(self.scaled_features) - self.seq_len
        
    def __getitem__(self, idx):
        # Numerical sequence: [seq_len, num_features]
        seq_data = self.scaled_features[idx : idx + self.seq_len]
        
        # Visual chart image: load from pre-rendered memory
        img_tensor = self.pre_rendered_images[idx]
        
        label = self.labels[idx + self.seq_len]
        
        return torch.tensor(seq_data, dtype=torch.float32), img_tensor, torch.tensor(label, dtype=torch.long)


def train_model(timeframe="15m", epochs=None, batch_size=None, lr=None):
    """Executes the training sequence for the hybrid fusion model."""
    epochs = epochs or config['model']['training']['epochs']
    batch_size = batch_size or config['model']['training']['batch_size']
    lr = lr or config['model']['training']['learning_rate']
    
    # 1. Download and load data
    raw_df = download_nifty_data(timeframe)
    
    # 2. Prep dataset
    seq_len = config['data']['seq_len']
    predict_ahead = config['data']['predict_ahead']
    
    scaled_feats, labels, scaler, prep_df = prepare_labeled_dataset(
        raw_df, seq_len=seq_len, predict_ahead=predict_ahead
    )
    
    # Save Scaler
    save_dir = Path(config['model']['training']['save_dir'])
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = save_dir / f"scaler_{timeframe}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved scaler to {scaler_path}")
    
    # 3. Form Dataset & split
    full_dataset = NiftyFusionDataset(prep_df, scaled_feats, labels, seq_len=seq_len)
    
    train_size = int(len(full_dataset) * config['data']['train_split'])
    val_size = len(full_dataset) - train_size
    
    # Create sequential splits (avoid random split for time series)
    train_dataset = torch.utils.data.Subset(full_dataset, range(0, train_size))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_size, len(full_dataset)))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Dataset summary: Train size = {train_size}, Validation size = {val_size}")
    
    # 4. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() and config['system']['device'] == "auto" else "cpu")
    if config['system']['device'] in ["cuda", "cpu"]:
        device = torch.device(config['system']['device'])
    logger.info(f"Using training device: {device}")
    
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
    
    # Calculate dynamic class weights for training set to counter class imbalance (soft square root weights)
    train_y = labels[seq_len : train_size + seq_len]
    class_counts = np.bincount(train_y, minlength=3)
    class_counts = np.clip(class_counts, 1, None)
    weights = np.sqrt(len(train_y) / (3.0 * class_counts))
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class counts in training labels: {class_counts.tolist()}, Computed soft dynamic weights: {weights.tolist()}")
    
    # 5. Loss, Optimizer, and Scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['model']['training']['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 6. Training Loop
    best_val_loss = float('inf')
    checkpoint_dir = Path(config['model']['training']['checkpoint_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Starting model training loop...")
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for seq_batch, img_batch, label_batch in train_loader:
            seq_batch, img_batch, label_batch = seq_batch.to(device), img_batch.to(device), label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(seq_batch, img_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * seq_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += label_batch.size(0)
            correct_train += (predicted == label_batch).sum().item()
            
        epoch_train_loss = running_loss / train_size
        epoch_train_acc = correct_train / total_train
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for seq_batch, img_batch, label_batch in val_loader:
                seq_batch, img_batch, label_batch = seq_batch.to(device), img_batch.to(device), label_batch.to(device)
                outputs = model(seq_batch, img_batch)
                loss = criterion(outputs, label_batch)
                
                running_val_loss += loss.item() * seq_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += label_batch.size(0)
                correct_val += (predicted == label_batch).sum().item()
                
        epoch_val_loss = running_val_loss / val_size
        epoch_val_acc = correct_val / total_val
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        # Call learning rate scheduler
        scheduler.step(epoch_val_loss)
        lr_curr = optimizer.param_groups[0]['lr']
        
        logger.info(f"Epoch {epoch+1}/{epochs} [LR: {lr_curr:.6f}] - "
                    f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2%}, "
                    f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2%}")
        
        # Save checkpoints
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': epoch_val_loss
        }, checkpoint_path)
        
        # Save best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            model_path = save_dir / f"fusion_model_{timeframe}.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Saved new best model checkpoint to {model_path} with Val Loss: {best_val_loss:.4f}")
            
    # Save training history metrics as metadata
    metrics_path = save_dir / f"training_metrics_{timeframe}.pkl"
    with open(metrics_path, 'wb') as f:
        pickle.dump(history, f)
        
    logger.info("Training completed.")
    return history

if __name__ == "__main__":
    train_model(timeframe="15m")
