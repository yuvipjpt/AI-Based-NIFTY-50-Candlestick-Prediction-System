import torch
import torch.nn as nn
import torchvision.models as models
import logging
from utils.config import setup_logger

logger = setup_logger("models", "system.log")

class LSTMNetwork(nn.Module):
    """LSTM model to process sequential market indicator features."""
    def __init__(self, input_size, hidden_dim=64, num_layers=2, dropout=0.2, feature_dim=64):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, seq_len, input_size]
        out, (hn, cn) = self.lstm(x)
        # Use the hidden state of the last time step
        last_hidden = out[:, -1, :]
        features = self.relu(self.fc(last_hidden))
        return features


class CNNNetwork(nn.Module):
    """Lightweight CNN model for processing screenshot and chart images."""
    def __init__(self, backbone="resnet18", pretrained=True, feature_dim=128):
        super(CNNNetwork, self).__init__()
        self.backbone_name = backbone
        
        # Try loading torchvision backbone
        self.use_resnet = False
        try:
            if backbone == "resnet18":
                # Initialize ResNet18
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                resnet = models.resnet18(weights=weights)
                # Remove final fully connected layer
                self.cnn = nn.Sequential(*list(resnet.children())[:-1])
                self.fc = nn.Linear(resnet.fc.in_features, feature_dim)
                self.use_resnet = True
                logger.info("ResNet18 visual backbone initialized successfully.")
        except Exception as e:
            logger.warning(f"Could not load ResNet18 ({e}). Building custom CNN backbone.")
            
        if not self.use_resnet:
            # Custom simple CNN
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 112
                
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 56
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2), # 28
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)) # 1
            )
            self.fc = nn.Linear(128, feature_dim)
            
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: [batch, 3, 224, 224]
        out = self.cnn(x)
        out = torch.flatten(out, 1)
        features = self.relu(self.fc(out))
        return features


class MarketFusionModel(nn.Module):
    """
    Hybrid Multimodal Fusion model that joins numerical sequence representations
    and chart image representations to make classification predictions.
    """
    def __init__(self, lstm_input_size, lstm_hidden_dim=64, lstm_layers=2, 
                 cnn_backbone="resnet18", cnn_feature_dim=128, 
                 fusion_hidden_dim=64, num_classes=3, dropout=0.3):
        super(MarketFusionModel, self).__init__()
        
        lstm_feature_dim = 64
        
        # Sub-networks
        self.lstm_branch = LSTMNetwork(
            input_size=lstm_input_size,
            hidden_dim=lstm_hidden_dim,
            num_layers=lstm_layers,
            feature_dim=lstm_feature_dim
        )
        
        self.cnn_branch = CNNNetwork(
            backbone=cnn_backbone,
            pretrained=True,
            feature_dim=cnn_feature_dim
        )
        
        # Fusion Classifier
        combined_dim = lstm_feature_dim + cnn_feature_dim
        self.fusion_fc1 = nn.Linear(combined_dim, fusion_hidden_dim)
        self.bn1 = nn.BatchNorm1d(fusion_hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.output_layer = nn.Linear(fusion_hidden_dim, num_classes)
        
    def forward(self, seq_data, img_data):
        # seq_data: [batch, seq_len, lstm_input_size]
        # img_data: [batch, 3, 224, 224]
        
        # Extract features
        seq_feats = self.lstm_branch(seq_data)
        img_feats = self.cnn_branch(img_data)
        
        # Concatenate features
        fused = torch.cat((seq_feats, img_feats), dim=1)
        
        # Classification layers
        x = self.fusion_fc1(fused)
        if x.size(0) > 1: # Batch normalization fails on single-item batches
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        
        logits = self.output_layer(x)
        return logits
        
    def predict_probs(self, seq_data, img_data):
        """Helper to run prediction and return probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(seq_data, img_data)
            probs = torch.softmax(logits, dim=1)
        return probs
