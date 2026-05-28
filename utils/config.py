import os
import yaml
import logging
from pathlib import Path

# Base Directory definition
BASE_DIR = Path(__file__).resolve().parent.parent

def load_config(config_path=None):
    """Loads configuration from config.yaml."""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def setup_logger(name="trading_system", log_file="system.log", level=logging.INFO):
    """Sets up system-wide logger."""
    config = load_config()
    log_dir = Path(config.get("system", {}).get("log_dir", "./logs"))
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # File Handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Global settings
try:
    CONFIG = load_config()
    # Create directory structure
    for path_key in ["cache_dir"]:
        path_val = CONFIG.get("data", {}).get(path_key, "")
        if path_val:
            os.makedirs(Path(BASE_DIR) / path_val, exist_ok=True)
            
    for path_key in ["feedback_dir"]:
        path_val = CONFIG.get("system", {}).get(path_key, "")
        if path_val:
            os.makedirs(Path(BASE_DIR) / path_val, exist_ok=True)
            
    for path_key in ["save_dir", "checkpoint_dir"]:
        path_val = CONFIG.get("model", {}).get("training", {}).get(path_key, "")
        if path_val:
            os.makedirs(Path(BASE_DIR) / path_val, exist_ok=True)
            
    log_dir = CONFIG.get("system", {}).get("log_dir", "./logs")
    os.makedirs(Path(BASE_DIR) / log_dir, exist_ok=True)
except Exception as e:
    print(f"Error initializing directories/configurations: {e}")
