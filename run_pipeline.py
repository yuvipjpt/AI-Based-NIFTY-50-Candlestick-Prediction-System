import os
import sys
import argparse
import subprocess
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("run_pipeline")

def parse_args():
    parser = argparse.ArgumentParser(description="NIFTY 50 Candlestick Prediction System Pipeline")
    parser.add_argument('--download', action='store_true', help="Download NIFTY 50 data")
    parser.add_argument('--train', action='store_true', help="Train the LSTM-CNN Fusion Model")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate the trained model performance")
    parser.add_argument('--backtest', action='store_true', help="Run the trading strategy backtester")
    parser.add_argument('--dashboard', action='store_true', help="Start the Streamlit dashboard")
    parser.add_argument('--timeframe', type=str, default="15m", 
                        choices=["1m", "5m", "15m", "1h", "1d"], 
                        help="Specify the NIFTY 50 timeframe (default: 15m)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # If no flags are set, show help
    if not (args.download or args.train or args.evaluate or args.backtest or args.dashboard):
        logger.info("No action arguments provided. Launching Streamlit dashboard by default...")
        args.dashboard = True
        
    project_root = Path(__file__).resolve().parent
    sys.path.append(str(project_root))
    
    # Imports
    from utils.config import CONFIG
    from utils.data_loader import download_nifty_data
    from training.train_fusion import train_model
    from training.evaluate import evaluate_model
    from utils.backtester import run_backtest
    
    # 1. Download
    if args.download:
        logger.info(f"Step 1: Downloading NIFTY 50 data for timeframe: {args.timeframe}")
        df = download_nifty_data(args.timeframe, force_download=True)
        logger.info(f"Download complete. Total candles: {len(df)}")
        
    # 2. Train
    if args.train:
        logger.info(f"Step 2: Training LSTM-CNN Fusion Model on {args.timeframe}")
        history = train_model(timeframe=args.timeframe)
        logger.info("Training complete.")
        
    # 3. Evaluate
    if args.evaluate:
        logger.info(f"Step 3: Evaluating model on {args.timeframe} dataset")
        eval_res = evaluate_model(timeframe=args.timeframe)
        if eval_res:
            logger.info(f"Evaluation finished. Accuracy: {eval_res['accuracy']:.2%}")
            
    # 4. Backtest
    if args.backtest:
        logger.info(f"Step 4: Running trading backtest on {args.timeframe}")
        backtest_res = run_backtest(timeframe=args.timeframe)
        if "error" not in backtest_res:
            logger.info(f"Backtest complete. Return: {backtest_res['total_return_pct']:.2f}%, Win Rate: {backtest_res['win_rate']:.2f}%")
            
    # 5. Start Dashboard
    if args.dashboard:
        logger.info("Step 5: Starting Streamlit Dashboard...")
        app_path = project_root / "ui" / "app.py"
        try:
            # Run Streamlit as a subprocess
            subprocess.run(["streamlit", "run", str(app_path)], check=True)
        except KeyboardInterrupt:
            logger.info("Dashboard shutdown by user request.")
        except Exception as e:
            logger.error(f"Error launching Streamlit dashboard: {e}")

if __name__ == "__main__":
    main()
