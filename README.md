# AI-Based NIFTY 50 Candlestick Prediction System

This is a production-ready, modular, and scalable Python application that predicts short-term movements of the NIFTY 50 Index (Bullish, Bearish, or Sideways) with confidence metrics. It uses a **hybrid multimodal fusion AI architecture** that combines sequence features (LSTM) and visual chart features (CNN), alongside a **Computer Vision (OpenCV)** screenshot parser to read uploaded trading screenshots directly.

---

## Key Features

1. **Multimodal Fusion Engine**: Joins PyTorch LSTM sequential representations (based on historical OHLCV data & indicators) with PyTorch CNN representations (based on charts).
2. **OpenCV Screenshot Analyzer**: Takes any uploaded chart screenshot, detects candle bodies (Bullish/Bearish segmentation), calculates wicks, estimates local prices, and overlays support/resistance levels, trend regression lines, and visual candlestick patterns.
3. **Comprehensive Indicators & Patterns**: Computes standard indices (RSI, MACD, EMA, SMA, VWAP, Bollinger Bands, ATR) and detects 14 candlestick patterns (Doji, Hammer, Engulfing, Stars, Harami, Marubozu, Tweezers, Soldiers, Crows).
4. **Trading Strategy Backtester**: Simulates historical trading outcomes using model predictions, ATR-based risk management (SL and Target zones), and computes return percentages, win rates, Sharpe ratios, and drawdowns.
5. **Streamlit Dark Dashboard**: A premium, trading-dashboard interface for uploading screenshots, visualizing predictions/visual overlays, running evaluations, and backtesting.
6. **Continuous Learning Loop**: Users can submit screenshots alongside actual outcomes. The system caches these locally and supports online fine-tuning on the user-feedback database.

---

## Directory Structure

```text
├── data/                    # Downloaded NIFTY 50 CSVs & feedback datasets
├── models/                  # Saved weights (fusion_model_*.pt) & scalers (scaler_*.pkl)
├── checkpoints/             # Intermediary training checkpoints
├── training/                # Train pipelines & evaluator
│   ├── train_fusion.py      # Core multimodal PyTorch trainer
│   └── evaluate.py          # Metric validation & confusion matrix plotting
├── inference/               # Production prediction models
│   ├── predict.py           # End-to-end predictor (placeholder for direct calls)
│   └── cv_analyzer.py       # OpenCV candle detection and visual overlays
├── ui/                      # Dashboard UI
│   └── app.py               # Streamlit trading dashboard
├── utils/                   # Shared utility modules
│   ├── config.py            # yaml configurations loader
│   ├── data_loader.py       # yfinance downloading & preprocessing
│   ├── indicators.py        # Vectorized math for technical indicators & patterns
│   └── backtester.py        # Trading strategy backtest engine
├── config.yaml              # Global hyperparameters, timeframes, model architectures
├── requirements.txt         # Package dependencies
└── run_pipeline.py          # Unified CLI entry-point
```

---

## System Architecture Details

### 1. Part 1: Time-Series Model (LSTM)
The LSTM sequence branch ingests a rolling window of 30 past candles consisting of 35 engineered indicators and candlestick pattern flags (e.g. RSI, MACD histograms, EMA ratios, Bollinger spreads, Doji flags). The network processes this sequential stream, outputting a 64-dimensional temporal embedding.

### 2. Part 2: Image Recognition Model (CNN)
The CNN branch processes the chart image. During training, historical data is plotted using a high-speed, in-memory OpenCV chart renderer that draws candles onto a `224x224` dark canvas. The image is passed through a deep convolutional network (ResNet18 or custom CNN blocks) to output a 128-dimensional visual embedding.

### 3. Part 3: Fusion Layer
The 192-dimensional combined embedding is concatenated, passed through dense layers with Batch Normalization and Dropout ($30\%$), and mapped to 3 logits corresponding to predicted market direction: `Bullish`, `Bearish`, or `Sideways`.

### 4. Computer Vision Screenshot Parsing
When a user uploads an arbitrary trading chart screenshot:
- **Segmentation**: The image is filtered using custom RGB/BGR masks. Green pixels indicate bullish candles, and Red pixels indicate bearish candles.
- **Bounding Boxes**: Contours are found for each candle body rectangle.
- **Chronological Ordering**: Detected shapes are sorted left-to-right to establish time progression.
- **Price Normalization**: The vertical span of candle bodies is scaled from `0` to `100` to establish relative prices. Open, High, Low, and Close values are reconstructed by matching bounding boxes and scanning local columns for candle wicks.
- **Support & Resistance**: Pivot points and local horizontal clusters are located and drawn as dashed horizontal overlays.
- **Trendline**: Linear regression fits a line through the centers of the candles to estimate trend slope and strength.

---

## Getting Started

### Installation

1. Clone or navigate to the workspace directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Standard fallback calculations are included, so it does not require complex C++ compilers on Windows for technical indicators.*

### Running the System

You can run the entire pipeline via `run_pipeline.py` or use the dashboard directly.

1. **Download NIFTY 50 Data**:
   ```bash
   python run_pipeline.py --download --timeframe 15m
   ```

2. **Train the Fusion Model**:
   ```bash
   python run_pipeline.py --train --timeframe 15m
   ```

3. **Evaluate Model Accuracy & Confusion Matrix**:
   ```bash
   python run_pipeline.py --evaluate --timeframe 15m
   ```

4. **Run Historical Strategy Backtester**:
   ```bash
   python run_pipeline.py --backtest --timeframe 15m
   ```

5. **Launch the Premium UI Dashboard**:
   ```bash
   python run_pipeline.py --dashboard
   ```
   This starts the Streamlit dashboard on your local machine (usually at `http://localhost:8501`).

---

## Continuous Learning Loop
To improve predictions:
1. Upload a chart screenshot in the **Continuous Learning & Feedback** tab.
2. Select the actual outcome (Bullish, Bearish, Sideways) that occurred next.
3. Click **Submit Outcome**.
4. To fine-tune the model, click **Fine-tune Model with Feedback Data**. This triggers online backpropagation updates of the CNN weights based on user submissions.
