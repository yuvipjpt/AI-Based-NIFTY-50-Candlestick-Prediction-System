import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from utils.config import setup_logger
from utils.indicators import detect_candlestick_patterns, identify_support_resistance

logger = setup_logger("cv_analyzer", "system.log")

class ChartCVAnalyzer:
    """Analyzes trading chart screenshots using computer vision."""
    def __init__(self):
        pass
        
    def analyze_screenshot(self, image_path):
        """
        Processes chart image, extracts candle structures, trends, and draws overlays.
        Returns:
            analyzed_image (np.ndarray): Image with bounding boxes and overlays.
            prediction_data (dict): Extracted metadata (trend, patterns, metrics).
        """
        # Load image
        if isinstance(image_path, (str, Path)):
            img = cv2.imread(str(image_path))
            if img is None:
                raise FileNotFoundError(f"Failed to load image at {image_path}")
        else:
            # Assume it's already an OpenCV image/numpy array
            img = image_path.copy()
            
        h_img, w_img, _ = img.shape
        logger.info(f"Loaded image of shape {img.shape} for analysis.")
        
        # Determine background type (dark or light)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        is_dark_theme = mean_brightness < 127
        logger.info(f"Theme detected: {'Dark' if is_dark_theme else 'Light'} (Mean Brightness: {mean_brightness:.2f})")
        
        # Create masks for red and green candles
        # In BGR: img[:,:,0]=Blue, img[:,:,1]=Green, img[:,:,2]=Red
        b, g, r = cv2.split(img)
        
        # Green Segment (Bullish): Green channel is significantly higher than red & blue
        # Adapt thresholds slightly based on dark/light backgrounds
        if is_dark_theme:
            green_mask = (g > r * 1.1) & (g > b * 1.1) & (g > 40)
        else:
            green_mask = (g > r * 1.05) & (g > b * 1.05) & (g > 30)
            
        # Red Segment (Bearish): Red channel is significantly higher than green & blue
        if is_dark_theme:
            red_mask = (r > g * 1.1) & (r > b * 1.1) & (r > 40)
        else:
            red_mask = (r > g * 1.05) & (r > b * 1.05) & (r > 30)
            
        # Convert boolean masks to uint8
        green_mask_uint8 = (green_mask * 255).astype(np.uint8)
        red_mask_uint8 = (red_mask * 255).astype(np.uint8)
        
        # Find contours
        contours_green, _ = cv2.findContours(green_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv2.findContours(red_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        raw_candles = []
        
        # Parse green contours
        for cnt in contours_green:
            x, y, w, h = cv2.boundingRect(cnt)
            # Filter noise (extremely small or wide shapes)
            if w > 2 and h > 2 and w < h_img * 0.1 and h < h_img * 0.8:
                raw_candles.append({
                    'bbox': (x, y, w, h),
                    'color': 'green',
                    'center_x': x + w/2,
                    'center_y': y + h/2,
                    'direction': 1
                })
                
        # Parse red contours
        for cnt in contours_red:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > 2 and h > 2 and w < h_img * 0.1 and h < h_img * 0.8:
                raw_candles.append({
                    'bbox': (x, y, w, h),
                    'color': 'red',
                    'center_x': x + w/2,
                    'center_y': y + h/2,
                    'direction': -1
                })
                
        logger.info(f"Detected {len(raw_candles)} potential candle structures.")
        
        if not raw_candles:
            # Fallback if detection fails entirely (e.g. black and white charts)
            logger.warning("No color candles detected. Using grayscale contours fallback.")
            edges = cv2.Canny(gray, 50, 150)
            contours_all, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours_all:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 3 and h > 5 and w < h_img * 0.05:
                    raw_candles.append({
                        'bbox': (x, y, w, h),
                        'color': 'green',  # Default
                        'center_x': x + w/2,
                        'center_y': y + h/2,
                        'direction': 1
                    })
                    
        if not raw_candles:
            return img, {"error": "No candles detected on image.", "trend": "Sideways", "strength": 0, "patterns": []}
            
        # Sort candles from left to right (chronologically)
        raw_candles.sort(key=lambda c: c['bbox'][0])
        
        # Merge overlapping candles horizontally (sometimes segmentation splits a single candle)
        merged_candles = []
        for candle in raw_candles:
            x, y, w, h = candle['bbox']
            if not merged_candles:
                merged_candles.append(candle)
                continue
            
            prev = merged_candles[-1]
            px, py, pw, ph = prev['bbox']
            
            # If they overlap significantly horizontally, merge them
            if x < px + pw * 0.8:
                # Merge bounding boxes
                nx = min(px, x)
                ny = min(py, y)
                nw = max(px + pw, x + w) - nx
                nh = max(py + ph, y + h) - ny
                prev['bbox'] = (nx, ny, nw, nh)
                prev['center_x'] = nx + nw/2
                prev['center_y'] = ny + nh/2
            else:
                merged_candles.append(candle)
                
        # Limit to the last 40 candles for clean representation and pattern logic
        merged_candles = merged_candles[-40:]
        
        # Find absolute coordinate bounds of the candle region to scale values
        all_ys = []
        for c in merged_candles:
            x, y, w, h = c['bbox']
            all_ys.extend([y, y + h])
            
        Y_min = min(all_ys) if all_ys else 0
        Y_max = max(all_ys) if all_ys else h_img
        Y_span = Y_max - Y_min + 1e-10
        
        # Reconstruct pseudo OHLCV data
        reconstructed_data = []
        overlay_img = img.copy()
        
        for idx, candle in enumerate(merged_candles):
            x, y, w, h = candle['bbox']
            
            # Draw candle body bounding box
            color_val = (0, 200, 0) if candle['color'] == 'green' else (0, 0, 220)
            cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color_val, 2)
            
            # Estimate wicks: look vertically for lines above and below
            # We check the column near center_x
            cx = int(candle['center_x'])
            wick_top = y
            wick_bottom = y + h
            
            # Scan upwards for wick pixel edges (Canny / threshold matches)
            # Find the top-most non-background pixel in the vicinity of the column
            col_x = np.clip(cx, 0, w_img - 1)
            column_slice = gray[max(0, y - int(h*2)) : y, col_x]
            if len(column_slice) > 0:
                # Find indices where grayscale differs from background
                bg_val = gray[0, col_x] # Approximate background
                diffs = np.where(np.abs(column_slice.astype(int) - int(bg_val)) > 15)[0]
                if len(diffs) > 0:
                    wick_top = y - (len(column_slice) - diffs[0])
            
            # Scan downwards
            column_slice_down = gray[y + h : min(h_img, y + h + int(h*2)), col_x]
            if len(column_slice_down) > 0:
                bg_val = gray[0, col_x]
                diffs = np.where(np.abs(column_slice_down.astype(int) - int(bg_val)) > 15)[0]
                if len(diffs) > 0:
                    wick_bottom = y + h + diffs[-1]
            
            # Draw wick lines
            cv2.line(overlay_img, (cx, int(wick_top)), (cx, y), color_val, 1)
            cv2.line(overlay_img, (cx, y + h), (cx, int(wick_bottom)), color_val, 1)
            
            # Map pixel Y-coordinates to normalized price (0 to 100)
            # Y-pixel coordinates are inverted (0 at top, h_img at bottom)
            p_high = 100.0 * (Y_max - wick_top) / Y_span
            p_low = 100.0 * (Y_max - wick_bottom) / Y_span
            
            if candle['color'] == 'green':
                p_open = 100.0 * (Y_max - (y + h)) / Y_span
                p_close = 100.0 * (Y_max - y) / Y_span
            else:
                p_open = 100.0 * (Y_max - y) / Y_span
                p_close = 100.0 * (Y_max - (y + h)) / Y_span
                
            # Volume is estimated roughly by candle body area
            p_volume = float(w * h)
            
            reconstructed_data.append({
                'open': p_open,
                'high': p_high,
                'low': p_low,
                'close': p_close,
                'volume': p_volume
            })
            
        # Create DataFrame
        df_candles = pd.DataFrame(reconstructed_data)
        
        # Calculate Trend Line using Linear Regression on centers
        centers_y = np.array([c['center_y'] for c in merged_candles])
        centers_x = np.array([c['center_x'] for c in merged_candles])
        
        slope, intercept = np.polyfit(centers_x, centers_y, 1)
        
        # Draw Trend line
        start_x = int(centers_x[0])
        start_y = int(slope * start_x + intercept)
        end_x = int(centers_x[-1])
        end_y = int(slope * end_x + intercept)
        cv2.line(overlay_img, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2) # Yellow trendline
        
        # Classify trend direction (pixel-Y is inverted)
        # Positive slope in pixels means going down in price, negative slope means going up in price.
        slope_pct = (start_y - end_y) / Y_span # positive is upward price
        if slope_pct > 0.05:
            trend = "Bullish"
        elif slope_pct < -0.05:
            trend = "Bearish"
        else:
            trend = "Sideways"
            
        strength = min(100, int(abs(slope_pct) * 200))
        
        # Detect support/resistance on reconstructed coordinates
        supports, resistances = identify_support_resistance(df_candles, window=min(5, len(df_candles)//4))
        
        # Map normalized support/resistances back to pixels and draw them
        # normalized_price = 100.0 * (Y_max - y) / Y_span  => y = Y_max - (normalized_price * Y_span / 100.0)
        detected_sr_levels = []
        for s in supports:
            y_pixel = int(Y_max - (s * Y_span / 100.0))
            cv2.line(overlay_img, (0, y_pixel), (w_img, y_pixel), (255, 0, 0), 1, cv2.LINE_AA) # Blue support
            detected_sr_levels.append(s)
            
        for r in resistances:
            y_pixel = int(Y_max - (r * Y_span / 100.0))
            cv2.line(overlay_img, (0, y_pixel), (w_img, y_pixel), (255, 100, 0), 1, cv2.LINE_AA) # Light blue resistance
            detected_sr_levels.append(r)
            
        # Detect candlestick patterns
        detected_patterns = []
        if len(df_candles) >= 5:
            patterns_df = detect_candlestick_patterns(df_candles)
            
            # Scan recent 3 candles for patterns
            recent_rows = patterns_df.tail(3)
            
            pattern_columns = [col for col in patterns_df.columns if col.startswith("pattern_")]
            
            for col in pattern_columns:
                pattern_name = col.replace("pattern_", "").replace("_", " ").title()
                if (recent_rows[col] == 1).any():
                    detected_patterns.append(f"Bullish {pattern_name}")
                elif (recent_rows[col] == -1).any() or (col == "pattern_shooting_star" and (recent_rows[col] == 1).any()):
                    # Shooting star is bearish
                    if col == "pattern_shooting_star":
                        detected_patterns.append(f"Bearish {pattern_name}")
                    else:
                        detected_patterns.append(f"Bearish {pattern_name}")
                elif col in ["pattern_doji", "pattern_harami"] and (recent_rows[col] != 0).any():
                    # Neutral / consolidation patterns
                    detected_patterns.append(pattern_name)
                    
        # Remove duplicates
        detected_patterns = list(set(detected_patterns))
        
        # Put text overlay on image
        y_text = 30
        cv2.putText(overlay_img, f"Trend: {trend} (Str: {strength}%)", (20, y_text), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        for pat in detected_patterns[:3]: # Show top 3
            y_text += 25
            cv2.putText(overlay_img, f"Pattern: {pat}", (20, y_text), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                        
        prediction_data = {
            "trend": trend,
            "strength": strength,
            "patterns": detected_patterns,
            "candles_count": len(merged_candles),
            "reconstructed_candles": df_candles.to_dict(orient="records"),
            "theme": "Dark" if is_dark_theme else "Light"
        }
        
        return overlay_img, prediction_data
