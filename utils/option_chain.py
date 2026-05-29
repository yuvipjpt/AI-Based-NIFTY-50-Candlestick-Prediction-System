import requests
import numpy as np
import pandas as pd
import logging
from utils.config import setup_logger

logger = setup_logger("option_chain", "system.log")

def get_nifty_option_chain_data(spot_price):
    """
    Fetches option chain data for NIFTY from NSE India website.
    If requests are rate-limited or blocked, it falls back to a realistic mock data generator.
    """
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en-US,en;q=0.9',
        'referer': 'https://www.nseindia.com/option-chain'
    }

    try:
        session = requests.Session()
        # Visit home page first to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("Successfully fetched Option Chain from NSE India API.")
            return parse_nse_option_chain(data)
        else:
            logger.warning(f"NSE India returned status code {response.status_code}. Reverting to Mock Generator.")
    except Exception as e:
        logger.warning(f"Failed to fetch Option Chain from NSE: {e}. Reverting to Mock Generator.")
        
    # Revert to Mock Option Chain if scraping fails
    return generate_mock_option_chain(spot_price)


def parse_nse_option_chain(data):
    """Parses raw NSE option chain JSON response."""
    try:
        raw_data = data['filtered']['data']
        records = []
        
        total_call_oi = 0
        total_put_oi = 0
        
        # Max pain tracking variables
        strikes = []
        
        for item in raw_data:
            strike = item['strikePrice']
            strikes.append(strike)
            
            call_oi = item.get('CE', {}).get('openInterest', 0) if 'CE' in item else 0
            put_oi = item.get('PE', {}).get('openInterest', 0) if 'PE' in item else 0
            
            total_call_oi += call_oi
            total_put_oi += put_oi
            
            records.append({
                'strike': strike,
                'call_oi': call_oi,
                'put_oi': put_oi,
                'call_volume': item.get('CE', {}).get('totalTradedVolume', 0) if 'CE' in item else 0,
                'put_volume': item.get('PE', {}).get('totalTradedVolume', 0) if 'PE' in item else 0,
            })
            
        df = pd.DataFrame(records)
        
        # Calculate Put-Call Ratio (PCR)
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
        
        # Calculate Support & Resistance walls based on max OI
        max_put_idx = df['put_oi'].idxmax()
        max_call_idx = df['call_oi'].idxmax()
        
        support_wall = float(df.loc[max_put_idx, 'strike'])
        resistance_wall = float(df.loc[max_call_idx, 'strike'])
        
        # Calculate Max Pain Strike
        max_pain = calculate_max_pain(df, strikes)
        
        return {
            "pcr": round(pcr, 3),
            "max_pain": max_pain,
            "support_wall": support_wall,
            "resistance_wall": resistance_wall,
            "total_call_oi": total_call_oi,
            "total_put_oi": total_put_oi,
            "chain_df": df,
            "source": "NSE India API"
        }
    except Exception as e:
        logger.error(f"Error parsing NSE option chain: {e}")
        # Return fallback mock if parser fails
        return generate_mock_option_chain(22000.0)


def calculate_max_pain(df, strikes):
    """Calculates Max Pain strike where option buyers experience maximum loss."""
    min_loss = float('inf')
    best_strike = strikes[len(strikes) // 2] if strikes else 0.0
    
    for target_strike in strikes:
        loss = 0
        for _, row in df.iterrows():
            strike = row['strike']
            call_oi = row['call_oi']
            put_oi = row['put_oi']
            
            # Call option payout if price settles at target_strike
            if target_strike > strike:
                loss += call_oi * (target_strike - strike)
                
            # Put option payout if price settles at target_strike
            if target_strike < strike:
                loss += put_oi * (strike - target_strike)
                
        if loss < min_loss:
            min_loss = loss
            best_strike = target_strike
            
    return float(best_strike)


def generate_mock_option_chain(spot_price):
    """Generates a realistic mock NIFTY option chain centered around spot price."""
    # Round spot price to nearest 50
    center_strike = int(round(spot_price / 50.0) * 50)
    
    # Generate strikes around center
    strikes = [center_strike + i * 50 for i in range(-8, 9)]
    
    records = []
    
    # We want to model higher Call OI above center, higher Put OI below center
    np.random.seed(int(spot_price) % 10000) # seed based on current price for consistency
    
    total_call_oi = 0
    total_put_oi = 0
    
    for strike in strikes:
        # Distance from center
        dist = (strike - center_strike) / 50.0
        
        # Normal distribution centered at +1 strike for calls, -1 strike for puts
        call_factor = np.exp(-((dist - 1.5) ** 2) / 10.0)
        put_factor = np.exp(-((dist + 1.5) ** 2) / 10.0)
        
        call_oi = int((call_factor * 150000 + np.random.randint(5000, 20000)))
        put_oi = int((put_factor * 135000 + np.random.randint(5000, 20000)))
        
        # Scale volume accordingly
        call_vol = int(call_oi * np.random.uniform(0.5, 1.5))
        put_vol = int(put_oi * np.random.uniform(0.5, 1.5))
        
        total_call_oi += call_oi
        total_put_oi += put_oi
        
        records.append({
            'strike': strike,
            'call_oi': call_oi,
            'put_oi': put_oi,
            'call_volume': call_vol,
            'put_volume': put_vol
        })
        
    df = pd.DataFrame(records)
    
    # Calculate Put-Call Ratio (PCR)
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0.95
    
    max_put_idx = df['put_oi'].idxmax()
    max_call_idx = df['call_oi'].idxmax()
    
    support_wall = float(df.loc[max_put_idx, 'strike'])
    resistance_wall = float(df.loc[max_call_idx, 'strike'])
    
    max_pain = calculate_max_pain(df, strikes)
    
    return {
        "pcr": round(pcr, 3),
        "max_pain": max_pain,
        "support_wall": support_wall,
        "resistance_wall": resistance_wall,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "chain_df": df,
        "source": "Mock Generator (NSE Blocked)"
    }
