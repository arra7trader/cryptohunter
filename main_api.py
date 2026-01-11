"""
CryptoHunter API Server
=======================
Backend untuk CryptoHunter Web Interface.
Menyediakan data DexScreener + Analisis AI via HTTP JSON API.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import threading
import time
from datetime import datetime

# Import existing modules
from modules.dex_api import DexScreenerAPI, get_historical_data
from modules.sna_analyzer import SNAAnalyzer
from modules.price_predictor import train_model, predict_pump_time

app = FastAPI(
    title="CryptoHunter API",
    description="Backend API untuk CryptoHunter AI Dashboard",
    version="2.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
dex_api = DexScreenerAPI()
sna_analyzer = SNAAnalyzer()
lstm_models = {}
cache_lock = threading.Lock()

# In-memory cache for expensive operations
class DataCache:
    def __init__(self):
        self.market_data = []
        self.last_update = 0
        self.update_interval = 30  # Update every 30s
        self.is_updating = False

data_cache = DataCache()

# --- Models ---

class TokenPrediction(BaseModel):
    confidence: int
    pump_in_hours: Optional[float]
    source: str

class TokenData(BaseModel):
    token: str
    chain: str
    token_address: str
    price_usd: float
    price_change_5m: float
    price_change_1h: float
    volume_1h: float
    liquidity_usd: float
    market_cap: float
    sna_score: float
    prediction: Optional[TokenPrediction] = None
    status: str  # PUMP, DUMP, NEUTRAL

class MarketSummary(BaseModel):
    total_volume: float
    avg_change: float
    gainers: int
    losers: int
    active_tokens: int

# --- Background Tasks ---

def update_market_cache():
    """Background task to fetch and update market data"""
    if data_cache.is_updating:
        return

    try:
        data_cache.is_updating = True
        print("[API] Updating market cache...")
        
        # Fetch tokens
        pairs_df = dex_api.search_new_pairs(min_liquidity=5000)
        
        if pairs_df.empty:
            print("[API] No pairs found")
            data_cache.is_updating = False
            return

        # Sort by volume
        pairs_df = pairs_df.sort_values('volume_1h', ascending=False).head(30)
        
        new_data = []
        
        # Process each token (SNA + Light Analysis)
        # We don't run full LSTM for list view to keep it fast, only on detail view or specific request
        sna_results = sna_analyzer.analyze_batch(pairs_df)
        
        for idx, row in pairs_df.iterrows():
            token_symbol = row['base_token']
            
            # Match SNA result
            sna_score = 0
            for r in sna_results:
                if r.token_symbol == token_symbol:
                    sna_score = r.sna_score
                    break
            
            # Simple Status
            c5m = row['price_change_5m']
            status = "NEUTRAL"
            if c5m <= -10: status = "CRASH"
            elif c5m <= -5: status = "DUMP"
            elif c5m > 5: status = "PUMP"
            
            # Basic Prediction (Fast)
            # Full LSTM is triggered on demand
            conf = 50 + (sna_score * 0.3)
            if row['volume_1h'] / max(1, row['liquidity_usd']) > 2:
                conf += 10
            conf = int(min(95, max(40, conf)))
            
            token_data = {
                "token": token_symbol,
                "chain": str(row['chain_id']).upper(),
                "token_address": row['base_token_address'],
                "price_usd": float(row['price_usd']),
                "price_change_5m": float(row['price_change_5m']),
                "price_change_1h": float(row['price_change_1h']),
                "volume_1h": float(row['volume_1h']),
                "liquidity_usd": float(row['liquidity_usd']),
                "market_cap": float(row['market_cap']),
                "sna_score": float(sna_score),
                "prediction": {
                    "confidence": conf,
                    "pump_in_hours": 2 if sna_score > 50 else 6,
                    "source": "sna_fast"
                },
                "status": status
            }
            new_data.append(token_data)
            
        with cache_lock:
            data_cache.market_data = new_data
            data_cache.last_update = time.time()
            
        print(f"[API] Cache updated with {len(new_data)} tokens")
        
    except Exception as e:
        print(f"[API] Error updating cache: {e}")
    finally:
        data_cache.is_updating = False

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "CryptoHunter API V2 Running"}

@app.get("/api/market/summary", response_model=MarketSummary)
def get_market_summary():
    """Get overall market statistics from cached data"""
    if not data_cache.market_data:
        # Trigger update if empty
        threading.Thread(target=update_market_cache).start()
        return MarketSummary(total_volume=0, avg_change=0, gainers=0, losers=0, active_tokens=0)
    
    df = pd.DataFrame(data_cache.market_data)
    
    return MarketSummary(
        total_volume=df['volume_1h'].sum(),
        avg_change=df['price_change_1h'].mean(),
        gainers=len(df[df['price_change_1h'] > 0]),
        losers=len(df[df['price_change_1h'] < 0]),
        active_tokens=len(df)
    )

@app.get("/api/tokens", response_model=List[TokenData])
async def get_tokens(refresh: bool = False):
    """Get list of hot tokens"""
    # Auto refresh if stale (> 30s)
    if refresh or (time.time() - data_cache.last_update > 30):
        # Run in background if not forced to wait, but for now we wait if it's the first time
        if not data_cache.market_data:
             update_market_cache()
        else:
             threading.Thread(target=update_market_cache).start()
    
    with cache_lock:
        return data_cache.market_data

@app.get("/api/tokens/{address}/predict")
async def predict_token(address: str, chain: str = "solana"):
    """
    Deep Analysis endpoint: Runs LSTM model for specific token.
    This is heavier than the list view prediction.
    """
    try:
        # 1. Get Historical Data
        hist_data = get_historical_data(address, chain_id=chain)
        if hist_data.empty:
             raise HTTPException(status_code=404, detail="Historical data not found")
             
        # 2. Get Token Info for SNA
        token_info = dex_api.get_token_info(address, chain_id=chain)
        if not token_info:
             raise HTTPException(status_code=404, detail="Token info not found")
             
        # Mock SNA result from token info (since we need dataframe for batch, we do single extraction here)
        # For simplicity reusing SNA logic requires a dataframe, lets make a temp one
        # Or just trust the fast SNA score passed from frontend if available, 
        # but let's do a quick recalc or use default
        sna_score = 50 # Default if full scan not run
        
        # 3. Model Training / Inference
        model_key = token_info.get('baseToken', {}).get('symbol', address)
        
        # Check if we have a trained model
        if model_key in lstm_models:
             model = lstm_models[model_key]
        else:
             # Train new model
             model, _ = train_model(hist_data, epochs=100)
             lstm_models[model_key] = model
             
        # 4. Predict
        prediction = predict_pump_time(model, hist_data, sna_score)
        
        return {
            "token": model_key,
            "address": address,
            "prediction": prediction,
            "last_price": hist_data['close'].iloc[-1],
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start loop in thread
    update_market_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000)
