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
from modules.db import db

app = FastAPI(
    title="CryptoHunter API",
    description="Backend API untuk CryptoHunter AI Dashboard",
    version="2.1.0"
)

# Initialize DB
@app.on_event("startup")
async def startup_event():
    db.connect()

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
        
        # 1. Fetch tokens (Now optimized via ThreadPool)
        pairs_df = dex_api.search_new_pairs(min_liquidity=5000)
        
        if pairs_df.empty:
            print("[API] No pairs found")
            data_cache.is_updating = False
            return

        # Sort by volume and take top 50 (Display ALL first)
        pairs_df = pairs_df.sort_values('volume_1h', ascending=False).head(50)
        
        # === STAGE 1: RAW DATA (IMMEDIATE DISPLAY) ===
        # We prepare the list with "SCANNING..." status first so user sees the tokens
        raw_data = []
        
        def safe_float(val):
            try:
                return float(val) if val is not None else 0.0
            except:
                return 0.0

        for idx, row in pairs_df.iterrows():
            token_symbol = str(row['base_token'])
            c5m = safe_float(row['price_change_5m'])
            
            # Basic Status based on price only
            status = "NEUTRAL"
            if c5m <= -10: status = "CRASH"
            elif c5m <= -5: status = "DUMP"
            elif c5m > 5: status = "PUMP"

            token_data = {
                "token": token_symbol,
                "chain": str(row['chain_id']).upper(),
                "token_address": str(row['base_token_address']),
                "price_usd": safe_float(row['price_usd']),
                "price_change_5m": c5m,
                "price_change_1h": safe_float(row['price_change_1h']),
                "volume_1h": safe_float(row['volume_1h']),
                "liquidity_usd": safe_float(row['liquidity_usd']),
                "market_cap": safe_float(row['market_cap']),
                "sna_score": 50.0, # Placeholder
                "prediction": {
                    "confidence": 0,
                    "pump_in_hours": 0,
                    "source": "scanning..." # Indicator for UI
                },
                "status": "SCANNING"
            }
            raw_data.append(token_data)
        
        # Update Cache IMMEDIATELY with Raw Data
        with cache_lock:
            data_cache.market_data = raw_data
            data_cache.last_update = time.time()
        print(f"[API] Stage 1: Displaying {len(raw_data)} tokens (Raw)")


        # === STAGE 2: ENRICHMENT (SNA + ANALYSIS) ===
        # Now we process SNA and Logic in the background
        
        sna_results = sna_analyzer.analyze_batch(pairs_df)
        enriched_data = []

        for item in raw_data:
            token_symbol = item['token']
            
            # Find SNA result
            sna_score = 50
            for r in sna_results:
                if r.token_symbol == token_symbol:
                    sna_score = r.sna_score
                    break
            
            item['sna_score'] = float(sna_score)
            
            # Update AI/Heuristic Prediction
            vol = item['volume_1h']
            liq = item['liquidity_usd']
            conf = 50 + (sna_score * 0.3)
            
            # Simple Heuristic
            if vol / max(1, liq) > 2:
                conf += 10
            
            final_conf = int(min(95, max(40, conf)))
            
            item['prediction'] = {
                "confidence": final_conf,
                "pump_in_hours": 2 if sna_score > 60 else 6,
                "source": "sna_fast"
            }
            
            # Final Status Update
            if item['price_change_5m'] > 5: item['status'] = "PUMP"
            elif item['price_change_5m'] < -5: item['status'] = "DUMP"
            else: item['status'] = "NEUTRAL"
            
            enriched_data.append(item)

        # Update Cache AGAIN with Enriched Data
        with cache_lock:
            data_cache.market_data = enriched_data
            data_cache.last_update = time.time()
            
        print(f"[API] Stage 2: Enriched {len(enriched_data)} tokens with SNA")
        
    except Exception as e:
        print(f"[API] Error updating cache: {e}")
        import traceback
        traceback.print_exc()
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
        # ALWAYS Run in background to prevent blocking
        # Frontend handles empty/stale state
        threading.Thread(target=update_market_cache).start()
    
    with cache_lock:
        return data_cache.market_data

@app.get("/api/tokens/{address}/predict")
async def predict_token(address: str, chain: str = "solana"):
    """
    Super-Ensemble Endpoint:
    Automatically selects between Deep Learning (if history exists)
    or Gem Hunter (if new/no history).
    """
    try:
        # 1. Get Historical Data (May be empty)
        hist_data = get_historical_data(address, chain_id=chain)
             
        # 2. Get Token Info for SNA/Metrics
        token_info = dex_api.get_token_info(address, chain_id=chain)
        if not token_info:
             raise HTTPException(status_code=404, detail="Token info not found")
        
        # 3. Calculate Real SNA Score (Simplified for speed)
        # In prod, you'd want to call the SNA module properly
        sna_score = 50 
        
        token_symbol = token_info.get('baseToken', {}).get('symbol', 'UNKNOWN')
        
        # 4. Run Comprehensive Analysis
        # Note: We replaced 'train_model' and 'predict_pump_time' with this unified call
        from modules.price_predictor import analyze_token_comprehensive
        
        prediction = analyze_token_comprehensive(hist_data, token_info, sna_score)
        
        return {
            "token": token_symbol,
            "address": address,
            "prediction": {
                "ensemble": prediction # Match existing frontend format
            },
            "last_price": hist_data['close'].iloc[-1] if not hist_data.empty else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start loop in thread
    update_market_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000)
