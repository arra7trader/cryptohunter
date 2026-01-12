"""
CryptoHunter API Server V3
==========================
Backend untuk CryptoHunter Web Interface.
Menyediakan data DexScreener + Analisis AI via HTTP JSON API.
Features: Real-time Scanning, Watchlist, Portfolio, Alert System
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import threading
import time
import json
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

# Import existing modules
from modules.dex_api import DexScreenerAPI, get_historical_data
from modules.sna_analyzer import SNAAnalyzer
from modules.price_predictor import analyze_token_comprehensive
from modules.db import db

app = FastAPI(
    title="CryptoHunter API",
    description="🚀 Advanced AI-Powered Crypto Scanner & Prediction Engine",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
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
        self.scan_progress = 100
        self.scan_status = "Idle"
        self.price_history = defaultdict(list)  # For sparkline charts
        self.alerts = []  # Price alerts

data_cache = DataCache()

# Watchlist storage (in production, use DB)
watchlist_storage = {}  # {user_id: [token_addresses]}
portfolio_storage = {}  # {user_id: {token_address: {amount, buy_price}}}

# WebSocket connections for real-time updates
active_connections: List[WebSocket] = []

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
    sparkline: Optional[List[float]] = None  # Mini chart data
    is_watchlisted: bool = False

class MarketSummary(BaseModel):
    total_volume: float
    avg_change: float
    gainers: int
    losers: int
    active_tokens: int
    scan_progress: int = 100
    scan_status: str = "Idle"
    top_gainer: Optional[Dict] = None
    top_loser: Optional[Dict] = None
    market_sentiment: str = "neutral"

class WatchlistItem(BaseModel):
    token_address: str
    chain: str
    added_at: Optional[str] = None

class PortfolioItem(BaseModel):
    token_address: str
    chain: str
    amount: float
    buy_price: float
    
class AlertCreate(BaseModel):
    token_address: str
    chain: str
    target_price: float
    alert_type: str  # "above" or "below"

# --- Background Tasks ---

def update_market_cache():
    """Background task to fetch and update market data"""
    if data_cache.is_updating:
        return

    try:
        data_cache.is_updating = True
        data_cache.scan_progress = 10
        data_cache.scan_status = "Fetching from DexScreener"
        print("[API] Updating market cache...")
        
        # 1. Fetch tokens (Now optimized via ThreadPool)
        pairs_df = dex_api.search_new_pairs(min_liquidity=5000)
        
        data_cache.scan_progress = 30
        
        if pairs_df.empty:
            print("[API] No pairs found")
            data_cache.is_updating = False
            data_cache.scan_progress = 100
            data_cache.scan_status = "Idle"
            return

        # Sort by volume and take top 50 (Display ALL first)
        pairs_df = pairs_df.sort_values('volume_1h', ascending=False).head(50)
        
        # === STAGE 1: RAW DATA (IMMEDIATE DISPLAY) ===
        data_cache.scan_status = "Processing Raw Data"
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
                    "source": "scanning..."
                },
                "status": "SCANNING"
            }
            raw_data.append(token_data)
        
        # Update Cache IMMEDIATELY with Raw Data
        with cache_lock:
            data_cache.market_data = raw_data
            data_cache.last_update = time.time()
            data_cache.scan_progress = 40
            
        print(f"[API] Stage 1: Displaying {len(raw_data)} tokens (Raw)")

        # === STAGE 2: ENRICHMENT (SNA + ANALYSIS) ===
        # Now we process SNA and Logic in the background
        data_cache.scan_status = "Running SNA & AI Analysis"
        
        sna_results = sna_analyzer.analyze_batch(pairs_df)
        data_cache.scan_progress = 60
        
        enriched_data = []
        total_items = len(raw_data)

        for i, item in enumerate(raw_data):
            # Slow down slightly just to show progress or just process
            # Realistically this is fast, but let's update progress per batch or item
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
            
            if vol / max(1, liq) > 2:
                conf += 10
            
            final_conf = int(min(95, max(40, conf)))
            
            item['prediction'] = {
                "confidence": final_conf,
                "pump_in_hours": 2 if sna_score > 60 else 6,
                "source": "sna_fast"
            }
            
            if item['price_change_5m'] > 5: item['status'] = "PUMP"
            elif item['price_change_5m'] < -5: item['status'] = "DUMP"
            else: item['status'] = "NEUTRAL"
            
            enriched_data.append(item)
            
            # Update progress dynamically between 60 and 90
            current_progress = 60 + int((i / total_items) * 30)
            data_cache.scan_progress = current_progress

        # Update Cache AGAIN with Enriched Data
        with cache_lock:
            data_cache.market_data = enriched_data
            data_cache.last_update = time.time()
            data_cache.scan_progress = 100
            data_cache.scan_status = "Idle"
            
        print(f"[API] Stage 2: Enriched {len(enriched_data)} tokens with SNA")
        
    except Exception as e:
        print(f"[API] Error updating cache: {e}")
        import traceback
        traceback.print_exc()
        data_cache.scan_progress = 0
        data_cache.scan_status = "Error"
    finally:
        data_cache.is_updating = False

# --- Endpoints ---

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "message": "🚀 CryptoHunter API V3 Running",
        "version": "3.0.0",
        "endpoints": {
            "docs": "/docs",
            "tokens": "/api/tokens",
            "summary": "/api/market/summary",
            "watchlist": "/api/watchlist",
            "trending": "/api/trending",
            "predict": "/api/tokens/{address}/predict"
        }
    }

@app.get("/api/market/summary")
def get_market_summary():
    """Get overall market statistics from cached data"""
    if not data_cache.market_data:
        # Trigger update if empty
        threading.Thread(target=update_market_cache).start()
        return {
            "total_volume": 0, "avg_change": 0, "gainers": 0, "losers": 0, 
            "active_tokens": 0, "scan_progress": data_cache.scan_progress, 
            "scan_status": data_cache.scan_status, "top_gainer": None,
            "top_loser": None, "market_sentiment": "neutral"
        }
    
    df = pd.DataFrame(data_cache.market_data)
    
    # Calculate top gainer and loser
    top_gainer = None
    top_loser = None
    if not df.empty:
        gainer_idx = df['price_change_1h'].idxmax()
        loser_idx = df['price_change_1h'].idxmin()
        top_gainer = {"token": df.loc[gainer_idx, 'token'], "change": df.loc[gainer_idx, 'price_change_1h']}
        top_loser = {"token": df.loc[loser_idx, 'token'], "change": df.loc[loser_idx, 'price_change_1h']}
    
    # Calculate market sentiment
    avg_change = df['price_change_1h'].mean()
    sentiment = "bullish" if avg_change > 2 else "bearish" if avg_change < -2 else "neutral"
    
    return {
        "total_volume": float(df['volume_1h'].sum()),
        "avg_change": float(df['price_change_1h'].mean()),
        "gainers": int(len(df[df['price_change_1h'] > 0])),
        "losers": int(len(df[df['price_change_1h'] < 0])),
        "active_tokens": len(df),
        "scan_progress": data_cache.scan_progress,
        "scan_status": data_cache.scan_status,
        "top_gainer": top_gainer,
        "top_loser": top_loser,
        "market_sentiment": sentiment
    }

@app.get("/api/tokens")
async def get_tokens(refresh: bool = False, sort_by: str = "volume_1h", limit: int = 50):
    """Get list of hot tokens with sorting options"""
    # Auto refresh if stale (> 30s)
    if refresh or (time.time() - data_cache.last_update > 30):
        threading.Thread(target=update_market_cache).start()
    
    with cache_lock:
        tokens = data_cache.market_data.copy()
    
    # Add sparkline data to each token
    for token in tokens:
        addr = token.get('token_address', '')
        if addr in data_cache.price_history:
            token['sparkline'] = data_cache.price_history[addr][-20:]  # Last 20 prices
        else:
            token['sparkline'] = []
    
    # Sort tokens
    if tokens and sort_by in ['volume_1h', 'price_change_5m', 'price_change_1h', 'sna_score', 'liquidity_usd', 'market_cap']:
        tokens = sorted(tokens, key=lambda x: x.get(sort_by, 0), reverse=True)
    
    return tokens[:limit]

@app.get("/api/trending")
async def get_trending():
    """Get trending tokens based on various metrics"""
    with cache_lock:
        tokens = data_cache.market_data.copy()
    
    if not tokens:
        return {"hot": [], "gainers": [], "new": [], "ai_picks": []}
    
    df = pd.DataFrame(tokens)
    
    return {
        "hot": df.nlargest(5, 'volume_1h')[['token', 'chain', 'token_address', 'price_change_1h', 'volume_1h']].to_dict('records'),
        "gainers": df.nlargest(5, 'price_change_1h')[['token', 'chain', 'token_address', 'price_change_1h']].to_dict('records'),
        "ai_picks": df.nlargest(5, 'sna_score')[['token', 'chain', 'token_address', 'sna_score', 'prediction']].to_dict('records'),
        "losers": df.nsmallest(5, 'price_change_1h')[['token', 'chain', 'token_address', 'price_change_1h']].to_dict('records')
    }

# === WATCHLIST ENDPOINTS ===

@app.get("/api/watchlist")
async def get_watchlist(user_id: str = "default"):
    """Get user's watchlist"""
    watchlist = watchlist_storage.get(user_id, [])
    
    # Enrich with current token data
    enriched = []
    with cache_lock:
        for item in watchlist:
            for token in data_cache.market_data:
                if token['token_address'] == item['token_address']:
                    enriched.append({**token, **item, "is_watchlisted": True})
                    break
    
    return enriched

@app.post("/api/watchlist")
async def add_to_watchlist(item: WatchlistItem, user_id: str = "default"):
    """Add token to watchlist"""
    if user_id not in watchlist_storage:
        watchlist_storage[user_id] = []
    
    # Check if already exists
    for existing in watchlist_storage[user_id]:
        if existing['token_address'] == item.token_address:
            return {"status": "already_exists"}
    
    watchlist_storage[user_id].append({
        "token_address": item.token_address,
        "chain": item.chain,
        "added_at": datetime.now().isoformat()
    })
    
    return {"status": "added", "watchlist_count": len(watchlist_storage[user_id])}

@app.delete("/api/watchlist/{address}")
async def remove_from_watchlist(address: str, user_id: str = "default"):
    """Remove token from watchlist"""
    if user_id in watchlist_storage:
        watchlist_storage[user_id] = [
            w for w in watchlist_storage[user_id] if w['token_address'] != address
        ]
    return {"status": "removed"}

# === ALERTS ENDPOINTS ===

@app.get("/api/alerts")
async def get_alerts(user_id: str = "default"):
    """Get user's price alerts"""
    return [a for a in data_cache.alerts if a.get('user_id') == user_id]

@app.post("/api/alerts")
async def create_alert(alert: AlertCreate, user_id: str = "default"):
    """Create price alert"""
    alert_data = {
        "id": len(data_cache.alerts) + 1,
        "user_id": user_id,
        "token_address": alert.token_address,
        "chain": alert.chain,
        "target_price": alert.target_price,
        "alert_type": alert.alert_type,
        "created_at": datetime.now().isoformat(),
        "triggered": False
    }
    data_cache.alerts.append(alert_data)
    return alert_data

@app.delete("/api/alerts/{alert_id}")
async def delete_alert(alert_id: int):
    """Delete an alert"""
    data_cache.alerts = [a for a in data_cache.alerts if a.get('id') != alert_id]
    return {"status": "deleted"}

# === HISTORICAL DATA ENDPOINT ===

@app.get("/api/tokens/{address}/history")
async def get_token_history(address: str, chain: str = "solana", interval: str = "1h"):
    """Get historical price data for charts"""
    try:
        hist_data = get_historical_data(address, chain_id=chain)
        
        if hist_data.empty:
            # Generate mock data for new tokens
            return {
                "address": address,
                "chain": chain,
                "data": [],
                "message": "No historical data available for new token"
            }
        
        # Format for frontend charts
        chart_data = []
        for _, row in hist_data.iterrows():
            chart_data.append({
                "timestamp": row.get('timestamp', datetime.now().isoformat()),
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0))
            })
        
        return {
            "address": address,
            "chain": chain,
            "interval": interval,
            "data": chart_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
        sna_score = 50 
        
        token_symbol = token_info.get('baseToken', {}).get('symbol', 'UNKNOWN')
        
        # 4. Run Comprehensive Analysis
        from modules.price_predictor import analyze_token_comprehensive
        
        prediction = analyze_token_comprehensive(hist_data, token_info, sna_score)
        
        return {
            "token": token_symbol,
            "address": address,
            "prediction": {
                "ensemble": prediction
            },
            "last_price": hist_data['close'].iloc[-1] if not hist_data.empty else 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# === WEBSOCKET FOR REAL-TIME UPDATES ===

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time price updates"""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Send market data every 5 seconds
            with cache_lock:
                data = {
                    "type": "market_update",
                    "data": data_cache.market_data[:10],  # Top 10
                    "timestamp": datetime.now().isoformat()
                }
            await websocket.send_json(data)
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

# === SEARCH ENDPOINT ===

@app.get("/api/search")
async def search_tokens(q: str):
    """Search tokens by name or address"""
    with cache_lock:
        tokens = data_cache.market_data.copy()
    
    q_lower = q.lower()
    results = [
        t for t in tokens 
        if q_lower in t.get('token', '').lower() or 
           q_lower in t.get('token_address', '').lower()
    ]
    
    return results[:20]

# === HEALTH CHECK ===

@app.get("/api/health")
async def health_check():
    """API Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_age_seconds": int(time.time() - data_cache.last_update),
        "tokens_cached": len(data_cache.market_data),
        "is_scanning": data_cache.is_updating
    }

if __name__ == "__main__":
    import uvicorn
    # Start cache update in thread
    update_market_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
