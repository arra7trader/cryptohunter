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
from modules.pump_predictor import predict_pump_dump, batch_predict
from modules.indodax_api import IndodaxAPI, get_indodax_market
from modules.indodax_forecaster import IndodaxAIForecaster, forecast_indodax_token
from modules.auto_trainer import auto_trainer, start_auto_training, stop_auto_training, get_training_status, force_train_coin
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
    # Start auto-training in background
    print("[API] Starting Auto-Trainer in background...")
    start_auto_training()

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
indodax_api = IndodaxAPI()
indodax_forecaster = IndodaxAIForecaster()  # AI Forecaster
lstm_models = {}
cache_lock = threading.Lock()

# In-memory cache for expensive operations
class DataCache:
    def __init__(self):
        self.market_data = []
        self.indodax_data = []  # Indodax cache
        self.indodax_last_update = 0
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

        # === STAGE 2: ENRICHMENT (SNA + ALPHA ANALYSIS) ===
        # Now we process SNA and Alpha Score in the background
        data_cache.scan_status = "Running Alpha Hunter V3 Analysis"
        
        sna_results = sna_analyzer.analyze_batch(pairs_df)
        data_cache.scan_progress = 60
        
        enriched_data = []
        total_items = len(raw_data)

        for i, item in enumerate(raw_data):
            token_symbol = item['token']
            
            # Find SNA result
            sna_result = None
            for r in sna_results:
                if r.token_symbol == token_symbol:
                    sna_result = r
                    break
            
            if sna_result:
                # Use Alpha Score V3 (more accurate)
                item['sna_score'] = float(sna_result.alpha_score)
                item['buy_pressure'] = float(sna_result.buy_pressure)
                item['token_age_hours'] = float(sna_result.token_age_hours)
                item['alpha_rating'] = sna_result.alpha_rating.value if sna_result.alpha_rating else "N/A"
                item['is_alpha'] = sna_result.is_potential_pump
                
                # Confidence is now based on alpha_score (more accurate)
                final_conf = int(min(95, max(40, sna_result.alpha_score)))
            else:
                item['sna_score'] = 50.0
                item['buy_pressure'] = 50.0
                item['token_age_hours'] = 999
                item['alpha_rating'] = "N/A"
                item['is_alpha'] = False
                final_conf = 50
            
            item['prediction'] = {
                "confidence": final_conf,
                "pump_in_hours": 2 if item['sna_score'] > 70 else 6,
                "source": "alpha_v3"
            }
            
            # Status based on alpha detection
            if item.get('is_alpha') and item['sna_score'] >= 70:
                item['status'] = "PUMP"
            elif item['price_change_5m'] > 5:
                item['status'] = "PUMP"
            elif item['price_change_5m'] < -5:
                item['status'] = "DUMP"
            else:
                item['status'] = "NEUTRAL"
            
            enriched_data.append(item)
            
            # Update progress dynamically between 60 and 90
            current_progress = 60 + int((i / total_items) * 30)
            data_cache.scan_progress = current_progress

        # Sort by alpha_score (highest first)
        enriched_data.sort(key=lambda x: x.get('sna_score', 0), reverse=True)

        # Update Cache AGAIN with Enriched Data
        with cache_lock:
            data_cache.market_data = enriched_data
            data_cache.last_update = time.time()
            data_cache.scan_progress = 100
            data_cache.scan_status = "Idle"
            
        print(f"[API] Stage 2: Enriched {len(enriched_data)} tokens with Alpha V3")
        
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


# === PUMP/DUMP PREDICTION ENDPOINT (NEW!) ===

@app.get("/api/tokens/{address}/pump-prediction")
async def get_pump_prediction(address: str, chain: str = "solana"):
    """
    🔮 Pump/Dump Predictor - Predicts price movement in next 1 hour
    
    Returns:
    - signal: STRONG_PUMP, PUMP, NEUTRAL, DUMP, STRONG_DUMP
    - confidence: 0-100%
    - predicted_change_pct: Expected % change
    - key_factors: Why this prediction
    - entry_suggestion: Trading recommendation
    - stop_loss_pct: Suggested stop loss
    - take_profit_pct: Suggested take profit
    """
    try:
        # Get token info from DexScreener
        token_info = dex_api.get_token_info(address, chain_id=chain)
        if not token_info:
            raise HTTPException(status_code=404, detail="Token not found")
        
        # Get SNA data if available
        sna_data = None
        with cache_lock:
            for token in data_cache.market_data:
                if token.get('token_address') == address:
                    sna_data = {
                        'alpha_score': token.get('sna_score', 50),
                        'is_alpha': token.get('is_alpha', False)
                    }
                    break
        
        # Run pump prediction
        prediction = predict_pump_dump(token_info, sna_data)
        
        token_symbol = token_info.get('baseToken', {}).get('symbol', 'UNKNOWN')
        current_price = float(token_info.get('priceUsd', 0))
        
        return {
            "token": token_symbol,
            "address": address,
            "chain": chain,
            "current_price": current_price,
            "prediction": prediction,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error in pump prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pump-signals")
async def get_all_pump_signals():
    """
    🚀 Get pump/dump signals for ALL cached tokens
    Sorted by: STRONG_PUMP first, then by confidence
    
    Use this to find tokens most likely to pump in 1 hour
    """
    try:
        with cache_lock:
            tokens = data_cache.market_data.copy()
        
        if not tokens:
            return {"signals": [], "message": "No tokens cached yet"}
        
        # Get raw token data from cache and analyze each
        signals = []
        
        for token in tokens:
            address = token.get('token_address')
            
            # Get full token info
            token_info = dex_api.get_token_info(address, chain_id=token.get('chain', 'solana').lower())
            
            if token_info:
                sna_data = {
                    'alpha_score': token.get('sna_score', 50),
                    'is_alpha': token.get('is_alpha', False)
                }
                
                prediction = predict_pump_dump(token_info, sna_data)
                
                signals.append({
                    "token": token.get('token'),
                    "address": address,
                    "chain": token.get('chain'),
                    "current_price": token.get('price_usd'),
                    "price_change_5m": token.get('price_change_5m'),
                    "volume_1h": token.get('volume_1h'),
                    "liquidity": token.get('liquidity_usd'),
                    **prediction
                })
        
        # Sort by signal strength and confidence
        signal_order = {
            'STRONG_PUMP': 0,
            'PUMP': 1,
            'NEUTRAL': 2,
            'DUMP': 3,
            'STRONG_DUMP': 4
        }
        
        signals.sort(key=lambda x: (
            signal_order.get(x.get('signal_type', 'NEUTRAL'), 2), 
            -x.get('confidence', 0)
        ))
        
        # Summary stats
        pump_count = len([s for s in signals if 'PUMP' in s.get('signal_type', '')])
        dump_count = len([s for s in signals if 'DUMP' in s.get('signal_type', '')])
        
        return {
            "signals": signals,
            "summary": {
                "total_analyzed": len(signals),
                "pump_signals": pump_count,
                "dump_signals": dump_count,
                "neutral": len(signals) - pump_count - dump_count
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error getting pump signals: {e}")
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


# === INDODAX ENDPOINTS ===

@app.get("/api/indodax/tokens")
async def get_indodax_tokens(refresh: bool = False, limit: int = 200):
    """
    Get Indodax market data with buy/sell signals
    
    Returns tokens from Indonesian exchange with:
    - Price in IDR and USD
    - 24h change, volume, volatility
    - Buy/Sell signals based on order book analysis
    - Buy pressure percentage
    """
    try:
        # Check if refresh needed
        cache_age = time.time() - data_cache.indodax_last_update
        
        if refresh or cache_age > 15 or not data_cache.indodax_data:
            print("[API] Fetching Indodax data...")
            data_cache.indodax_data = get_indodax_market()
            data_cache.indodax_last_update = time.time()
        
        return data_cache.indodax_data[:limit]
        
    except Exception as e:
        print(f"[API] Indodax error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indodax/summary")
async def get_indodax_summary():
    """Get Indodax market summary"""
    try:
        tokens = data_cache.indodax_data or get_indodax_market()
        
        if not tokens:
            return {
                "total_tokens": 0,
                "total_volume_idr": 0,
                "buy_signals": 0,
                "sell_signals": 0,
                "avg_change": 0,
                "usd_idr_rate": indodax_api.usd_idr_rate
            }
        
        total_volume = sum(t.get('volume_24h_idr', 0) for t in tokens)
        buy_signals = len([t for t in tokens if 'BUY' in t.get('signal_type', '')])
        sell_signals = len([t for t in tokens if 'SELL' in t.get('signal_type', '')])
        avg_change = sum(t.get('change_24h', 0) for t in tokens) / len(tokens) if tokens else 0
        
        # Top gainer and loser
        sorted_by_change = sorted(tokens, key=lambda x: x.get('change_24h', 0), reverse=True)
        top_gainer = sorted_by_change[0] if sorted_by_change else None
        top_loser = sorted_by_change[-1] if sorted_by_change else None
        
        return {
            "total_tokens": len(tokens),
            "total_volume_idr": total_volume,
            "total_volume_usd": total_volume / indodax_api.usd_idr_rate if indodax_api.usd_idr_rate > 0 else 0,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "neutral_signals": len(tokens) - buy_signals - sell_signals,
            "avg_change": round(avg_change, 2),
            "usd_idr_rate": indodax_api.usd_idr_rate,
            "top_gainer": {
                "symbol": top_gainer.get('symbol'),
                "change": top_gainer.get('change_24h')
            } if top_gainer else None,
            "top_loser": {
                "symbol": top_loser.get('symbol'),
                "change": top_loser.get('change_24h')
            } if top_loser else None,
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Indodax summary error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indodax/token/{symbol}")
async def get_indodax_token_detail(symbol: str):
    """Get detailed info for a specific Indodax token"""
    try:
        pair_id = f"{symbol.lower()}idr"
        
        # Get order book analysis
        order_analysis = indodax_api.analyze_order_book(pair_id)
        
        # Get trade analysis
        trade_analysis = indodax_api.analyze_trades(pair_id)
        
        # Find token in cache
        token_data = None
        for t in data_cache.indodax_data:
            if t.get('symbol', '').upper() == symbol.upper():
                token_data = t
                break
        
        return {
            "symbol": symbol.upper(),
            "pair_id": pair_id,
            "token_data": token_data,
            "order_book_analysis": order_analysis,
            "trade_analysis": trade_analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Indodax token detail error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === AI FORECASTING ENDPOINTS ===

@app.get("/api/indodax/forecast/{symbol}")
async def get_ai_forecast(symbol: str):
    """
    Get AI-powered price forecast for an Indodax token
    Uses ensemble of LSTM, Bi-LSTM, GRU, Conv1D, Transformer
    """
    try:
        pair_id = f"{symbol.lower()}_idr"
        
        # Find current price from cache
        current_price = None
        for t in data_cache.indodax_data:
            if t.get('symbol', '').upper() == symbol.upper():
                current_price = t.get('price_idr', 0)
                break
        
        if not current_price:
            # Fetch from API
            ticker = indodax_api.get_ticker(pair_id.replace('_', ''))
            if ticker:
                current_price = float(ticker.get('last', 0))
        
        if not current_price:
            raise HTTPException(status_code=404, detail="Token not found")
        
        # Get AI forecast
        forecast = forecast_indodax_token(pair_id, current_price)
        
        if not forecast:
            raise HTTPException(status_code=500, detail="Forecast failed - insufficient data")
        
        return {
            "success": True,
            "forecast": forecast,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] AI Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indodax/forecast-batch")
async def get_batch_forecast(
    symbols: str = Query(None, description="Comma-separated symbols"),
    limit: int = Query(10, description="Number of tokens to forecast")
):
    """
    Get AI forecasts for multiple tokens
    Returns top potential pump candidates
    """
    try:
        results = []
        
        if symbols:
            symbol_list = [s.strip().upper() for s in symbols.split(',')]
        else:
            # Use top volume tokens
            sorted_tokens = sorted(
                data_cache.indodax_data, 
                key=lambda x: x.get('volume_idr', 0), 
                reverse=True
            )[:limit]
            symbol_list = [t.get('symbol', '').upper() for t in sorted_tokens]
        
        for symbol in symbol_list[:limit]:
            try:
                pair_id = f"{symbol.lower()}_idr"
                
                # Find current price
                current_price = None
                for t in data_cache.indodax_data:
                    if t.get('symbol', '').upper() == symbol:
                        current_price = t.get('price_idr', 0)
                        break
                
                if current_price:
                    forecast = forecast_indodax_token(pair_id, current_price)
                    if forecast:
                        results.append(forecast)
                        
            except Exception as e:
                print(f"[API] Forecast error for {symbol}: {e}")
                continue
        
        # Sort by potential gain
        results.sort(key=lambda x: x.get('change_24h_pct', 0), reverse=True)
        
        # Separate into buy/sell signals
        buy_signals = [r for r in results if 'BUY' in r.get('signal_type', '')]
        sell_signals = [r for r in results if 'SELL' in r.get('signal_type', '')]
        
        return {
            "success": True,
            "total": len(results),
            "forecasts": results,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Batch forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indodax/top-predictions")
async def get_top_predictions(
    limit: int = Query(20, description="Number of predictions")
):
    """
    Get top AI pump predictions from Indodax
    Returns tokens with highest predicted gains
    """
    try:
        # Get high volume tokens
        sorted_tokens = sorted(
            data_cache.indodax_data, 
            key=lambda x: x.get('volume_idr', 0), 
            reverse=True
        )[:50]
        
        predictions = []
        
        for token in sorted_tokens:
            symbol = token.get('symbol', '').upper()
            current_price = token.get('price_idr', 0)
            
            if not current_price:
                continue
                
            try:
                pair_id = f"{symbol.lower()}_idr"
                forecast = forecast_indodax_token(pair_id, current_price)
                
                if forecast:
                    forecast['current_volume_idr'] = token.get('volume_idr', 0)
                    forecast['change_24h_actual'] = token.get('change_24h', 0)
                    predictions.append(forecast)
                    
            except Exception as e:
                continue
        
        # Sort by predicted gain + confidence
        predictions.sort(
            key=lambda x: x.get('change_24h_pct', 0) * (x.get('confidence', 0) / 100),
            reverse=True
        )
        
        top_buys = [p for p in predictions if 'BUY' in p.get('signal_type', '')][:limit]
        top_sells = [p for p in predictions if 'SELL' in p.get('signal_type', '')][:limit]
        
        return {
            "success": True,
            "top_buy_predictions": top_buys,
            "top_sell_predictions": top_sells,
            "total_analyzed": len(predictions),
            "model_info": {
                "models": ["LSTM", "Bi-LSTM", "GRU", "Conv1D", "Transformer"],
                "ensemble_type": "Average",
                "target_accuracy": "90%+"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Top predictions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# === HEALTH CHECK ===

@app.get("/api/health")
async def health_check():
    """API Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cache_age_seconds": int(time.time() - data_cache.last_update),
        "tokens_cached": len(data_cache.market_data),
        "indodax_cached": len(data_cache.indodax_data),
        "is_scanning": data_cache.is_updating
    }


# === AUTO-TRAINING ENDPOINTS ===

@app.get("/api/training/status")
async def training_status():
    """Get current auto-training status"""
    try:
        status = get_training_status()
        return {
            "success": True,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/start")
async def start_training():
    """Start auto-training scheduler"""
    try:
        result = start_auto_training()
        return {
            "success": result,
            "message": "Auto-training started" if result else "Already running",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/stop")
async def stop_training():
    """Stop auto-training scheduler"""
    try:
        result = stop_auto_training()
        return {
            "success": result,
            "message": "Auto-training stopped",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/training/force/{symbol}")
async def force_training(symbol: str):
    """Force train a specific coin immediately"""
    try:
        result = force_train_coin(symbol)
        return {
            "success": result.get("success", False),
            "symbol": symbol.upper(),
            "accuracy": result.get("accuracy"),
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/results")
async def training_results():
    """Get all training results"""
    try:
        status = get_training_status()
        return {
            "success": True,
            "results": status.get("training_results", {}),
            "last_trained": status.get("last_trained", {}),
            "total_trained_today": status.get("total_trained_today", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Start cache update in thread
    update_market_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
