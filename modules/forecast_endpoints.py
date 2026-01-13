# === TIMEGPT & PROPHET FORECASTING ENDPOINTS ===
# Add these endpoints to main_api.py before the "if __name__ == '__main__':" block

from fastapi import HTTPException
from datetime import datetime


@app.get("/api/forecast/timegpt/{symbol}")
async def forecast_with_timegpt(symbol: str):
    """
    🔮 TimeGPT Forecast
    
    Uses Nixtla TimeGPT API for crypto price forecasting
    Requires API key in .env: TIMEGPT_API_KEY
    
    Args:
        symbol: Crypto symbol (e.g., BTC, ETH, SOL)
    
    Returns:
        Forecast for 1h, 4h, 24h with confidence intervals
    """
    try:
        # Get historical data
        if symbol.upper() in ['BTC', 'ETH', 'SOL', 'DOGE']:
            # For major coins, try to get from aggregator
            from modules.crypto_data_aggregator import get_aggregated_data
            aggregated = get_aggregated_data(symbol, interval='1h')
            
            if aggregated and aggregated.total_candles >= 100:
                df = aggregated.ohlcv
            else:
                return {"error": f"Insufficient data for {symbol}"}
        else:
            return {"error": f"TimeGPT currently supports major coins only (BTC, ETH, SOL, DOGE)"}
        
        # Run forecast
        result = unified_forecaster.forecast_with_timegpt(df)
        
        if result is None:
            return {
                "error": "TimeGPT not available",
                "message": "TimeGPT requires API key. Set TIMEGPT_API_KEY in .env file",
                "get_api_key": "https://dashboard.nixtla.io/"
            }
        
        return unified_forecaster._result_to_dict(result)
        
    except Exception as e:
        print(f"[API] TimeGPT forecast error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/prophet/{symbol}")
async def forecast_with_prophet(symbol: str):
    """
    📊 Prophet Forecast
    
    Uses Meta Prophet for crypto price forecasting
    100% Free, no API key required
    
    Args:
        symbol: Crypto symbol (e.g., BTC, ETH, SOL)
    
    Returns:
        Forecast for 1h, 4h, 24h with confidence intervals
    """
    try:
        # Get historical data
        if symbol.upper() in ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'MATIC', 'LINK']:
            # Try to get from aggregator
            from modules.crypto_data_aggregator import get_aggregated_data
            aggregated = get_aggregated_data(symbol, interval='1h')
            
            if aggregated and aggregated.total_candles >= 50:
                df = aggregated.ohlcv
            else:
                return {"error": f"Insufficient data for {symbol}"}
        else:
            return {"error": f"Symbol not supported yet: {symbol}"}
        
        # Run forecast
        result = unified_forecaster.forecast_with_prophet(df)
        
        if result is None:
            return {
                "error": "Prophet not available",
                "message": "Install Prophet with: pip install prophet"
            }
        
        return unified_forecaster._result_to_dict(result)
        
    except Exception as e:
        print(f"[API] Prophet forecast error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/forecast/compare/{symbol}")
async def compare_forecasts(symbol: str):
    """
    🎯 Compare TimeGPT vs Prophet vs LSTM
    
    Get forecasts from all available models and compare them
    Shows agreement level and recommended action
    
    Args:
        symbol: Crypto symbol (e.g., BTC, ETH, SOL)
    
    Returns:
        Comparison of all models with agreement score
    """
    try:
        # Get historical data
        if symbol.upper() in ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'MATIC', 'LINK']:
            from modules.crypto_data_aggregator import get_aggregated_data
            aggregated = get_aggregated_data(symbol, interval='1h')
            
            if aggregated and aggregated.total_candles >= 50:
                df = aggregated.ohlcv
            else:
                return {"error": f"Insufficient data for {symbol}"}
        else:
            return {"error": f"Symbol not supported yet: {symbol}"}
        
        # Get comparison
        comparison = unified_forecaster.compare_models(df)
        
        # Add ensemble prediction
        ensemble = unified_forecaster.forecast_ensemble(df)
        if ensemble:
            comparison['ensemble'] = unified_forecaster._result_to_dict(ensemble)
        
        return {
            "symbol": symbol.upper(),
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Forecast comparison error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
