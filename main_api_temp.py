
# === FEAR & GREED INDEX ENDPOINTS ===

@app.get("/api/sentiment/fear-greed")
async def get_fear_greed():
    """Get current Fear & Greed Index"""
    try:
        result = fear_greed_api.get_current()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/fear-greed/history")
async def get_fear_greed_history(days: int = 30):
    """Get Fear & Greed historical data"""
    try:
        df = fear_greed_api.get_historical(days)
        return {
            "data": df.to_dict('records') if not df.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sentiment/signal")
async def get_sentiment_signal():
    """Get trading signal from Fear & Greed"""
    try:
        signal = fear_greed_api.get_signal()
        return signal
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === BINANCE API ENDPOINTS ===

@app.get("/api/binance/ticker/{symbol}")
async def get_binance_ticker(symbol: str):
    """Get Binance 24h ticker"""
    try:
        ticker = binance_api.get_ticker(symbol)
        return ticker if ticker else {"error": "No data"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/binance/price/{symbol}")
async def get_binance_price(symbol: str):
    """Get Binance current price"""
    try:
        price = binance_api.get_price(symbol)
        return {"symbol": symbol.upper(), "price": price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/binance/klines/{symbol}")
async def get_binance_klines(symbol: str, interval: str = "1h", limit: int = 100):
    """Get Binance historical candles"""
    try:
        df = binance_api.get_klines(symbol, interval, limit)
        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "data": df.to_dict('records') if not df.empty else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === ENHANCED PREDICTION ENDPOINT ===

@app.get("/api/predict/enhanced/{symbol}")
async def get_enhanced_prediction(symbol: str):
    """
    Get enhanced prediction with Binance data + Fear & Greed sentiment
    """
    try:
        from modules.crypto_data_aggregator import get_aggregated_data
        
        # Get data
        aggregated = get_aggregated_data(symbol, interval='1h')
        if not aggregated or aggregated.total_candles < 50:
            return {"error": f"Insufficient data for {symbol}"}
        
        df = aggregated.ohlcv
        current_price = df.iloc[-1]['close']
        
        # Get base prediction
        base_pred = unified_forecaster.forecast_ensemble(df)
        if not base_pred:
            return {"error": "Prediction failed"}
        
        # Get Binance data
        try:
            binance_ticker = binance_api.get_ticker(symbol)
            binance_volume = binance_ticker.get('volume_24h', 0) if binance_ticker else 0
            binance_change = binance_ticker.get('price_change_pct_24h', 0) if binance_ticker else 0
        except:
            binance_volume = 0
            binance_change = 0
        
        # Get Fear & Greed
        fg = fear_greed_api.get_current()
        
        # Adjust with sentiment
        adjusted = fear_greed_api.adjust_prediction(
            base_pred.signal.value,
            confidence=base_pred.confidence / 100
        )
        
        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "predicted_price_24h": base_pred.predicted_price_24h,
            "change_24h_pct": base_pred.change_24h_pct,
            "original_signal": base_pred.signal.value,
            "adjusted_signal": adjusted['adjusted_signal'],
            "confidence": adjusted['confidence'],
            "binance_volume_24h": binance_volume,
            "binance_change_24h": binance_change,
            "fear_greed_value": fg.get('value', 50),
            "fear_greed_classification": fg.get('classification', 'Neutral'),
            "sentiment_adjustment": adjusted['adjustment_reason'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"[API] Enhanced prediction error: {e}")
        import traceback
        traceback.print_exc()
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


# === PROPHET FORECASTING ENDPOINT ===

@app.get("/api/forecast/prophet/{symbol}")
async def forecast_prophet_simple(symbol: str):
    """Prophet Forecast - 100% Free"""
    try:
        from modules.crypto_data_aggregator import get_aggregated_data
        
        if symbol.upper() not in ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'MATIC', 'LINK']:
            return {"error": f"Symbol not supported: {symbol}"}
            
        aggregated = get_aggregated_data(symbol, interval='1h')
        
        if not aggregated or aggregated.total_candles < 50:
            return {"error": f"Insufficient data for {symbol}"}
        
        result = unified_forecaster.forecast_with_prophet(aggregated.ohlcv)
        
        if not result:
            return {"error": "Prophet not available"}
        
        return unified_forecaster._result_to_dict(result)
        
    except Exception as e:
        print(f"[API] Prophet error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Start cache update in thread
    update_market_cache()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
