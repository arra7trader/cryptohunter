"""
Enhanced Predictor V1.0
======================
Unified prediction system combining:
- Prophet forecasting
- LSTM ensemble
- Binance market data
- Fear & Greed sentiment

Author: CryptoHunter Team
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class EnhancedPrediction:
    """Enhanced prediction result"""
    symbol: str
    current_price: float
    predicted_price_24h: float
    change_24h_pct: float
    signal: str
    confidence: float
    
    # Model predictions
    prophet_prediction: Optional[Dict] = None
    lstm_prediction: Optional[Dict] = None
    
    # Market data
    binance_volume_24h: float = 0
    binance_change_24h: float = 0
    
    # Sentiment
    fear_greed_value: int = 50
    fear_greed_classification: str = "Neutral"
    sentiment_adjustment: str = ""
    
    # Metadata
    timestamp: datetime = None
    data_sources: list = None
    
    def to_dict(self):
        d = asdict(self)
        if self.timestamp:
            d['timestamp'] = self.timestamp.isoformat()
        return d


class EnhancedPredictor:
    """
    Enhanced prediction engine with multi-source data
    """
    
    def __init__(self):
        """Initialize enhanced predictor"""
        from modules.timegpt_prophet_forecaster import UnifiedForecaster
        from modules.binance_api import BinanceAPI
        from modules.fear_greed import FearGreedIndex
        
        self.forecaster = UnifiedForecaster()
        self.binance = BinanceAPI()
        self.fear_greed = FearGreedIndex()
        
        print("[ENHANCED] Predictor initialized")
    
    def predict(self, symbol: str, use_binance: bool = True) -> EnhancedPrediction:
        """
        Generate enhanced prediction
        
        Args:
            symbol: Crypto symbol (BTC, ETH, etc)
            use_binance: Use Binance data if available
        
        Returns:
            EnhancedPrediction object
        """
        print(f"\n[ENHANCED] Generating prediction for {symbol}...")
        
        data_sources = []
        
        # 1. Get historical data
        try:
            from modules.crypto_data_aggregator import get_aggregated_data
            aggregated = get_aggregated_data(symbol, interval='1h')
            
            if not aggregated or aggregated.total_candles < 50:
                print(f"[ENHANCED] Insufficient data for {symbol}")
                return None
            
            df = aggregated.ohlcv
            current_price = df.iloc[-1]['close']
            data_sources.extend(aggregated.sources)
            
        except Exception as e:
            print(f"[ENHANCED] Error fetching data: {e}")
            return None
        
        # 2. Get Binance market data
        binance_volume = 0
        binance_change = 0
        
        if use_binance:
            try:
                ticker = self.binance.get_ticker(symbol)
                if ticker:
                    binance_volume = ticker.get('volume_24h', 0)
                    binance_change = ticker.get('price_change_pct_24h', 0)
                    data_sources.append('binance')
                    print(f"[ENHANCED] Binance: Volume {binance_volume:.2f}, Change {binance_change:+.2f}%")
            except Exception as e:
                print(f"[ENHANCED] Binance error: {e}")
        
        # 3. Get Fear & Greed
        fg = self.fear_greed.get_current()
        fg_value = fg.get('value', 50)
        fg_class = fg.get('classification', 'Neutral')
        print(f"[ENHANCED] Fear & Greed: {fg_value} ({fg_class})")
        
        # 4. Get base prediction (ensemble)
        print(f"[ENHANCED] Running AI models...")
        base_prediction = self.forecaster.forecast_ensemble(df)
        
        if not base_prediction:
            print(f"[ENHANCED] No prediction from models")
            return None
        
        # 5. Adjust prediction with Fear & Greed
        original_signal = base_prediction.signal.value
        adjusted = self.fear_greed.adjust_prediction(
            original_signal, 
            confidence=base_prediction.confidence / 100
        )
        
        final_signal = adjusted['adjusted_signal']
        final_confidence = adjusted['confidence']
        sentiment_adjustment = adjusted['adjustment_reason']
        
        print(f"[ENHANCED] Original: {original_signal} | Adjusted: {final_signal}")
        print(f"[ENHANCED] Confidence: {base_prediction.confidence:.1f}% → {final_confidence:.1f}%")
        
        # 6. Create enhanced prediction
        result = EnhancedPrediction(
            symbol=symbol.upper(),
            current_price=current_price,
            predicted_price_24h=base_prediction.predicted_price_24h,
            change_24h_pct=base_prediction.change_24h_pct,
            signal=final_signal,
            confidence=final_confidence,
            binance_volume_24h=binance_volume,
            binance_change_24h=binance_change,
            fear_greed_value=fg_value,
            fear_greed_classification=fg_class,
            sentiment_adjustment=sentiment_adjustment,
            timestamp=datetime.now(),
            data_sources=data_sources
        )
        
        print(f"[ENHANCED] ✓ Complete: {result.signal} with {result.confidence:.1f}% confidence")
        
        return result


# Convenience function
def get_enhanced_prediction(symbol: str) -> Dict:
    """Get enhanced prediction for symbol"""
    predictor = EnhancedPredictor()
    result = predictor.predict(symbol)
    return result.to_dict() if result else {}


# Test
if __name__ == "__main__":
    print("\n=== ENHANCED PREDICTOR TEST ===\n")
    
    predictor = EnhancedPredictor()
    
    # Test BTC prediction
    result = predictor.predict("BTC")
    
    if result:
        print("\n" + "="*50)
        print("ENHANCED PREDICTION RESULT")
        print("="*50)
        print(f"Symbol: {result.symbol}")
        print(f"Current Price: ${result.current_price:,.2f}")
        print(f"Predicted (24h): ${result.predicted_price_24h:,.2f}")
        print(f"Change: {result.change_24h_pct:+.2f}%")
        print(f"Signal: {result.signal}")
        print(f"Confidence: {result.confidence:.1f}%")
        print(f"\nBinance Volume: {result.binance_volume_24h:,.2f}")
        print(f"Fear & Greed: {result.fear_greed_value} ({result.fear_greed_classification})")
        print(f"Sentiment Adjustment: {result.sentiment_adjustment}")
        print(f"Data Sources: {', '.join(result.data_sources)}")
        print("="*50)
    
    print("\nTest complete!")
