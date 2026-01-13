"""
Fear & Greed Index Integration V1.0
===================================
Market sentiment indicator for crypto (0-100 scale)

Data Source: Alternative.me API (free, no API key)

Features:
- Current Fear & Greed value
- Historical data (30+ days)
- Classification (Extreme Fear → Extreme Greed)
- Trading signals based on extremes
- Integration with predictions

Author: CryptoHunter Team
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import time


class SentimentLevel(Enum):
    """Sentiment classification"""
    EXTREME_FEAR = "Extreme Fear"
    FEAR = "Fear"
    NEUTRAL = "Neutral"
    GREED = "Greed"
    EXTREME_GREED = "Extreme Greed"


class TradingSignal(Enum):
    """Trading signal based on sentiment"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class FearGreedIndex:
    """
    Fear & Greed Index API client
    Source: https://alternative.me/crypto/fear-and-greed-index/
    """
    
    API_URL = "https://api.alternative.me/fng/"
    
    def __init__(self):
        """Initialize Fear & Greed API client"""
        self.session = requests.Session()
        self.cache = {}
        self.cache_duration = 3600  # 1 hour
        print("[FEAR&GREED] API client initialized")
    
    def get_current(self) -> Dict:
        """
        Get current Fear & Greed Index value
        
        Returns:
            Dictionary with value, classification, and timestamp
        """
        # Check cache
        cache_key = 'current'
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
        
        try:
            response = self.session.get(f"{self.API_URL}?limit=1", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and 'data' in data and len(data['data']) > 0:
                fng_data = data['data'][0]
                
                value = int(fng_data.get('value', 50))
                classification = self._classify(value)
                signal = self._get_signal(value)
                
                result = {
                    'value': value,
                    'classification': classification.value,
                    'signal': signal.value,
                    'value_classification': fng_data.get('value_classification', ''),
                    'timestamp': datetime.fromtimestamp(int(fng_data.get('timestamp', time.time()))),
                    'time_until_update': fng_data.get('time_until_update', '')
                }
                
                # Cache result
                self.cache[cache_key] = (result, time.time())
                
                return result
                
        except Exception as e:
            print(f"[FEAR&GREED] Error fetching current index: {e}")
            
        # Return neutral if error
        return {
            'value': 50,
            'classification': SentimentLevel.NEUTRAL.value,
            'signal': TradingSignal.HOLD.value,
            'timestamp': datetime.now()
        }
    
    def get_historical(self, days: int = 30) -> pd.DataFrame:
        """
        Get historical Fear & Greed data
        
        Args:
            days: Number of days to fetch (max ~365)
        
        Returns:
            DataFrame with historical data
        """
        try:
            response = self.session.get(f"{self.API_URL}?limit={days}", timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data and 'data' in data:
                records = []
                for item in data['data']:
                    records.append({
                        'timestamp': datetime.fromtimestamp(int(item.get('timestamp', 0))),
                        'value': int(item.get('value', 50)),
                        'classification': item.get('value_classification', '')
                    })
                
                df = pd.DataFrame(records)
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                return df
                
        except Exception as e:
            print(f"[FEAR&GREED] Error fetching historical data: {e}")
        
        return pd.DataFrame()
    
    def get_signal(self, value: Optional[int] = None) -> Dict:
        """
        Get trading signal based on Fear & Greed value
        
        Args:
            value: F&G value (0-100), if None uses current
        
        Returns:
            Trading signal with reasoning
        """
        if value is None:
            current = self.get_current()
            value = current['value']
        
        signal = self._get_signal(value)
        classification = self._classify(value)
        
        # Generate reasoning
        if value <= 20:
            reason = "Extreme fear - strong buy opportunity (contrarian)"
            confidence = "high"
        elif value <= 40:
            reason = "Fear dominant - consider accumulation"
            confidence = "medium"
        elif value <= 60:
            reason = "Neutral sentiment - no strong signal"
            confidence = "low"
        elif value <= 80:
            reason = "Greed building - consider taking profits"
            confidence = "medium"
        else:
            reason = "Extreme greed - strong sell/caution signal"
            confidence = "high"
        
        return {
            'value': value,
            'classification': classification.value,
            'signal': signal.value,
            'reason': reason,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
    
    def _classify(self, value: int) -> SentimentLevel:
        """Classify F&G value into sentiment level"""
        if value <= 25:
            return SentimentLevel.EXTREME_FEAR
        elif value <= 45:
            return SentimentLevel.FEAR
        elif value <= 55:
            return SentimentLevel.NEUTRAL
        elif value <= 75:
            return SentimentLevel.GREED
        else:
            return SentimentLevel.EXTREME_GREED
    
    def _get_signal(self, value: int) -> TradingSignal:
        """Generate trading signal from F&G value"""
        if value <= 20:
            return TradingSignal.STRONG_BUY
        elif value <= 35:
            return TradingSignal.BUY
        elif value <= 65:
            return TradingSignal.HOLD
        elif value <= 80:
            return TradingSignal.SELL
        else:
            return TradingSignal.STRONG_SELL
    
    def adjust_prediction(self, base_signal: str, confidence: float = 1.0) -> Dict:
        """
        Adjust a prediction based on current F&G
        
        Args:
            base_signal: Original signal ("BUY", "SELL", etc)
            confidence: Confidence multiplier (0-1)
        
        Returns:
            Adjusted prediction with reasoning
        """
        fg = self.get_current()
        fg_value = fg['value']
        
        # Adjustment logic
        adjusted_signal = base_signal
        adjustment_reason = ""
        confidence_adjustment = 0
        
        if base_signal.upper() in ["BUY", "STRONG BUY"]:
            if fg_value > 80:
                # Extreme greed - reduce buy confidence
                adjusted_signal = "HOLD"
                adjustment_reason = "F&G extreme greed - halted buy signal"
                confidence_adjustment = -30
            elif fg_value < 25:
                # Extreme fear - increase buy confidence
                adjusted_signal = "STRONG BUY"
                adjustment_reason = "F&G extreme fear - strengthened buy signal"
                confidence_adjustment = +20
        
        elif base_signal.upper() in ["SELL", "STRONG SELL"]:
            if fg_value < 25:
                # Extreme fear - reduce sell confidence
                adjusted_signal = "HOLD"
                adjustment_reason = "F&G extreme fear - halted sell signal"
                confidence_adjustment = -30
            elif fg_value > 80:
                # Extreme greed - increase sell confidence
                adjusted_signal = "STRONG SELL"
                adjustment_reason = "F&G extreme greed - strengthened sell signal"
                confidence_adjustment = +20
        
        # Adjust confidence
        adjusted_confidence = min(100, max(0, confidence * 100 + confidence_adjustment))
        
        return {
            'original_signal': base_signal,
            'adjusted_signal': adjusted_signal,
            'fear_greed_value': fg_value,
            'fear_greed_classification': fg['classification'],
            'adjustment_reason': adjustment_reason or "No F&G adjustment needed",
            'confidence': adjusted_confidence,
            'timestamp': datetime.now()
        }


# Convenience functions
def get_fear_greed_value() -> int:
    """Get current F&G value"""
    api = FearGreedIndex()
    current = api.get_current()
    return current['value']


def get_fear_greed_signal() -> Dict:
    """Get trading signal from F&G"""
    api = FearGreedIndex()
    return api.get_signal()


def get_fear_greed_history(days: int = 30) -> pd.DataFrame:
    """Get F&G historical data"""
    api = FearGreedIndex()
    return api.get_historical(days)


# Test
if __name__ == "__main__":
    print("\n=== FEAR & GREED INDEX TEST ===\n")
    
    api = FearGreedIndex()
    
    # Test current value
    print("1. Current Fear & Greed:")
    current = api.get_current()
    print(f"Value: {current['value']}")
    print(f"Classification: {current['classification']}")
    print(f"Signal: {current['signal']}")
    print(f"Timestamp: {current['timestamp']}")
    
    # Test signal
    print("\n2. Trading Signal:")
    signal = api.get_signal()
    print(f"Signal: {signal['signal']}")
    print(f"Reason: {signal['reason']}")
    print(f"Confidence: {signal['confidence']}")
    
    # Test historical
    print("\n3. Historical Data (last 7 days):")
    history = api.get_historical(7)
    if not history.empty:
        print(history)
        print(f"\nAverage F&G (7 days): {history['value'].mean():.1f}")
    
    # Test adjustment
    print("\n4. Prediction Adjustment:")
    adjustment = api.adjust_prediction("BUY", confidence=0.85)
    print(f"Original: {adjustment['original_signal']}")
    print(f"Adjusted: {adjustment['adjusted_signal']}")
    print(f"Reason: {adjustment['adjustment_reason']}")
    print(f"Confidence: {adjustment['confidence']:.1f}%")
    
    print("\nTest complete!")
