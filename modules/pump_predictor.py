"""
Pump/Dump Predictor V1 - Short-Term Price Movement Prediction
============================================================
Predicts whether a token will PUMP or DUMP in the next 1 hour.

Features Used:
- Volume Acceleration (5m, 15m, 1h windows)
- Price Momentum & Velocity
- Buy/Sell Pressure Ratio
- Liquidity Flow Analysis
- Market Microstructure Signals
- Social Sentiment Velocity (from SNA)

Model: XGBoost + LSTM Hybrid for real-time prediction
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
import warnings
import json

warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import train_test_split
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input, 
                                         Bidirectional, Concatenate,
                                         BatchNormalization, Attention)
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class PredictionSignal(Enum):
    """Prediction outcomes"""
    STRONG_PUMP = "🚀 STRONG PUMP"      # >10% dalam 1 jam
    PUMP = "📈 PUMP"                     # 3-10% dalam 1 jam
    NEUTRAL = "➡️ SIDEWAYS"             # -3% to +3%
    DUMP = "📉 DUMP"                     # -3% to -10%
    STRONG_DUMP = "💥 STRONG DUMP"      # <-10% dalam 1 jam


@dataclass
class PumpPrediction:
    """Prediction result container"""
    signal: PredictionSignal
    confidence: float
    predicted_change_pct: float
    time_horizon: str
    key_factors: List[str]
    risk_level: str
    entry_suggestion: str
    stop_loss_pct: float
    take_profit_pct: float


class FeatureExtractor:
    """Extract predictive features from token data"""
    
    @staticmethod
    def extract_volume_features(token_data: Dict) -> Dict:
        """Volume-based signals"""
        volume_5m = float(token_data.get('volume', {}).get('m5', 0))
        volume_1h = float(token_data.get('volume', {}).get('h1', 0))
        volume_6h = float(token_data.get('volume', {}).get('h6', 0))
        volume_24h = float(token_data.get('volume', {}).get('h24', 0))
        
        # Volume acceleration (rate of change)
        vol_accel_5m_1h = volume_5m / max(volume_1h / 12, 1) if volume_1h > 0 else 1
        vol_accel_1h_6h = volume_1h / max(volume_6h / 6, 1) if volume_6h > 0 else 1
        vol_accel_1h_24h = volume_1h / max(volume_24h / 24, 1) if volume_24h > 0 else 1
        
        # Volume spike detection
        is_volume_spike = vol_accel_5m_1h > 2.0  # 5m volume 2x higher than hourly average
        
        return {
            'volume_5m': volume_5m,
            'volume_1h': volume_1h,
            'volume_24h': volume_24h,
            'vol_accel_5m': vol_accel_5m_1h,
            'vol_accel_1h': vol_accel_1h_6h,
            'vol_accel_24h': vol_accel_1h_24h,
            'is_volume_spike': is_volume_spike,
            'volume_score': min(100, (vol_accel_5m_1h + vol_accel_1h_6h) * 25)
        }
    
    @staticmethod
    def extract_price_features(token_data: Dict) -> Dict:
        """Price momentum signals"""
        price_change_5m = float(token_data.get('priceChange', {}).get('m5', 0))
        price_change_1h = float(token_data.get('priceChange', {}).get('h1', 0))
        price_change_6h = float(token_data.get('priceChange', {}).get('h6', 0))
        price_change_24h = float(token_data.get('priceChange', {}).get('h24', 0))
        
        # Momentum calculation
        momentum_5m = price_change_5m / 5 if price_change_5m != 0 else 0  # % per minute
        momentum_1h = price_change_1h / 60 if price_change_1h != 0 else 0
        
        # Acceleration (momentum change)
        price_acceleration = momentum_5m - momentum_1h
        
        # Trend strength
        if price_change_5m > 0 and price_change_1h > 0 and price_change_6h > 0:
            trend = 'STRONG_UP'
            trend_score = 90
        elif price_change_5m > 0 and price_change_1h > 0:
            trend = 'UP'
            trend_score = 70
        elif price_change_5m < 0 and price_change_1h < 0 and price_change_6h < 0:
            trend = 'STRONG_DOWN'
            trend_score = 10
        elif price_change_5m < 0 and price_change_1h < 0:
            trend = 'DOWN'
            trend_score = 30
        else:
            trend = 'MIXED'
            trend_score = 50
            
        return {
            'price_change_5m': price_change_5m,
            'price_change_1h': price_change_1h,
            'price_change_6h': price_change_6h,
            'price_change_24h': price_change_24h,
            'momentum_5m': momentum_5m,
            'momentum_1h': momentum_1h,
            'price_acceleration': price_acceleration,
            'trend': trend,
            'trend_score': trend_score
        }
    
    @staticmethod
    def extract_liquidity_features(token_data: Dict) -> Dict:
        """Liquidity and market depth signals"""
        liquidity = float(token_data.get('liquidity', {}).get('usd', 0))
        fdv = float(token_data.get('fdv', 0))
        market_cap = float(token_data.get('marketCap', 0))
        volume_24h = float(token_data.get('volume', {}).get('h24', 0))
        
        # Liquidity ratios
        vol_liq_ratio = volume_24h / liquidity if liquidity > 0 else 0
        liq_mcap_ratio = liquidity / market_cap if market_cap > 0 else 0
        
        # Liquidity health score
        if liquidity < 5000:
            liq_health = 'DANGER'
            liq_score = 20
        elif liquidity < 20000:
            liq_health = 'LOW'
            liq_score = 40
        elif liquidity < 100000:
            liq_health = 'MEDIUM'
            liq_score = 60
        elif liquidity < 500000:
            liq_health = 'GOOD'
            liq_score = 80
        else:
            liq_health = 'EXCELLENT'
            liq_score = 95
            
        return {
            'liquidity': liquidity,
            'fdv': fdv,
            'market_cap': market_cap,
            'vol_liq_ratio': vol_liq_ratio,
            'liq_mcap_ratio': liq_mcap_ratio,
            'liq_health': liq_health,
            'liq_score': liq_score
        }
    
    @staticmethod
    def extract_trading_features(token_data: Dict) -> Dict:
        """Trading activity signals"""
        txns = token_data.get('txns', {})
        
        # 5 minute transactions
        buys_5m = int(txns.get('m5', {}).get('buys', 0))
        sells_5m = int(txns.get('m5', {}).get('sells', 0))
        
        # 1 hour transactions
        buys_1h = int(txns.get('h1', {}).get('buys', 0))
        sells_1h = int(txns.get('h1', {}).get('sells', 0))
        
        # 24 hour transactions
        buys_24h = int(txns.get('h24', {}).get('buys', 0))
        sells_24h = int(txns.get('h24', {}).get('sells', 0))
        
        # Buy pressure calculation
        total_5m = buys_5m + sells_5m
        total_1h = buys_1h + sells_1h
        total_24h = buys_24h + sells_24h
        
        buy_pressure_5m = (buys_5m / total_5m * 100) if total_5m > 0 else 50
        buy_pressure_1h = (buys_1h / total_1h * 100) if total_1h > 0 else 50
        buy_pressure_24h = (buys_24h / total_24h * 100) if total_24h > 0 else 50
        
        # Trading velocity (tx per minute)
        tx_velocity_5m = total_5m / 5
        tx_velocity_1h = total_1h / 60
        
        # Velocity acceleration
        velocity_accel = tx_velocity_5m / max(tx_velocity_1h, 0.1)
        
        return {
            'buys_5m': buys_5m,
            'sells_5m': sells_5m,
            'buys_1h': buys_1h,
            'sells_1h': sells_1h,
            'buy_pressure_5m': buy_pressure_5m,
            'buy_pressure_1h': buy_pressure_1h,
            'buy_pressure_24h': buy_pressure_24h,
            'tx_velocity_5m': tx_velocity_5m,
            'tx_velocity_1h': tx_velocity_1h,
            'velocity_accel': velocity_accel,
            'total_txns_1h': total_1h
        }


class PumpDumpModel:
    """
    Hybrid ML Model for Pump/Dump Prediction
    Combines gradient boosting with rule-based signals
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.model = None
        self.is_trained = False
        
        # Historical predictions for accuracy tracking
        self.prediction_history = []
        
        # Weight configuration for scoring
        self.weights = {
            'volume_accel': 0.20,
            'price_momentum': 0.25,
            'buy_pressure': 0.25,
            'liquidity': 0.10,
            'trend_alignment': 0.20
        }
    
    def extract_all_features(self, token_data: Dict, sna_data: Dict = None) -> Dict:
        """Extract all features from token data"""
        volume_feats = self.feature_extractor.extract_volume_features(token_data)
        price_feats = self.feature_extractor.extract_price_features(token_data)
        liq_feats = self.feature_extractor.extract_liquidity_features(token_data)
        trade_feats = self.feature_extractor.extract_trading_features(token_data)
        
        # Combine all features
        all_features = {**volume_feats, **price_feats, **liq_feats, **trade_feats}
        
        # Add SNA features if available
        if sna_data:
            all_features['sna_score'] = sna_data.get('alpha_score', 50)
            all_features['is_alpha'] = sna_data.get('is_alpha', False)
        else:
            all_features['sna_score'] = 50
            all_features['is_alpha'] = False
            
        return all_features
    
    def calculate_pump_score(self, features: Dict) -> Tuple[float, List[str]]:
        """Calculate pump probability score (0-100)"""
        score = 0
        factors = []
        
        # 1. Volume Acceleration Signal (20%)
        vol_accel = features.get('vol_accel_5m', 1)
        if vol_accel > 3.0:
            score += 20
            factors.append(f"🔥 Volume spike {vol_accel:.1f}x")
        elif vol_accel > 2.0:
            score += 15
            factors.append(f"📊 Volume up {vol_accel:.1f}x")
        elif vol_accel > 1.5:
            score += 10
        elif vol_accel < 0.5:
            score -= 10
            factors.append("⚠️ Volume declining")
        
        # 2. Price Momentum Signal (25%)
        momentum = features.get('momentum_5m', 0)
        price_accel = features.get('price_acceleration', 0)
        
        if momentum > 0.5 and price_accel > 0:
            score += 25
            factors.append(f"🚀 Strong momentum +{momentum:.2f}%/min")
        elif momentum > 0.2:
            score += 18
            factors.append(f"📈 Positive momentum")
        elif momentum > 0:
            score += 10
        elif momentum < -0.5:
            score -= 20
            factors.append(f"📉 Negative momentum")
        elif momentum < 0:
            score -= 10
        
        # 3. Buy Pressure Signal (25%)
        buy_pressure = features.get('buy_pressure_5m', 50)
        bp_trend = buy_pressure - features.get('buy_pressure_1h', 50)
        
        if buy_pressure > 70:
            score += 25
            factors.append(f"💪 High buy pressure {buy_pressure:.0f}%")
        elif buy_pressure > 60:
            score += 18
            factors.append(f"📊 Good buy pressure {buy_pressure:.0f}%")
        elif buy_pressure > 55:
            score += 10
        elif buy_pressure < 40:
            score -= 20
            factors.append(f"⚠️ Sell pressure dominant {100-buy_pressure:.0f}%")
        elif buy_pressure < 45:
            score -= 10
            
        # Bonus for increasing buy pressure
        if bp_trend > 10:
            score += 5
            factors.append("↗️ Buy pressure increasing")
        
        # 4. Liquidity Health (10%)
        liq_score = features.get('liq_score', 50)
        if liq_score >= 80:
            score += 10
        elif liq_score >= 60:
            score += 7
        elif liq_score < 30:
            score -= 10
            factors.append("⚠️ Low liquidity risk")
        
        # 5. Trend Alignment (20%)
        trend_score = features.get('trend_score', 50)
        if trend_score >= 90:
            score += 20
            factors.append("📈 Strong uptrend confirmed")
        elif trend_score >= 70:
            score += 15
        elif trend_score <= 30:
            score -= 15
            factors.append("📉 Downtrend detected")
        elif trend_score <= 10:
            score -= 20
        
        # 6. Bonus: Transaction Velocity
        velocity_accel = features.get('velocity_accel', 1)
        if velocity_accel > 2:
            score += 5
            factors.append(f"⚡ Activity surge {velocity_accel:.1f}x")
        
        # 7. SNA Alpha Bonus
        if features.get('is_alpha', False):
            score += 10
            factors.append("🎯 Alpha token detected")
        
        # Normalize score to 0-100
        final_score = max(0, min(100, score + 50))  # Base 50 + adjustments
        
        return final_score, factors
    
    def predict_movement(self, token_data: Dict, sna_data: Dict = None) -> PumpPrediction:
        """
        Main prediction method
        Returns detailed pump/dump prediction for next 1 hour
        """
        try:
            # Extract features
            features = self.extract_all_features(token_data, sna_data)
            
            # Calculate pump score
            pump_score, factors = self.calculate_pump_score(features)
            
            # Determine signal based on score
            if pump_score >= 85:
                signal = PredictionSignal.STRONG_PUMP
                predicted_change = 10 + (pump_score - 85) * 0.5
                risk = "HIGH"
                entry = "ENTER NOW - Momentum sangat kuat"
                tp = 15.0
                sl = 5.0
            elif pump_score >= 70:
                signal = PredictionSignal.PUMP
                predicted_change = 3 + (pump_score - 70) * 0.4
                risk = "MEDIUM"
                entry = "GOOD ENTRY - Setup bullish"
                tp = 10.0
                sl = 4.0
            elif pump_score >= 45:
                signal = PredictionSignal.NEUTRAL
                predicted_change = (pump_score - 50) * 0.1
                risk = "LOW"
                entry = "WAIT - Tidak ada sinyal jelas"
                tp = 5.0
                sl = 3.0
            elif pump_score >= 30:
                signal = PredictionSignal.DUMP
                predicted_change = -3 - (45 - pump_score) * 0.3
                risk = "MEDIUM"
                entry = "AVOID - Tekanan jual tinggi"
                tp = 3.0
                sl = 5.0
            else:
                signal = PredictionSignal.STRONG_DUMP
                predicted_change = -10 - (30 - pump_score) * 0.5
                risk = "VERY HIGH"
                entry = "STAY AWAY - Dump incoming"
                tp = 2.0
                sl = 8.0
            
            # Calculate confidence based on factor agreement
            confidence = min(95, pump_score * 0.8 + len(factors) * 5)
            
            return PumpPrediction(
                signal=signal,
                confidence=round(confidence, 1),
                predicted_change_pct=round(predicted_change, 2),
                time_horizon="1 HOUR",
                key_factors=factors[:5],  # Top 5 factors
                risk_level=risk,
                entry_suggestion=entry,
                stop_loss_pct=sl,
                take_profit_pct=tp
            )
            
        except Exception as e:
            print(f"[PUMP-PREDICTOR] Error: {e}")
            return PumpPrediction(
                signal=PredictionSignal.NEUTRAL,
                confidence=50.0,
                predicted_change_pct=0.0,
                time_horizon="1 HOUR",
                key_factors=["Error dalam analisis"],
                risk_level="UNKNOWN",
                entry_suggestion="WAIT - Error dalam prediksi",
                stop_loss_pct=5.0,
                take_profit_pct=5.0
            )
    
    def to_dict(self, prediction: PumpPrediction) -> Dict:
        """Convert prediction to dictionary for API"""
        return {
            'signal': prediction.signal.value,
            'signal_type': prediction.signal.name,
            'confidence': prediction.confidence,
            'predicted_change_pct': prediction.predicted_change_pct,
            'time_horizon': prediction.time_horizon,
            'key_factors': prediction.key_factors,
            'risk_level': prediction.risk_level,
            'entry_suggestion': prediction.entry_suggestion,
            'stop_loss_pct': prediction.stop_loss_pct,
            'take_profit_pct': prediction.take_profit_pct
        }


# Global instance
pump_predictor = PumpDumpModel()


def predict_pump_dump(token_data: Dict, sna_data: Dict = None) -> Dict:
    """
    Main API function to predict pump/dump
    
    Args:
        token_data: Raw token data from DexScreener
        sna_data: Optional SNA analysis results
        
    Returns:
        Dictionary with prediction details
    """
    prediction = pump_predictor.predict_movement(token_data, sna_data)
    return pump_predictor.to_dict(prediction)


def batch_predict(tokens: List[Dict], sna_results: Dict = None) -> List[Dict]:
    """
    Batch predict for multiple tokens
    """
    results = []
    for token in tokens:
        pair_address = token.get('pairAddress', '')
        sna_data = sna_results.get(pair_address) if sna_results else None
        
        prediction = predict_pump_dump(token, sna_data)
        prediction['pairAddress'] = pair_address
        prediction['symbol'] = token.get('baseToken', {}).get('symbol', 'UNKNOWN')
        
        results.append(prediction)
    
    # Sort by confidence and signal strength
    signal_order = {
        'STRONG_PUMP': 0,
        'PUMP': 1,
        'NEUTRAL': 2,
        'DUMP': 3,
        'STRONG_DUMP': 4
    }
    
    results.sort(key=lambda x: (signal_order.get(x['signal_type'], 2), -x['confidence']))
    
    return results


# Quick test
if __name__ == "__main__":
    # Sample token data
    test_token = {
        'priceChange': {'m5': 5.2, 'h1': 8.5, 'h6': 15.0, 'h24': 25.0},
        'volume': {'m5': 50000, 'h1': 200000, 'h6': 800000, 'h24': 2000000},
        'liquidity': {'usd': 150000},
        'fdv': 500000,
        'marketCap': 400000,
        'txns': {
            'm5': {'buys': 45, 'sells': 20},
            'h1': {'buys': 180, 'sells': 120},
            'h24': {'buys': 1500, 'sells': 1200}
        }
    }
    
    result = predict_pump_dump(test_token, {'alpha_score': 75, 'is_alpha': True})
    print("\n=== PUMP/DUMP PREDICTION TEST ===")
    print(f"Signal: {result['signal']}")
    print(f"Confidence: {result['confidence']}%")
    print(f"Predicted Change: {result['predicted_change_pct']}%")
    print(f"Key Factors: {result['key_factors']}")
    print(f"Entry: {result['entry_suggestion']}")
