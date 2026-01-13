"""
TimeGPT & Prophet Forecaster V1.0
==================================
Dual forecasting dengan TimeGPT API (Nixtla) dan Prophet (Meta)

Features:
- TimeGPT: AI-powered forecasting with confidence intervals
- Prophet: Open-source time series forecasting
- Unified interface for both models
- Comparison mode
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
import time

warnings.filterwarnings('ignore')

# TimeGPT
try:
    from nixtla import NixtlaClient
    TIMEGPT_AVAILABLE = True
except ImportError:
    TIMEGPT_AVAILABLE = False
    print("[WARNING] TimeGPT not available. Install with: pip install nixtla")

# Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("[WARNING] Prophet not available. Install with: pip install prophet")


class ForecastSignal(Enum):
    """Forecast signal types"""
    STRONG_BUY = "🚀 STRONG BUY"
    BUY = "📈 BUY"
    HOLD = "➡️ HOLD"
    SELL = "📉 SELL"
    STRONG_SELL = "💥 STRONG SELL"


@dataclass
class ForecastResult:
    """Hasil prediksi"""
    symbol: str
    model: str  # "timegpt", "prophet", or "ensemble"
    current_price: float
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_24h: float
    change_1h_pct: float
    change_4h_pct: float
    change_24h_pct: float
    signal: ForecastSignal
    confidence: float
    risk_level: str
    recommendation: str
    lower_bound_24h: Optional[float] = None  # Confidence interval
    upper_bound_24h: Optional[float] = None


class TimeGPTForecaster:
    """
    TimeGPT API Integration (Nixtla)
    Requires API key from https://dashboard.nixtla.io/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('TIMEGPT_API_KEY')
        self.enabled = os.getenv('TIMEGPT_ENABLED', 'true').lower() == 'true'
        self.client = None
        
        if TIMEGPT_AVAILABLE and self.api_key and self.enabled:
            try:
                self.client = NixtlaClient(api_key=self.api_key)
                print("[TimeGPT] [OK] Initialized successfully")
            except Exception as e:
                print(f"[TimeGPT] [ERROR] Initialization failed: {e}")
                self.client = None
        else:
            if not TIMEGPT_AVAILABLE:
                print("[TimeGPT] [ERROR] Library not installed")
            elif not self.api_key:
                print("[TimeGPT] [WARNING] No API key found")
            elif not self.enabled:
                print("[TimeGPT] [WARNING] Disabled in config")
    
    def is_available(self) -> bool:
        """Check if TimeGPT is available"""
        return self.client is not None
    
    def forecast(self, df: pd.DataFrame, horizon: int = 24) -> Optional[ForecastResult]:
        """
        Generate forecast using TimeGPT
        
        Args:
            df: DataFrame with columns ['timestamp', 'close']
            horizon: Number of hours to forecast
        
        Returns:
            ForecastResult or None
        """
        if not self.is_available():
            return None
        
        try:
            # Prepare data for TimeGPT
            df_prep = df[['timestamp', 'close']].copy()
            df_prep.columns = ['ds', 'y']  # TimeGPT expects 'ds' and 'y'
            
            # Make sure timestamp is datetime
            df_prep['ds'] = pd.to_datetime(df_prep['ds'])
            
            # Get current price
            current_price = float(df_prep['y'].iloc[-1])
            
            print(f"[TimeGPT] Forecasting {horizon}h ahead...")
            
            # Call TimeGPT API
            forecast_df = self.client.forecast(
                df=df_prep,
                h=horizon,
                level=[80, 95]  # Confidence intervals
            )
            
            # Extract predictions
            pred_1h = float(forecast_df['TimeGPT'].iloc[0]) if len(forecast_df) >= 1 else current_price
            pred_4h = float(forecast_df['TimeGPT'].iloc[3]) if len(forecast_df) >= 4 else current_price
            pred_24h = float(forecast_df['TimeGPT'].iloc[-1])
            
            # Confidence intervals
            lower_bound = float(forecast_df['TimeGPT-lo-95'].iloc[-1]) if 'TimeGPT-lo-95' in forecast_df else None
            upper_bound = float(forecast_df['TimeGPT-hi-95'].iloc[-1]) if 'TimeGPT-hi-95' in forecast_df else None
            
            # Calculate changes
            change_1h = ((pred_1h - current_price) / current_price) * 100
            change_4h = ((pred_4h - current_price) / current_price) * 100
            change_24h = ((pred_24h - current_price) / current_price) * 100
            
            # Determine signal
            avg_change = (change_1h + change_4h + change_24h) / 3
            
            if avg_change > 3:
                signal = ForecastSignal.STRONG_BUY
                risk = "HIGH"
                rec = f"🔥 TimeGPT predicts strong upward trend (+{avg_change:.1f}%)"
            elif avg_change > 1:
                signal = ForecastSignal.BUY
                risk = "MEDIUM"
                rec = f"📈 TimeGPT expects positive movement (+{avg_change:.1f}%)"
            elif avg_change < -3:
                signal = ForecastSignal.STRONG_SELL
                risk = "HIGH"
                rec = f"[WARNING] TimeGPT predicts strong downward trend ({avg_change:.1f}%)"
            elif avg_change < -1:
                signal = ForecastSignal.SELL
                risk = "MEDIUM"
                rec = f"📉 TimeGPT expects negative movement ({avg_change:.1f}%)"
            else:
                signal = ForecastSignal.HOLD
                risk = "LOW"
                rec = f"➡️ TimeGPT sees sideways movement ({avg_change:.1f}%)"
            
            # Confidence based on interval width
            if lower_bound and upper_bound:
                interval_width = (upper_bound - lower_bound) / current_price * 100
                confidence = max(50, min(95, 95 - interval_width))
            else:
                confidence = 85.0  # Default
            
            result = ForecastResult(
                symbol=df.get('symbol', ['UNKNOWN'])[0] if 'symbol' in df else 'UNKNOWN',
                model="timegpt",
                current_price=current_price,
                predicted_price_1h=pred_1h,
                predicted_price_4h=pred_4h,
                predicted_price_24h=pred_24h,
                change_1h_pct=round(change_1h, 2),
                change_4h_pct=round(change_4h, 2),
                change_24h_pct=round(change_24h, 2),
                signal=signal,
                confidence=round(confidence, 2),
                risk_level=risk,
                recommendation=rec,
                lower_bound_24h=lower_bound,
                upper_bound_24h=upper_bound
            )
            
            print(f"[TimeGPT] [OK] Forecast complete: {change_24h:+.2f}% in 24h")
            return result
            
        except Exception as e:
            print(f"[TimeGPT] [ERROR] Forecast error: {e}")
            import traceback
            traceback.print_exc()
            return None


class ProphetForecaster:
    """
    Prophet Forecasting (Meta)
    100% Free, No API key needed
    """
    
    def __init__(self):
        self.enabled = os.getenv('PROPHET_ENABLED', 'true').lower() == 'true'
        self.available = PROPHET_AVAILABLE and self.enabled
        
        if self.available:
            print("[Prophet] [OK] Ready")
        else:
            if not PROPHET_AVAILABLE:
                print("[Prophet] [ERROR] Library not installed")
            else:
                print("[Prophet] [WARNING] Disabled in config")
    
    def is_available(self) -> bool:
        """Check if Prophet is available"""
        return self.available
    
    def forecast(self, df: pd.DataFrame, horizon_hours: int = 24) -> Optional[ForecastResult]:
        """
        Generate forecast using Prophet
        
        Args:
            df: DataFrame with columns ['timestamp', 'close']
            horizon_hours: Number of hours to forecast
        
        Returns:
            ForecastResult or None
        """
        if not self.is_available():
            return None
        
        try:
            # Prepare data for Prophet
            df_prep = df[['timestamp', 'close']].copy()
            df_prep.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y'
            
            # Make sure timestamp is datetime
            df_prep['ds'] = pd.to_datetime(df_prep['ds'])
            
            # Get current price
            current_price = float(df_prep['y'].iloc[-1])
            
            print(f"[Prophet] Training model...")
            
            # Initialize and train Prophet
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                interval_width=0.95,
                changepoint_prior_scale=0.05  # More flexible to recent changes
            )
            
            # Suppress Prophet's verbose output
            import logging
            logging.getLogger('prophet').setLevel(logging.WARNING)
            
            model.fit(df_prep)
            
            # Create future dataframe
            # Infer frequency from data
            if len(df_prep) >= 2:
                freq = pd.infer_freq(df_prep['ds'])
                if freq is None:
                    # Calculate average time difference
                    time_diff = (df_prep['ds'].iloc[-1] - df_prep['ds'].iloc[-2]).total_seconds() / 3600
                    freq = f'{int(time_diff)}H'
            else:
                freq = '1H'  # Default to 1 hour
            
            future = model.make_future_dataframe(periods=horizon_hours, freq=freq)
            
            print(f"[Prophet] Forecasting {horizon_hours}h ahead...")
            forecast = model.predict(future)
            
            # Get predictions (last rows are the future)
            future_forecast = forecast.tail(horizon_hours)
            
            # Extract predictions at different horizons
            pred_1h = float(future_forecast['yhat'].iloc[0]) if len(future_forecast) >= 1 else current_price
            pred_4h = float(future_forecast['yhat'].iloc[3]) if len(future_forecast) >= 4 else current_price
            pred_24h = float(future_forecast['yhat'].iloc[-1])
            
            # Confidence intervals
            lower_bound = float(future_forecast['yhat_lower'].iloc[-1])
            upper_bound = float(future_forecast['yhat_upper'].iloc[-1])
            
            # Calculate changes
            change_1h = ((pred_1h - current_price) / current_price) * 100
            change_4h = ((pred_4h - current_price) / current_price) * 100
            change_24h = ((pred_24h - current_price) / current_price) * 100
            
            # Determine signal
            avg_change = (change_1h + change_4h + change_24h) / 3
            
            if avg_change > 3:
                signal = ForecastSignal.STRONG_BUY
                risk = "HIGH"
                rec = f"🔥 Prophet predicts strong upward trend (+{avg_change:.1f}%)"
            elif avg_change > 1:
                signal = ForecastSignal.BUY
                risk = "MEDIUM"
                rec = f"📈 Prophet expects positive movement (+{avg_change:.1f}%)"
            elif avg_change < -3:
                signal = ForecastSignal.STRONG_SELL
                risk = "HIGH"
                rec = f"[WARNING] Prophet predicts strong downward trend ({avg_change:.1f}%)"
            elif avg_change < -1:
                signal = ForecastSignal.SELL
                risk = "MEDIUM"
                rec = f"📉 Prophet expects negative movement ({avg_change:.1f}%)"
            else:
                signal = ForecastSignal.HOLD
                risk = "LOW"
                rec = f"➡️ Prophet sees sideways movement ({avg_change:.1f}%)"
            
            # Confidence based on interval width
            interval_width = (upper_bound - lower_bound) / current_price * 100
            confidence = max(50, min(95, 95 - interval_width / 2))
            
            result = ForecastResult(
                symbol=df.get('symbol', ['UNKNOWN'])[0] if 'symbol' in df else 'UNKNOWN',
                model="prophet",
                current_price=current_price,
                predicted_price_1h=pred_1h,
                predicted_price_4h=pred_4h,
                predicted_price_24h=pred_24h,
                change_1h_pct=round(change_1h, 2),
                change_4h_pct=round(change_4h, 2),
                change_24h_pct=round(change_24h, 2),
                signal=signal,
                confidence=round(confidence, 2),
                risk_level=risk,
                recommendation=rec,
                lower_bound_24h=lower_bound,
                upper_bound_24h=upper_bound
            )
            
            print(f"[Prophet] [OK] Forecast complete: {change_24h:+.2f}% in 24h")
            return result
            
        except Exception as e:
            print(f"[Prophet] [ERROR] Forecast error: {e}")
            import traceback
            traceback.print_exc()
            return None


class UnifiedForecaster:
    """
    Unified interface for TimeGPT, Prophet, and LSTM Ensemble
    Provides ensemble predictions and model comparison
    """
    
    def __init__(self, timegpt_api_key: Optional[str] = None):
        self.timegpt = TimeGPTForecaster(api_key=timegpt_api_key)
        self.prophet = ProphetForecaster()
        
        # Import LSTM ensemble from existing module
        try:
            from modules.indodax_forecaster import DeepLearningForecaster
            self.lstm_ensemble = DeepLearningForecaster(lookback=30, forecast_horizon=24)
            self.lstm_available = True
            print("[UnifiedForecaster] [OK] LSTM Ensemble loaded (5 models)")
        except ImportError:
            try:
                from indodax_forecaster import DeepLearningForecaster
                self.lstm_ensemble = DeepLearningForecaster(lookback=30, forecast_horizon=24)
                self.lstm_available = True
                print("[UnifiedForecaster] [OK] LSTM Ensemble loaded (5 models)")
            except ImportError:
                self.lstm_ensemble = None
                self.lstm_available = False
                print("[UnifiedForecaster] [WARNING] LSTM Ensemble not available")
        
        # Cache for predictions
        self.cache = {}
        self.cache_duration = 600  # 10 minutes
    
    def forecast_with_timegpt(self, df: pd.DataFrame) -> Optional[ForecastResult]:
        """Forecast using TimeGPT only"""
        return self.timegpt.forecast(df)
    
    def forecast_with_prophet(self, df: pd.DataFrame) -> Optional[ForecastResult]:
        """Forecast using Prophet only"""
        return self.prophet.forecast(df)
    
    def forecast_ensemble(self, df: pd.DataFrame) -> Optional[ForecastResult]:
        """
        Ensemble prediction using both models
        Returns weighted average with higher weight for more confident model
        """
        timegpt_result = self.timegpt.forecast(df)
        prophet_result = self.prophet.forecast(df)
        
        # If only one model available, return that
        if timegpt_result and not prophet_result:
            return timegpt_result
        if prophet_result and not timegpt_result:
            return prophet_result
        if not timegpt_result and not prophet_result:
            return None
        
        # Both available - create ensemble
        try:
            # Weight by confidence
            w_timegpt = timegpt_result.confidence / (timegpt_result.confidence + prophet_result.confidence)
            w_prophet = 1 - w_timegpt
            
            # Weighted predictions
            pred_1h = (timegpt_result.predicted_price_1h * w_timegpt + 
                      prophet_result.predicted_price_1h * w_prophet)
            pred_4h = (timegpt_result.predicted_price_4h * w_timegpt + 
                      prophet_result.predicted_price_4h * w_prophet)
            pred_24h = (timegpt_result.predicted_price_24h * w_timegpt + 
                       prophet_result.predicted_price_24h * w_prophet)
            
            current_price = timegpt_result.current_price
            
            # Calculate changes
            change_1h = ((pred_1h - current_price) / current_price) * 100
            change_4h = ((pred_4h - current_price) / current_price) * 100
            change_24h = ((pred_24h - current_price) / current_price) * 100
            
            # Determine signal (require agreement from both models for strong signals)
            avg_change = (change_1h + change_4h + change_24h) / 3
            both_bullish = (timegpt_result.signal in [ForecastSignal.BUY, ForecastSignal.STRONG_BUY] and
                          prophet_result.signal in [ForecastSignal.BUY, ForecastSignal.STRONG_BUY])
            both_bearish = (timegpt_result.signal in [ForecastSignal.SELL, ForecastSignal.STRONG_SELL] and
                          prophet_result.signal in [ForecastSignal.SELL, ForecastSignal.STRONG_SELL])
            
            if avg_change > 3 and both_bullish:
                signal = ForecastSignal.STRONG_BUY
                risk = "HIGH"
                rec = f"🔥 ENSEMBLE STRONG BUY - Both models agree (+{avg_change:.1f}%)"
            elif avg_change > 1 and both_bullish:
                signal = ForecastSignal.BUY
                risk = "MEDIUM"
                rec = f"📈 ENSEMBLE BUY - Models align on uptrend (+{avg_change:.1f}%)"
            elif avg_change < -3 and both_bearish:
                signal = ForecastSignal.STRONG_SELL
                risk = "HIGH"
                rec = f"[WARNING] ENSEMBLE STRONG SELL - Both models agree ({avg_change:.1f}%)"
            elif avg_change < -1 and both_bearish:
                signal = ForecastSignal.SELL
                risk = "MEDIUM"
                rec = f"📉 ENSEMBLE SELL - Models align on downtrend ({avg_change:.1f}%)"
            else:
                signal = ForecastSignal.HOLD
                risk = "LOW"
                rec = f"➡️ ENSEMBLE HOLD - Mixed signals ({avg_change:.1f}%)"
            
            # Ensemble confidence (average of both)
            confidence = (timegpt_result.confidence + prophet_result.confidence) / 2
            
            result = ForecastResult(
                symbol=timegpt_result.symbol,
                model="ensemble",
                current_price=current_price,
                predicted_price_1h=round(pred_1h, 2),
                predicted_price_4h=round(pred_4h, 2),
                predicted_price_24h=round(pred_24h, 2),
                change_1h_pct=round(change_1h, 2),
                change_4h_pct=round(change_4h, 2),
                change_24h_pct=round(change_24h, 2),
                signal=signal,
                confidence=round(confidence, 2),
                risk_level=risk,
                recommendation=rec,
                lower_bound_24h=None,  # Not applicable for ensemble
                upper_bound_24h=None
            )
            
            print(f"[ENSEMBLE] [OK] Forecast complete: {change_24h:+.2f}% in 24h (confidence: {confidence:.1f}%)")
            return result
            
        except Exception as e:
            print(f"[ENSEMBLE] [ERROR] Error: {e}")
            # Fallback to higher confidence model
            if timegpt_result.confidence > prophet_result.confidence:
                return timegpt_result
            else:
                return prophet_result
    
    def compare_models(self, df: pd.DataFrame) -> Dict:
        """
        Compare predictions from both models
        Returns dict with both results and comparison metrics
        """
        timegpt_result = self.timegpt.forecast(df)
        prophet_result = self.prophet.forecast(df)
        
        comparison = {
            'timegpt': self._result_to_dict(timegpt_result) if timegpt_result else None,
            'prophet': self._result_to_dict(prophet_result) if prophet_result else None,
            'agreement': None,
            'recommendation': None
        }
        
        if timegpt_result and prophet_result:
            # Calculate agreement
            signal_match = timegpt_result.signal == prophet_result.signal
            pred_diff = abs(timegpt_result.change_24h_pct - prophet_result.change_24h_pct)
            
            if signal_match and pred_diff < 2:
                comparison['agreement'] = 'HIGH'
                comparison['recommendation'] = 'Strong signal - both models agree'
            elif signal_match and pred_diff < 5:
                comparison['agreement'] = 'MEDIUM'
                comparison['recommendation'] = 'Moderate agreement'
            else:
                comparison['agreement'] = 'LOW'
                comparison['recommendation'] = 'Conflicting signals - use caution'
        
        return comparison
    
    def _result_to_dict(self, result: Optional[ForecastResult]) -> Optional[Dict]:
        """Convert ForecastResult to dictionary"""
        if not result:
            return None
        
        return {
            'symbol': result.symbol,
            'model': result.model,
            'current_price': result.current_price,
            'predicted_price_1h': result.predicted_price_1h,
            'predicted_price_4h': result.predicted_price_4h,
            'predicted_price_24h': result.predicted_price_24h,
            'change_1h_pct': result.change_1h_pct,
            'change_4h_pct': result.change_4h_pct,
            'change_24h_pct': result.change_24h_pct,
            'signal': result.signal.value,
            'confidence': result.confidence,
            'risk_level': result.risk_level,
            'recommendation': result.recommendation,
            'lower_bound_24h': result.lower_bound_24h,
            'upper_bound_24h': result.upper_bound_24h
        }


# Test function
if __name__ == "__main__":
    print("[TEST] Testing TimeGPT & Prophet Forecasters...")
    
    # Create sample data
    dates = pd.date_range(end=datetime.now(), periods=100, freq='1H')
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)  # Random walk
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices
    })
    
    forecaster = UnifiedForecaster()
    
    print("\n[CHART] Prophet Forecast:")
    prophet_result = forecaster.forecast_with_prophet(df)
    if prophet_result:
        print(f"   Current: ${prophet_result.current_price:.2f}")
        print(f"   24h Prediction: ${prophet_result.predicted_price_24h:.2f} ({prophet_result.change_24h_pct:+.2f}%)")
        print(f"   Signal: {prophet_result.signal.value}")
        print(f"   Confidence: {prophet_result.confidence:.1f}%")
    
    print("\n[FORECAST] TimeGPT Forecast:")
    timegpt_result = forecaster.forecast_with_timegpt(df)
    if timegpt_result:
        print(f"   Current: ${timegpt_result.current_price:.2f}")
        print(f"   24h Prediction: ${timegpt_result.predicted_price_24h:.2f} ({timegpt_result.change_24h_pct:+.2f}%)")
        print(f"   Signal: {timegpt_result.signal.value}")
        print(f"   Confidence: {timegpt_result.confidence:.1f}%")
    else:
        print("   [WARNING] Not available (API key required)")
    
    print("\n[TARGET] Ensemble Forecast:")
    ensemble_result = forecaster.forecast_ensemble(df)
    if ensemble_result:
        print(f"   Current: ${ensemble_result.current_price:.2f}")
        print(f"   24h Prediction: ${ensemble_result.predicted_price_24h:.2f} ({ensemble_result.change_24h_pct:+.2f}%)")
        print(f"   Signal: {ensemble_result.signal.value}")
        print(f"   Confidence: {ensemble_result.confidence:.1f}%")
    
    print("\n[OK] Test complete!")
