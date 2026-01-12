"""
Indodax AI Forecaster V2.0
==========================
Multi-Model Deep Learning untuk prediksi harga crypto Indodax

Data Sources:
- Binance (primary, 1000+ candles)
- CryptoCompare (backup)
- CoinGecko (additional)
- Indodax (Indonesian market)

Models:
1. LSTM - Long Short-Term Memory
2. Bi-LSTM - Bidirectional LSTM
3. GRU - Gated Recurrent Unit
4. Conv1D - 1D Convolutional Neural Network
5. Transformer - Attention-based model (Time-GPT style)

Ensemble averaging untuk akurasi maksimal
Higher epochs (100-200) untuk training lebih baik
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
import time
import requests

warnings.filterwarnings('ignore')

# Import data aggregator
try:
    from modules.crypto_data_aggregator import CryptoDataAggregator, get_aggregated_data
    AGGREGATOR_AVAILABLE = True
except ImportError:
    try:
        from crypto_data_aggregator import CryptoDataAggregator, get_aggregated_data
        AGGREGATOR_AVAILABLE = True
    except ImportError:
        AGGREGATOR_AVAILABLE = False
        print("[WARNING] Data aggregator not available, using Indodax only")

# ML Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import (
        LSTM, GRU, Dense, Dropout, Input, Bidirectional,
        Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D,
        MultiHeadAttention, LayerNormalization, Add,
        BatchNormalization, Concatenate
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not available")


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
    current_price: float
    predicted_price_1h: float
    predicted_price_4h: float
    predicted_price_24h: float
    change_1h_pct: float
    change_4h_pct: float
    change_24h_pct: float
    signal: ForecastSignal
    accuracy: float  # Changed from confidence to accuracy
    model_scores: Dict[str, float]
    model_accuracies: Dict[str, float]  # Individual model accuracies
    risk_level: str
    recommendation: str
    data_sources: List[str]  # Sources used for training
    total_candles: int  # Total data points used


class IndodaxHistoricalAPI:
    """
    Fetch historical data from Indodax for training
    """
    
    BASE_URL = "https://indodax.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0',
            'Accept': 'application/json'
        })
    
    def get_trades_history(self, pair: str, limit: int = 500) -> pd.DataFrame:
        """Get recent trades for a pair"""
        try:
            # Normalize pair format: btc_idr -> btcidr
            pair_normalized = pair.replace('_', '')
            
            url = f"{self.BASE_URL}/trades/{pair_normalized}"
            print(f"[INDODAX-HIST] Fetching trades from {url}")
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, list):
                    trades = data
                elif isinstance(data, dict) and 'error' not in data:
                    # Could be dict with trades array
                    trades = data.get('trades', data.get('data', []))
                    if not trades and len(data) > 0:
                        # Try to use dict values if list-like structure
                        trades = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
                else:
                    trades = []
                
                if trades and len(trades) > 0:
                    df = pd.DataFrame(trades)
                    
                    # Convert date from unix timestamp
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'].astype(int), unit='s')
                    
                    # Convert price and amount to float
                    if 'price' in df.columns:
                        df['price'] = df['price'].astype(float)
                    if 'amount' in df.columns:
                        df['amount'] = df['amount'].astype(float)
                    
                    df = df.sort_values('date')
                    print(f"[INDODAX-HIST] Got {len(df)} trades")
                    return df
                    
        except Exception as e:
            print(f"[INDODAX-HIST] Error fetching trades: {e}")
            import traceback
            traceback.print_exc()
        
        return pd.DataFrame()
    
    def get_ticker_history(self, pair: str) -> Dict:
        """Get current ticker data"""
        try:
            url = f"{self.BASE_URL}/ticker/{pair}"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                return response.json().get('ticker', {})
        except Exception as e:
            print(f"[INDODAX-HIST] Error fetching ticker: {e}")
        
        return {}
    
    def generate_ohlcv_from_trades(self, trades_df: pd.DataFrame, interval: str = '5min') -> pd.DataFrame:
        """Generate OHLCV candles from trades"""
        if trades_df.empty:
            print("[INDODAX-HIST] Empty trades dataframe")
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying original
            df = trades_df.copy()
            
            # Ensure date column exists and is datetime
            if 'date' not in df.columns:
                print("[INDODAX-HIST] No date column in trades")
                return pd.DataFrame()
            
            df = df.set_index('date')
            
            # Check if price column exists
            if 'price' not in df.columns:
                print("[INDODAX-HIST] No price column in trades")
                return pd.DataFrame()
            
            # Resample to create OHLCV
            ohlcv = df['price'].resample(interval).ohlc()
            
            # Volume - use amount if exists
            if 'amount' in df.columns:
                ohlcv['volume'] = df['amount'].resample(interval).sum()
            else:
                ohlcv['volume'] = 1.0  # Default volume
            
            ohlcv = ohlcv.dropna()
            ohlcv = ohlcv.reset_index()
            ohlcv.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            print(f"[INDODAX-HIST] Generated {len(ohlcv)} OHLCV candles")
            
            return ohlcv
        except Exception as e:
            print(f"[INDODAX-HIST] Error generating OHLCV: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()


class FeatureEngineer:
    """Generate technical indicators for ML"""
    
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators"""
        if df.empty or 'close' not in df.columns:
            return df
        
        df = df.copy()
        
        try:
            # Price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Moving averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_10'] = df['close'].rolling(window=10).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_5'] = df['close'].ewm(span=5).mean()
            df['ema_10'] = df['close'].ewm(span=10).mean()
            
            # Price relative to MA
            df['price_sma5_ratio'] = df['close'] / df['sma_5']
            df['price_sma10_ratio'] = df['close'] / df['sma_10']
            
            # Volatility
            df['volatility'] = df['returns'].rolling(window=10).std()
            df['high_low_ratio'] = (df['high'] - df['low']) / df['low']
            
            # Momentum
            df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
            df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume features
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Fill NaN
            df = df.bfill().fillna(0)
            
            # Replace infinities
            df = df.replace([np.inf, -np.inf], 0)
            
        except Exception as e:
            print(f"[FEATURE] Error: {e}")
            df = df.fillna(0)
        
        return df


class DeepLearningForecaster:
    """
    Multi-Model Deep Learning Forecaster
    Combines LSTM, Bi-LSTM, GRU, Conv1D, Transformer
    """
    
    def __init__(self, lookback: int = 15, forecast_horizon: int = 6):
        """
        Args:
            lookback: Number of past time steps to use (reduced for Indodax data)
            forecast_horizon: Number of future steps to predict
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.scalers = {
            'features': RobustScaler(),
            'target': MinMaxScaler()
        }
        self.is_trained = False
        self.feature_cols = []
        self.training_history = {}
    
    def _build_lstm_model(self, input_shape: Tuple) -> Model:
        """Standard LSTM model"""
        inputs = Input(shape=input_shape)
        x = LSTM(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model
    
    def _build_bilstm_model(self, input_shape: Tuple) -> Model:
        """Bidirectional LSTM model"""
        inputs = Input(shape=input_shape)
        x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        x = Dropout(0.2)(x)
        x = Bidirectional(LSTM(32, return_sequences=False))(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model
    
    def _build_gru_model(self, input_shape: Tuple) -> Model:
        """GRU model"""
        inputs = Input(shape=input_shape)
        x = GRU(64, return_sequences=True)(inputs)
        x = Dropout(0.2)(x)
        x = GRU(32, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(16, activation='relu')(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model
    
    def _build_conv1d_model(self, input_shape: Tuple) -> Model:
        """1D Convolutional model for pattern recognition"""
        inputs = Input(shape=input_shape)
        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model
    
    def _build_transformer_model(self, input_shape: Tuple) -> Model:
        """Transformer model (Time-GPT style)"""
        inputs = Input(shape=input_shape)
        
        # Positional encoding would go here in full implementation
        x = inputs
        
        # Multi-head attention block
        x = LayerNormalization(epsilon=1e-6)(x)
        attention_output = MultiHeadAttention(
            key_dim=32, num_heads=4, dropout=0.1
        )(x, x)
        x = Add()([x, attention_output])
        
        # Feed forward
        x = LayerNormalization(epsilon=1e-6)(x)
        ff = Dense(64, activation='relu')(x)
        ff = Dropout(0.1)(ff)
        ff = Dense(input_shape[-1])(ff)
        x = Add()([x, ff])
        
        # Output
        x = GlobalAveragePooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.forecast_horizon)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        # Add features
        df = FeatureEngineer.add_features(df)
        
        # Select feature columns
        exclude_cols = ['timestamp', 'date', 'open', 'high', 'low']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Scale features
        features = df[self.feature_cols].values
        target = df['close'].values.reshape(-1, 1)
        
        features_scaled = self.scalers['features'].fit_transform(features)
        target_scaled = self.scalers['target'].fit_transform(target)
        
        # Create sequences
        X, y = [], []
        for i in range(self.lookback, len(features_scaled) - self.forecast_horizon):
            X.append(features_scaled[i-self.lookback:i])
            y.append(target_scaled[i:i+self.forecast_horizon].flatten())
        
        return np.array(X), np.array(y)
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train all models with higher epochs for better accuracy"""
        if not TF_AVAILABLE:
            return {"status": "error", "message": "TensorFlow not available"}
        
        min_required = self.lookback + self.forecast_horizon + 5
        if len(df) < min_required:
            return {"status": "error", "message": f"Insufficient data: need {min_required}, got {len(df)}"}
        
        print(f"[AI-FORECAST] Training ensemble on {len(df)} data points with {epochs} epochs...")
        
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            
            if len(X) < 5:
                return {"status": "error", "message": f"Not enough sequences: {len(X)}"}
            
            # Split data - 80% train, 20% validation
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )
            
            input_shape = (X.shape[1], X.shape[2])
            
            # Callbacks - more patience for higher epochs
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
            ]
            
            # Train each model
            model_builders = {
                'lstm': self._build_lstm_model,
                'bilstm': self._build_bilstm_model,
                'gru': self._build_gru_model,
                'conv1d': self._build_conv1d_model,
                'transformer': self._build_transformer_model
            }
            
            accuracies = {}
            val_predictions = {}
            
            for name, builder in model_builders.items():
                print(f"   > Training {name.upper()}...")
                model = builder(input_shape)
                
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=0
                )
                
                self.models[name] = model
                
                # Calculate REAL accuracy using validation set
                y_pred = model.predict(X_val, verbose=0)
                
                # Inverse transform predictions
                y_pred_inv = self.scalers['target'].inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_val_inv = self.scalers['target'].inverse_transform(y_val.reshape(-1, 1)).flatten()
                
                # Calculate MAPE (Mean Absolute Percentage Error)
                # Accuracy = 100 - MAPE
                mape = np.mean(np.abs((y_val_inv - y_pred_inv) / (y_val_inv + 1e-10))) * 100
                accuracy = max(0, min(99.9, 100 - mape))
                
                accuracies[name] = round(accuracy, 2)
                val_predictions[name] = y_pred
                
                self.training_history[name] = history.history
                
                print(f"      {name.upper()} accuracy: {accuracy:.2f}%")
            
            self.is_trained = True
            self.model_accuracies = accuracies
            
            avg_accuracy = np.mean(list(accuracies.values()))
            print(f"[AI-FORECAST] Ensemble trained! Average accuracy: {avg_accuracy:.2f}%")
            
            return {
                "status": "success",
                "accuracies": accuracies,
                "avg_accuracy": round(avg_accuracy, 2),
                "samples": len(X),
                "train_samples": len(X_train),
                "val_samples": len(X_val)
            }
            
        except Exception as e:
            print(f"[AI-FORECAST] Training error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict future prices using ensemble"""
        if not self.is_trained:
            return {"error": "Model not trained"}
        
        try:
            # Prepare latest data
            df = FeatureEngineer.add_features(df)
            features = df[self.feature_cols].values[-self.lookback:]
            
            if len(features) < self.lookback:
                # Pad if needed
                padding = np.tile(features[0], (self.lookback - len(features), 1))
                features = np.vstack([padding, features])
            
            features_scaled = self.scalers['features'].transform(features)
            X = features_scaled.reshape(1, self.lookback, -1)
            
            # Predict with all models
            predictions = {}
            for name, model in self.models.items():
                pred_scaled = model.predict(X, verbose=0)[0]
                # Inverse transform
                pred = self.scalers['target'].inverse_transform(
                    pred_scaled.reshape(-1, 1)
                ).flatten()
                predictions[name] = pred
            
            # Ensemble average
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            
            # Calculate confidence based on model agreement
            std_pred = np.std(list(predictions.values()), axis=0)
            mean_std = np.mean(std_pred)
            confidence = max(50, min(98, 95 - mean_std * 10))
            
            return {
                "ensemble": ensemble_pred,
                "individual": predictions,
                "confidence": confidence,
                "model_accuracies": getattr(self, 'model_accuracies', {})
            }
            
        except Exception as e:
            print(f"[AI-FORECAST] Prediction error: {e}")
            return {"error": str(e)}


class IndodaxAIForecaster:
    """
    Main class for Indodax AI forecasting
    Uses multi-source data aggregator for big data training
    """
    
    def __init__(self):
        self.historical_api = IndodaxHistoricalAPI()
        self.forecaster = DeepLearningForecaster(lookback=30, forecast_horizon=12)
        self.trained_pairs = {}  # Cache trained models per pair
        self.last_predictions = {}  # Cache predictions
        self.cache_duration = 600  # 10 minutes cache (longer because training takes time)
        
        # Initialize data aggregator
        if AGGREGATOR_AVAILABLE:
            self.data_aggregator = CryptoDataAggregator()
            print("[AI] Multi-source data aggregator initialized")
        else:
            self.data_aggregator = None
            print("[AI] Using Indodax data only")
    
    def train_for_pair(self, pair: str) -> Dict:
        """Train model for a specific pair using aggregated big data"""
        symbol = pair.replace('_idr', '').upper()
        print(f"\n[AI] ====== Training AI Model for {symbol} ======")
        
        ohlcv = None
        sources_used = []
        total_candles = 0
        
        # Try to get big data from aggregator first
        if self.data_aggregator:
            print(f"[AI] Fetching big data from multiple sources...")
            try:
                aggregated = self.data_aggregator.aggregate_data(symbol, interval='1h')
                
                if aggregated.total_candles >= 100:
                    ohlcv = aggregated.ohlcv
                    sources_used = aggregated.sources_used
                    total_candles = aggregated.total_candles
                    print(f"[AI] Got {total_candles} candles from: {', '.join(sources_used)}")
            except Exception as e:
                print(f"[AI] Aggregator error: {e}")
        
        # Fallback to Indodax if aggregator failed or not enough data
        if ohlcv is None or len(ohlcv) < 100:
            print(f"[AI] Falling back to Indodax data...")
            trades_df = self.historical_api.get_trades_history(pair, limit=1000)
            
            if trades_df.empty:
                return {"status": "error", "message": "No historical data from any source"}
            
            ohlcv = self.historical_api.generate_ohlcv_from_trades(trades_df, interval='1min')
            sources_used = ['Indodax']
            total_candles = len(ohlcv)
        
        if ohlcv is None or len(ohlcv) < 50:
            return {"status": "error", "message": f"Insufficient data: only {len(ohlcv) if ohlcv is not None else 0} candles"}
        
        print(f"[AI] Training with {len(ohlcv)} candles...")
        
        # Determine epochs based on data size
        if len(ohlcv) >= 500:
            epochs = 150  # More data = more epochs
        elif len(ohlcv) >= 200:
            epochs = 100
        else:
            epochs = 50
        
        # Train with higher epochs
        result = self.forecaster.train(ohlcv, epochs=epochs, batch_size=16)
        
        if result.get('status') == 'success':
            self.trained_pairs[pair] = {
                'trained_at': time.time(),
                'accuracy': result.get('avg_accuracy', 0),
                'accuracies': result.get('accuracies', {}),
                'sources': sources_used,
                'total_candles': total_candles
            }
            print(f"[AI] ====== Training Complete! Accuracy: {result.get('avg_accuracy', 0):.2f}% ======\n")
        
        return result
    
    def forecast_pair(self, pair: str, current_price: float) -> Optional[ForecastResult]:
        """Generate forecast for a pair"""
        
        # Check cache
        cache_key = f"{pair}_{int(time.time() // self.cache_duration)}"
        if cache_key in self.last_predictions:
            return self.last_predictions[cache_key]
        
        # Train if needed
        if pair not in self.trained_pairs:
            train_result = self.train_for_pair(pair)
            if train_result.get('status') != 'success':
                return None
        
        # Get training info
        training_info = self.trained_pairs.get(pair, {})
        accuracy = training_info.get('accuracy', 0)
        model_accuracies = training_info.get('accuracies', {})
        data_sources = training_info.get('sources', ['Indodax'])
        total_candles = training_info.get('total_candles', 0)
        
        # Get latest data for prediction (use aggregated if available)
        symbol = pair.replace('_idr', '').upper()
        ohlcv = None
        
        if self.data_aggregator:
            try:
                aggregated = self.data_aggregator.aggregate_data(symbol, interval='1h')
                if aggregated.total_candles >= 30:
                    ohlcv = aggregated.ohlcv
            except:
                pass
        
        if ohlcv is None or ohlcv.empty:
            trades_df = self.historical_api.get_trades_history(pair, limit=500)
            if trades_df.empty:
                return None
            ohlcv = self.historical_api.generate_ohlcv_from_trades(trades_df, interval='1min')
        
        if ohlcv is None or ohlcv.empty:
            return None
        
        # Predict
        pred_result = self.forecaster.predict(ohlcv)
        
        if 'error' in pred_result:
            return None
        
        ensemble_pred = pred_result['ensemble']
        
        # Get the last known price from training data (in USD if from external source)
        # This is the price scale used by the model
        last_model_price = float(ohlcv['close'].iloc[-1])
        
        # Extract predictions at different horizons
        pred_1h_usd = ensemble_pred[min(3, len(ensemble_pred)-1)]
        pred_4h_usd = ensemble_pred[min(7, len(ensemble_pred)-1)]
        pred_24h_usd = ensemble_pred[-1]
        
        # Calculate percentage changes based on model's price scale
        # This works regardless of currency (USD or IDR)
        change_1h = ((pred_1h_usd - last_model_price) / last_model_price) * 100
        change_4h = ((pred_4h_usd - last_model_price) / last_model_price) * 100
        change_24h = ((pred_24h_usd - last_model_price) / last_model_price) * 100
        
        # Convert predictions to IDR using current_price
        # pred_idr = current_price * (1 + change_pct/100)
        pred_1h = current_price * (1 + change_1h / 100)
        pred_4h = current_price * (1 + change_4h / 100)
        pred_24h = current_price * (1 + change_24h / 100)
        
        # Determine signal based on prediction and accuracy
        avg_change = (change_1h + change_4h + change_24h) / 3
        
        # Adjusted thresholds for realistic crypto movements
        # BTC typically moves 1-3% daily, altcoins can move 5-15%
        if avg_change > 3 and accuracy > 75:
            signal = ForecastSignal.STRONG_BUY
            risk = "HIGH"
            rec = f"🔥 STRONG BUY - AI ({accuracy:.1f}% accuracy) predicts significant pump!"
        elif avg_change > 1.5 and accuracy > 70:
            signal = ForecastSignal.BUY
            risk = "MEDIUM"
            rec = f"📈 BUY - Positive momentum expected (Accuracy: {accuracy:.1f}%)"
        elif avg_change < -3 and accuracy > 75:
            signal = ForecastSignal.STRONG_SELL
            risk = "HIGH"
            rec = f"⚠️ STRONG SELL - AI ({accuracy:.1f}% accuracy) predicts dump!"
        elif avg_change < -1.5 and accuracy > 70:
            signal = ForecastSignal.SELL
            risk = "MEDIUM"
            rec = f"📉 SELL - Negative trend expected (Accuracy: {accuracy:.1f}%)"
        else:
            signal = ForecastSignal.HOLD
            risk = "LOW"
            rec = f"➡️ HOLD - No clear direction (Accuracy: {accuracy:.1f}%)"
        
        # Model scores (predicted % change based on model's scale)
        model_scores = {}
        for name, preds in pred_result.get('individual', {}).items():
            last_pred = preds[-1]
            # Use model's price scale for accurate % change
            model_change = ((last_pred - last_model_price) / last_model_price) * 100
            model_scores[name] = float(round(model_change, 2))
        
        result = ForecastResult(
            symbol=pair.replace('_idr', '').upper(),
            current_price=float(current_price),
            predicted_price_1h=float(round(pred_1h, 2)),
            predicted_price_4h=float(round(pred_4h, 2)),
            predicted_price_24h=float(round(pred_24h, 2)),
            change_1h_pct=float(round(change_1h, 2)),
            change_4h_pct=float(round(change_4h, 2)),
            change_24h_pct=float(round(change_24h, 2)),
            signal=signal,
            accuracy=float(round(accuracy, 2)),  # Real accuracy
            model_scores=model_scores,
            model_accuracies={k: float(round(v, 2)) for k, v in model_accuracies.items()},
            risk_level=risk,
            recommendation=rec,
            data_sources=data_sources,
            total_candles=total_candles
        )
        
        # Cache result
        self.last_predictions[cache_key] = result
        
        return result
    
    def to_dict(self, result: ForecastResult) -> Dict:
        """Convert forecast result to dictionary - ensure all values are JSON serializable"""
        return {
            'symbol': str(result.symbol),
            'current_price': float(result.current_price),
            'predicted_price_1h': float(result.predicted_price_1h),
            'predicted_price_4h': float(result.predicted_price_4h),
            'predicted_price_24h': float(result.predicted_price_24h),
            'change_1h_pct': float(result.change_1h_pct),
            'change_4h_pct': float(result.change_4h_pct),
            'change_24h_pct': float(result.change_24h_pct),
            'signal': str(result.signal.value),
            'signal_type': str(result.signal.name),
            'accuracy': float(result.accuracy),  # Real accuracy instead of confidence
            'model_scores': {k: float(v) for k, v in result.model_scores.items()},
            'model_accuracies': {k: float(v) for k, v in result.model_accuracies.items()},
            'risk_level': str(result.risk_level),
            'recommendation': str(result.recommendation),
            'data_sources': result.data_sources,
            'total_candles': int(result.total_candles)
        }


# Global instance
indodax_forecaster = IndodaxAIForecaster()


def forecast_indodax_token(pair: str, current_price: float) -> Optional[Dict]:
    """API function to forecast a token"""
    result = indodax_forecaster.forecast_pair(pair, current_price)
    if result:
        return indodax_forecaster.to_dict(result)
    return None


# Test
if __name__ == "__main__":
    print("\n=== INDODAX AI FORECASTER TEST ===\n")
    
    forecaster = IndodaxAIForecaster()
    
    # Test with BTC
    result = forecaster.forecast_pair("btc_idr", 1_600_000_000)
    
    if result:
        print(f"Symbol: {result.symbol}")
        print(f"Current: Rp {result.current_price:,.0f}")
        print(f"Predicted 1h: Rp {result.predicted_price_1h:,.0f} ({result.change_1h_pct:+.2f}%)")
        print(f"Predicted 4h: Rp {result.predicted_price_4h:,.0f} ({result.change_4h_pct:+.2f}%)")
        print(f"Signal: {result.signal.value}")
        print(f"Confidence: {result.confidence}%")
        print(f"Recommendation: {result.recommendation}")
    else:
        print("Failed to generate forecast")
