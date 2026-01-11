"""
Price Predictor V4 - Super-Ensemble & Gem Hunter
=============================================
1. Super-Ensemble: Combines Transformer + Bi-LSTM + Conv1D for deep analysis.
2. Gem Hunter: Heuristic scanner for new tokens (no history).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Union
from colorama import Fore, Style
from datetime import datetime
import warnings
import traceback

warnings.filterwarnings('ignore')

# Technical Analysis
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

# TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Bidirectional, 
                                          Input, Attention, MultiHeadAttention,
                                          LayerNormalization, GlobalAveragePooling1D,
                                          Conv1D, MaxPooling1D, Flatten, Add, Concatenate)
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.preprocessing import MinMaxScaler, RobustScaler
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print(f"{Fore.YELLOW}[WARNING] TensorFlow not available{Style.RESET_ALL}")


class FeatureEngineer:
    """Generate technical indicators"""
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        if not TA_AVAILABLE or 'close' not in df.columns:
            return df
        
        df = df.copy()
        try:
             # Basic Returns
            df['returns'] = df['close'].pct_change()
            
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd_diff()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
            
            # Volume
            if 'volume' in df.columns:
                df['volume_sma'] = df['volume'].rolling(window=10).mean()
                df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-6)
            
            # Fill NaN safely
            df = df.bfill().fillna(0)
            
        except Exception as e:
            print(f"Feature Eng Error: {e}")
            df = df.fillna(0)
            
        return df


class GemHunter:
    """
    Fallback Analysis for New Tokens (Zero History/404).
    Uses Snapshot Data (SNA, Liquidity, FDV) to score potential.
    """
    
    @staticmethod
    def analyze_token(token_info: Dict, sna_score: float) -> Dict:
        """Score a token based on static metrics"""
        try:
            # Extract metrics with defaults
            liquidity = float(token_info.get('liquidity', {}).get('usd', 0))
            fdv = float(token_info.get('fdv', 0))
            volume_24h = float(token_info.get('volume', {}).get('h24', 0))
            
            # Heuristic Scoring
            score = 0
            
            # 1. SNA Score (Social/Smart Money) - High Weight
            score += sna_score * 0.4
            
            # 2. Liquidity Health (Avoid too low or too high for moonshots)
            if 1000 < liquidity < 500000: # Sweet spot for gems
                score += 20
            elif liquidity >= 500000:
                score += 10 # Stable but less X potential
            else:
                score -= 20 # Rug risk
                
            # 3. Volume/Liquidity Ratio (Activity)
            if liquidity > 0:
                vol_liq_ratio = volume_24h / liquidity
                if vol_liq_ratio > 0.5: score += 15 # High activity
                elif vol_liq_ratio > 0.1: score += 10
            
            # 4. FDV Check (Undervalued?)
            if 0 < fdv < 1000000: # Micro cap
                score += 15
            
            # Normalize Score (0-100)
            final_score = min(max(score, 10), 95)
            
            # Estimate Pump Timeframe based on Hype (SNA)
            if sna_score > 80: hours = 1   # Imminent
            elif sna_score > 60: hours = 4
            else: hours = 12
            
            return {
                "confidence": round(final_score, 1),
                "pump_in_hours": hours,
                "model": "GEM_PATTERN",
                "message": "Analisis Pattern Dasar (Data History Kurang)"
            }
            
        except Exception as e:
            print(f"GemHunter Error: {e}")
            return {"confidence": 50, "pump_in_hours": 24, "model": "ERROR", "message": "Gagal Menganalisis"}


class SuperEnsemblePredictor:
    """
    Unified Deep Learning Engine.
    Trains Transformer, Bi-LSTM, and Conv1D internally.
    """
    
    def __init__(self, lookback: int = 24):
        self.lookback = lookback
        self.models = {} # Store sub-models
        self.is_trained = False
        self.scaler = RobustScaler() if TF_AVAILABLE else None
        self.feature_scaler = RobustScaler() if TF_AVAILABLE else None
        self.feature_cols = []

    def prepare_data(self, df: pd.DataFrame):
        df = FeatureEngineer.add_technical_indicators(df)
        self.feature_cols = [c for c in df.columns if c not in ['timestamp', 'open', 'high', 'low']]
        features = df[self.feature_cols].values
        target = df['close'].values.reshape(-1, 1)
        
        feat_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.scaler.fit_transform(target)
        
        X, y = [], []
        for i in range(self.lookback, len(feat_scaled)):
            X.append(feat_scaled[i-self.lookback:i])
            y.append(target_scaled[i, 0])
            
        return np.array(X), np.array(y)

    def _build_model_arch(self, arch: str, input_shape):
        inputs = Input(shape=input_shape)
        
        if arch == 'transformer':
            # Time-GPT Lite
            x = LayerNormalization(epsilon=1e-6)(inputs)
            x = MultiHeadAttention(key_dim=64, num_heads=4, dropout=0.1)(x, x)
            x = Dropout(0.1)(x)
            res = Add()([x, inputs])
            x = LayerNormalization(epsilon=1e-6)(res)
            x = Conv1D(filters=64, kernel_size=1, activation="relu")(x)
            x = GlobalAveragePooling1D()(x)
            
        elif arch == 'conv1d':
            # Pattern Recognition
            x = Conv1D(64, 3, activation='relu')(inputs)
            x = MaxPooling1D(2)(x)
            x = Flatten()(x)
            
        else: # 'bilstm' (Default)
            x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
            x = GlobalAveragePooling1D()(x)
            
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(0.001), loss='huber', metrics=['mae'])
        return model

    def train_ensemble(self, df: pd.DataFrame, epochs: int = 50) -> Dict:
        """Trains multiple architectures sequentially"""
        if not TF_AVAILABLE:
            self.is_trained = True
            return {"status": "simulated", "accuracy": 88.0}
            
        print(f"{Fore.CYAN}[AI-SUPER] Training Unified Ensemble (Transformer + LSTM + CNN)...{Style.RESET_ALL}")
        
        try:
            X, y = self.prepare_data(df)
            if len(X) < 10: return {"status": "insufficient_data"}
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train 3 variations
            accuracies = []
            for arch in ['transformer', 'bilstm', 'conv1d']:
                print(f"   > Training submodule: {arch}...")
                model = self._build_model_arch(arch, (self.lookback, X.shape[2]))
                
                hist = model.fit(
                    X_train, y_train, 
                    epochs=epochs, 
                    batch_size=16, 
                    validation_data=(X_val, y_val),
                    verbose=0,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
                )
                self.models[arch] = model
                
                # Est Accuracy
                val_mae = min(hist.history['val_mae'])
                accuracies.append(max(0, (1 - val_mae) * 100))
                
            self.is_trained = True
            avg_acc = np.mean(accuracies)
            print(f"{Fore.GREEN}[AI-SUPER] Ensemble Ready! Avg Accuracy: {avg_acc:.1f}%{Style.RESET_ALL}")
            return {"status": "success", "accuracy": avg_acc}
            
        except Exception as e:
            print(f"Ensemble Train Error: {e}")
            traceback.print_exc()
            return {"status": "error"}

    def predict(self, df: pd.DataFrame) -> Dict:
        """Predict using average of all models"""
        if not self.is_trained or not TF_AVAILABLE:
             return {"error": "Model not trained"}
             
        try:
            df_feat = FeatureEngineer.add_technical_indicators(df)
            features = df_feat[self.feature_cols].values[-self.lookback:]
            
            if len(features) < self.lookback:
                 padding = np.tile(features[0], (self.lookback - len(features), 1))
                 features = np.vstack([padding, features])
                 
            features_scaled = self.feature_scaler.transform(features)
            X = features_scaled.reshape(1, self.lookback, -1)
            
            # Predict with all models
            preds = []
            for name, model in self.models.items():
                p = model.predict(X, verbose=0)[0, 0]
                preds.append(p)
            
            # Average Prediction
            avg_pred_scaled = np.mean(preds)
            pred_price = self.scaler.inverse_transform([[avg_pred_scaled]])[0][0]
            current_price = df['close'].iloc[-1]
            pump_pct = (pred_price - current_price) / current_price * 100
            
            # Calc Confidence
            divergence = np.std(preds) # High divergence = low confidence
            confidence = max(50, 95 - (divergence * 50))
            
            return {
                "confidence": confidence,
                "pump_in_hours": 4 if pump_pct > 2 else 12, # Simplified logic for now
                "predicted_price": pred_price,
                "model": "SUPER_ENSEMBLE",
                "message": "Deep Learning Multi-Model Consensus"
            }
            
        except Exception as e:
            print(f"Ensemble Predict Error: {e}")
            return {"error": str(e)}


# Wrapper
def analyze_token_comprehensive(hist_data: pd.DataFrame, token_info: Dict, sna_score: float) -> Dict:
    """
    Main Entry Point: Decides between Deep Learning or GemHunter
    """
    # CASE 1: No Historical Data -> Gem Hunter
    if hist_data.empty or len(hist_data) < 5:
        print(f"{Fore.YELLOW}[ANALYSIS] No history. Switching to GEM HUNTER mode.{Style.RESET_ALL}")
        return GemHunter.analyze_token(token_info, sna_score)
        
    # CASE 2: Sufficient Data -> Super Ensemble
    print(f"{Fore.CYAN}[ANALYSIS] Historical Data Found ({len(hist_data)} candles). Running SUPER ENSEMBLE.{Style.RESET_ALL}")
    
    predictor = SuperEnsemblePredictor()
    train_res = predictor.train_ensemble(hist_data, epochs=30)
    
    if train_res['status'] != 'success':
         # Fallback if training fails
         return GemHunter.analyze_token(token_info, sna_score)
         
    pred = predictor.predict(hist_data)
    if "error" in pred:
        return GemHunter.analyze_token(token_info, sna_score)
        
    # Enhance prediction with SNA
    final_conf = (pred['confidence'] * 0.7) + (sna_score * 0.3)
    
    return {
        "confidence": min(final_conf, 99),
        "pump_in_hours": pred['pump_in_hours'],
        "model": "SUPER_ENSEMBLE",
        "message": f"Deep Learning Analysis (Acc: {train_res.get('accuracy',0):.0f}%)"
    }
