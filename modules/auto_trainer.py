"""
Auto Training Scheduler V1.0
============================
Sistem otomatis untuk melatih ulang model AI secara berkala
agar prediksi selalu up-to-date dengan data terbaru.

Features:
- Background training scheduler
- Prioritas training berdasarkan volume/popularity
- Model caching untuk efisiensi
- Auto-update setiap interval tertentu
"""

import threading
import time
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os

# Training status
training_status = {
    "is_running": False,
    "current_coin": None,
    "last_trained": {},
    "training_queue": [],
    "total_trained_today": 0,
    "errors": []
}

# Training mode: "smart" (dynamic) or "fixed" (static list)
TRAINING_MODE = os.getenv('TRAINING_MODE', 'smart')  # Default to smart mode

# Coins that are ALWAYS trained regardless of mode
ALWAYS_TRAIN = ["btc", "eth", "sol", "xrp", "doge"]

# Old static list (used when TRAINING_MODE = "fixed")
FIXED_PRIORITY_COINS = [
    "btc", "eth", "sol", "xrp", "doge", "ada", "avax", "dot", 
    "link", "matic", "shib", "pepe", "uni", "atom", "near",
    "arb", "op", "apt", "sui", "inj", "sei", "tia", "jup"
]

# Smart mode configuration
MAX_TRAINING_COINS = 25
MIN_VOLUME_24H = 10_000_000   # 10M IDR minimum (lowered for more coins)
MIN_MARKET_CAP = 5_000_000    # 5M IDR minimum (lowered for more coins)

# Training interval in minutes
TRAINING_INTERVAL = 30  # Retrain setiap 30 menit
MAX_TRAINING_AGE = 60  # Model dianggap stale setelah 60 menit


class AutoTrainer:
    """
    Auto Training Manager
    Mengelola training otomatis model AI
    """
    
    def __init__(self):
        self.is_running = False
        self.scheduler_thread = None
        self.training_lock = threading.Lock()
        self.forecaster = None
        self.last_trained: Dict[str, datetime] = {}
        self.training_results: Dict[str, Dict] = {}
        self.status_file = "training_status.json"
        
        # Load previous status if exists
        self._load_status()
        
        print("[AUTO-TRAINER] Initialized")
    
    def _load_status(self):
        """Load training status from file"""
        try:
            if os.path.exists(self.status_file):
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.last_trained = {
                        k: datetime.fromisoformat(v) 
                        for k, v in data.get('last_trained', {}).items()
                    }
                    self.training_results = data.get('training_results', {})
                    print(f"[AUTO-TRAINER] Loaded status: {len(self.last_trained)} coins")
        except Exception as e:
            print(f"[AUTO-TRAINER] Error loading status: {e}")
    
    def _save_status(self):
        """Save training status to file"""
        try:
            data = {
                'last_trained': {
                    k: v.isoformat() for k, v in self.last_trained.items()
                },
                'training_results': self.training_results,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[AUTO-TRAINER] Error saving status: {e}")
    
    def _get_forecaster(self):
        """Lazy load forecaster"""
        if self.forecaster is None:
            try:
                from modules.indodax_forecaster import IndodaxAIForecaster
                self.forecaster = IndodaxAIForecaster()
            except ImportError:
                from indodax_forecaster import IndodaxAIForecaster
                self.forecaster = IndodaxAIForecaster()
        return self.forecaster
    
    def calculate_training_priority(self, coin_data: dict) -> float:
        """
        Calculate priority score for training a coin
        Higher score = higher priority
        """
        try:
            volume_24h = float(coin_data.get('volume_24h_idr', 0))
            market_cap = float(coin_data.get('market_cap', 0))
            change_24h = float(coin_data.get('change_24h', 0))
            volatility = abs(change_24h)
            
            # Weighted scoring
            # Volume: 40%, Market Cap: 30%, Volatility: 20%, Base: 10%
            score = (
                (volume_24h / 1e12) * 40 +      # Normalize to trillions
                (market_cap / 1e12) * 30 +
                min(volatility, 20) * 20 +       # Cap volatility at 20%
                10                                # Base score
            )
            
            return score
        except:
            return 0.0
    
    def get_smart_selection(self) -> List[str]:
        """
        Smart dynamic coin selection based on Indodax market data
        Returns top coins by volume, market cap, and activity
        """
        try:
            # Get Indodax market data
            from modules.indodax_api import get_indodax_market
            market_data = get_indodax_market()
            
            if not market_data:
                print("[AUTO-TRAINER] No market data, using always-train list")
                return ALWAYS_TRAIN[:]
            
            # Score and filter coins
            scored_coins = []
            for coin in market_data:
                symbol = coin.get('symbol', '').replace('_IDR', '').lower()
                
                # Get volume (use 24h volume in IDR)
                volume = float(coin.get('volume_24h_idr', 0))
                if volume == 0:
                    # Try alternative volume field
                    volume = float(coin.get('volume', 0))
                
                # Skip very low volume coins
                if volume < MIN_VOLUME_24H:
                    continue
                
                # Calculate simplified score (volume-based since market_cap often missing)
                try:
                    score = self.calculate_training_priority(coin)
                except:
                    # Fallback to volume-only scoring
                    score = (volume / 1e12) * 100
                
                scored_coins.append((symbol, score, volume))
            
            # Sort by score descending
            scored_coins.sort(key=lambda x: x[1], reverse=True)
            
            # Get top N coins
            selected = [c[0] for c in scored_coins[:MAX_TRAINING_COINS]]
            
            # Ensure ALWAYS_TRAIN coins are included
            for coin in ALWAYS_TRAIN:
                if coin not in selected:
                    selected.insert(0, coin)
            
            # Trim to max
            selected = selected[:MAX_TRAINING_COINS]
            
            print(f"[AUTO-TRAINER] Smart selection: {len(selected)} coins from {len(scored_coins)} qualified")
            if scored_coins:
                print(f"[AUTO-TRAINER] Top 5 by score: {[c[0] for c in scored_coins[:5]]}")
            return selected
            
        except Exception as e:
            print(f"[AUTO-TRAINER] Smart selection error: {e}, using always-train list")
            import traceback
            traceback.print_exc()
            return ALWAYS_TRAIN[:]
    
    def needs_training(self, coin: str) -> bool:
        """Check if coin needs retraining"""
        if coin not in self.last_trained:
            return True
        
        age = datetime.now() - self.last_trained[coin]
        return age.total_seconds() > (MAX_TRAINING_AGE * 60)
    
    
    def get_training_queue(self) -> List[str]:
        """Get list of coins that need training, prioritized"""
        queue = []
        
        # Select coins based on training mode
        if TRAINING_MODE == 'smart':
            print("[AUTO-TRAINER] Using SMART mode (dynamic selection)")
            priority_coins = self.get_smart_selection()
        else:
            print("[AUTO-TRAINER] Using FIXED mode (static list)")
            priority_coins = FIXED_PRIORITY_COINS
        
        # Add priority coins that need training
        for coin in priority_coins:
            if self.needs_training(coin):
                queue.append(coin)
        
        return queue
    
    def train_coin(self, coin: str) -> Dict:
        """Train a single coin"""
        global training_status
        
        with self.training_lock:
            training_status["is_running"] = True
            training_status["current_coin"] = coin
            
            try:
                print(f"\n[AUTO-TRAINER] ========== Training {coin.upper()} ==========")
                
                forecaster = self._get_forecaster()
                pair = f"{coin}_idr"
                
                # Train the model
                result = forecaster.train_for_pair(pair)
                
                if result.get('status') == 'success':
                    self.last_trained[coin] = datetime.now()
                    self.training_results[coin] = {
                        'accuracy': result.get('avg_accuracy', 0),
                        'trained_at': datetime.now().isoformat(),
                        'candles': result.get('data_points', 0),
                        'sources': result.get('sources', [])
                    }
                    training_status["total_trained_today"] += 1
                    self._save_status()
                    
                    print(f"[AUTO-TRAINER] [OK] {coin.upper()} trained successfully! Accuracy: {result.get('avg_accuracy', 0):.2f}%")
                    return {"success": True, "accuracy": result.get('avg_accuracy', 0)}
                else:
                    error_msg = result.get('message', 'Unknown error')
                    training_status["errors"].append({
                        'coin': coin,
                        'error': error_msg,
                        'time': datetime.now().isoformat()
                    })
                    print(f"[AUTO-TRAINER] [ERROR] {coin.upper()} training failed: {error_msg}")
                    return {"success": False, "error": error_msg}
                    
            except Exception as e:
                error_msg = str(e)
                training_status["errors"].append({
                    'coin': coin,
                    'error': error_msg,
                    'time': datetime.now().isoformat()
                })
                print(f"[AUTO-TRAINER] [ERROR] {coin.upper()} error: {e}")
                return {"success": False, "error": error_msg}
            finally:
                training_status["is_running"] = False
                training_status["current_coin"] = None
    
    def run_training_cycle(self):
        """Run one training cycle for all coins that need it"""
        print(f"\n[AUTO-TRAINER] ======== Starting Training Cycle at {datetime.now().strftime('%H:%M:%S')} ========")
        
        queue = self.get_training_queue()
        
        if not queue:
            print("[AUTO-TRAINER] All models are up-to-date!")
            return
        
        print(f"[AUTO-TRAINER] Training queue: {', '.join(queue)}")
        
        for coin in queue:
            if not self.is_running:
                print("[AUTO-TRAINER] Training stopped by user")
                break
            
            self.train_coin(coin)
            
            # Small delay between trainings to prevent overload
            time.sleep(5)
        
        print(f"[AUTO-TRAINER] ======== Training Cycle Complete ========\n")
    
    def _scheduler_loop(self):
        """Background scheduler loop"""
        print("[AUTO-TRAINER] [START] Scheduler started!")
        
        # Run initial training cycle
        self.run_training_cycle()
        
        # Schedule periodic training
        schedule.every(TRAINING_INTERVAL).minutes.do(self.run_training_cycle)
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(10)
        
        print("[AUTO-TRAINER] Scheduler stopped")
    
    def start(self):
        """Start the auto trainer in background"""
        if self.is_running:
            print("[AUTO-TRAINER] Already running!")
            return False
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        
        print(f"[AUTO-TRAINER] [OK] Started! Training every {TRAINING_INTERVAL} minutes")
        return True
    
    def stop(self):
        """Stop the auto trainer"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        print("[AUTO-TRAINER] Stopped")
        return True
    
    def get_status(self) -> Dict:
        """Get current training status"""
        global training_status
        
        return {
            "is_running": self.is_running,
            "current_training": training_status.get("current_coin"),
            "total_trained_today": training_status.get("total_trained_today", 0),
            "training_interval_minutes": TRAINING_INTERVAL,
            "last_trained": {
                k: v.isoformat() for k, v in self.last_trained.items()
            },
            "training_results": self.training_results,
            "queue": self.get_training_queue(),
            "recent_errors": training_status.get("errors", [])[-5:],  # Last 5 errors
            "priority_coins": PRIORITY_COINS
        }
    
    def force_train(self, coin: str) -> Dict:
        """Force train a specific coin immediately"""
        return self.train_coin(coin.lower())


# Global instance
auto_trainer = AutoTrainer()


def start_auto_training():
    """Start auto training"""
    return auto_trainer.start()


def stop_auto_training():
    """Stop auto training"""
    return auto_trainer.stop()


def get_training_status():
    """Get training status"""
    return auto_trainer.get_status()


def force_train_coin(coin: str):
    """Force train a specific coin"""
    return auto_trainer.force_train(coin)


# Test
if __name__ == "__main__":
    print("\n=== AUTO TRAINER TEST ===\n")
    
    trainer = AutoTrainer()
    
    # Start training
    trainer.start()
    
    # Let it run for a while
    try:
        while True:
            time.sleep(60)
            status = trainer.get_status()
            print(f"\nStatus: {json.dumps(status, indent=2)}")
    except KeyboardInterrupt:
        trainer.stop()
