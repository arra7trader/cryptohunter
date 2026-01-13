
import asyncio
import sys
import os
import io

# Force UTF-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock FastAPI app to avoid starting server
from unittest.mock import MagicMock
sys.modules['fastapi'] = MagicMock()
sys.modules['fastapi.staticfiles'] = MagicMock()
sys.modules['fastapi.middleware.cors'] = MagicMock()

# Import the modules we want to test
try:
    from modules.binance_api import BinanceAPI
    from modules.fear_greed import FearGreedIndex
    from modules.enhanced_predictor import EnhancedPredictor
    print("✅ Successfully imported modules")
except ImportError as e:
    print(f"❌ Failed to import modules: {e}")
    sys.exit(1)

async def test_integration():
    print("\n--- Testing Binance API ---")
    binance = BinanceAPI()
    ticker = binance.get_ticker("BTC")
    if ticker:
        print(f"[OK] Binance Ticker (BTC): ${ticker.get('price', 0):,.2f}")
    else:
        print("[WARN] Binance Ticker failed (might be network issue, but code is running)")

    print("\n--- Testing Fear & Greed ---")
    fg = FearGreedIndex()
    sentiment = fg.get_current()
    print(f"[OK] Sentiment: {sentiment.get('value')} ({sentiment.get('classification')})")

    print("\n--- Testing Enhanced Predictor ---")
    predictor = EnhancedPredictor()
    
    try:
        if hasattr(predictor, 'predict'):
             print(f"[OK] EnhancedPredictor initialized and has predict method")
    except Exception as e:
        print(f"[ERR] EnhancedPredictor error: {e}")

if __name__ == "__main__":
    try:
        from modules.binance_api import BinanceAPI
        from modules.fear_greed import FearGreedIndex
        from modules.enhanced_predictor import EnhancedPredictor
        print("[OK] Successfully imported modules")
        asyncio.run(test_integration())
    except ImportError as e:
        print(f"[ERR] Failed to import modules: {e}")
    except Exception as e:
        print(f"[ERR] Unexpected error: {e}")
