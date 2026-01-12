"""
Indodax API Module
==================
Fetches real-time crypto data from Indodax (Indonesian Exchange)
Public API: https://indodax.com/api/

Endpoints Used:
- /api/pairs - Get all trading pairs
- /api/ticker_all - Get all tickers with price data
- /api/summaries - Get 24h summaries
- /api/depth/{pair} - Get order book
- /api/trades/{pair} - Get recent trades
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


class IndodaxSignal(Enum):
    STRONG_BUY = "🚀 STRONG BUY"
    BUY = "📈 BUY"
    NEUTRAL = "➡️ HOLD"
    SELL = "📉 SELL"
    STRONG_SELL = "💥 STRONG SELL"


@dataclass
class IndodaxToken:
    """Token data from Indodax"""
    pair_id: str
    symbol: str
    name: str
    price_idr: float
    price_usd: float
    change_24h: float
    volume_24h_idr: float
    volume_24h_coin: float
    high_24h: float
    low_24h: float
    buy_price: float
    sell_price: float
    spread_pct: float
    last_update: str
    
    # Analysis fields
    signal: Optional[str] = None
    signal_type: Optional[str] = None
    confidence: float = 50.0
    buy_pressure: float = 50.0
    volatility: float = 0.0


class IndodaxAPI:
    """
    Indodax Public API Client
    Uses ticker_all endpoint for efficiency (1 request for all data)
    """
    
    BASE_URL = "https://indodax.com/api"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
            'Referer': 'https://indodax.com/'
        })
        self.usd_idr_rate = 16000  # Default rate
        self.cache = {}
        self.cache_time = 0
        self.cache_duration = 60  # 60 seconds cache to avoid rate limit
        
    def _get(self, endpoint: str, retries: int = 3) -> Optional[Dict]:
        """Make GET request to Indodax API with retry logic"""
        for attempt in range(retries):
            try:
                url = f"{self.BASE_URL}/{endpoint}"
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 429:
                    wait_time = (attempt + 1) * 10
                    print(f"[INDODAX] Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                print(f"[INDODAX] API Error (attempt {attempt+1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(5)
                continue
                
        return None
    
    def get_usd_rate(self) -> float:
        """Get current USD/IDR rate"""
        try:
            # Use USDT/IDR pair as reference
            ticker = self._get("ticker/usdtidr")
            if ticker and 'ticker' in ticker:
                rate = float(ticker['ticker'].get('last', 16000))
                self.usd_idr_rate = rate
                return rate
        except:
            pass
        return self.usd_idr_rate
    
    def get_all_pairs(self) -> List[Dict]:
        """Get all available trading pairs"""
        data = self._get("pairs")
        if data:
            return data
        return []
    
    def get_all_tickers(self) -> Dict:
        """Get all tickers with current prices"""
        data = self._get("ticker_all")
        if data and 'tickers' in data:
            return data['tickers']
        return {}
    
    def get_summaries(self) -> Dict:
        """Get 24h summaries for all pairs"""
        data = self._get("summaries")
        if data and 'tickers' in data:
            return data['tickers']
        return {}
    
    def get_depth(self, pair: str, limit: int = 20) -> Dict:
        """Get order book for a pair"""
        data = self._get(f"depth/{pair}")
        if data:
            return {
                'buy': data.get('buy', [])[:limit],
                'sell': data.get('sell', [])[:limit]
            }
        return {'buy': [], 'sell': []}
    
    def get_trades(self, pair: str) -> List[Dict]:
        """Get recent trades for a pair"""
        data = self._get(f"trades/{pair}")
        if data:
            return data[:50]  # Last 50 trades
        return []
    
    def analyze_order_book(self, pair: str) -> Dict:
        """Analyze order book for buy/sell pressure"""
        depth = self.get_depth(pair)
        
        total_buy_volume = sum(float(order[1]) for order in depth['buy'][:10])
        total_sell_volume = sum(float(order[1]) for order in depth['sell'][:10])
        total_volume = total_buy_volume + total_sell_volume
        
        buy_pressure = (total_buy_volume / total_volume * 100) if total_volume > 0 else 50
        
        # Calculate spread
        if depth['buy'] and depth['sell']:
            best_bid = float(depth['buy'][0][0])
            best_ask = float(depth['sell'][0][0])
            spread_pct = ((best_ask - best_bid) / best_bid * 100) if best_bid > 0 else 0
        else:
            spread_pct = 0
            
        return {
            'buy_pressure': buy_pressure,
            'sell_pressure': 100 - buy_pressure,
            'spread_pct': spread_pct,
            'buy_wall': total_buy_volume,
            'sell_wall': total_sell_volume
        }
    
    def analyze_trades(self, pair: str) -> Dict:
        """Analyze recent trades for momentum"""
        trades = self.get_trades(pair)
        
        if not trades:
            return {'momentum': 0, 'buy_ratio': 50, 'avg_size': 0}
        
        buy_trades = [t for t in trades if t.get('type') == 'buy']
        sell_trades = [t for t in trades if t.get('type') == 'sell']
        
        buy_ratio = (len(buy_trades) / len(trades) * 100) if trades else 50
        
        # Calculate momentum (price direction of recent trades)
        if len(trades) >= 2:
            first_price = float(trades[-1].get('price', 0))
            last_price = float(trades[0].get('price', 0))
            momentum = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        else:
            momentum = 0
            
        avg_size = sum(float(t.get('amount', 0)) for t in trades) / len(trades) if trades else 0
        
        return {
            'momentum': momentum,
            'buy_ratio': buy_ratio,
            'avg_size': avg_size,
            'total_trades': len(trades)
        }
    
    def calculate_signal(self, token_data: Dict, order_analysis: Dict, trade_analysis: Dict) -> Tuple[IndodaxSignal, float]:
        """Calculate buy/sell signal based on multiple factors"""
        score = 50  # Base score
        
        # 1. Price Change (25%)
        change_24h = token_data.get('change_24h', 0)
        if change_24h > 10:
            score += 25
        elif change_24h > 5:
            score += 15
        elif change_24h > 0:
            score += 5
        elif change_24h < -10:
            score -= 25
        elif change_24h < -5:
            score -= 15
        elif change_24h < 0:
            score -= 5
        
        # 2. Buy Pressure from Order Book (25%)
        buy_pressure = order_analysis.get('buy_pressure', 50)
        if buy_pressure > 70:
            score += 25
        elif buy_pressure > 60:
            score += 15
        elif buy_pressure > 55:
            score += 5
        elif buy_pressure < 30:
            score -= 25
        elif buy_pressure < 40:
            score -= 15
        elif buy_pressure < 45:
            score -= 5
        
        # 3. Trade Momentum (25%)
        momentum = trade_analysis.get('momentum', 0)
        if momentum > 2:
            score += 25
        elif momentum > 1:
            score += 15
        elif momentum > 0:
            score += 5
        elif momentum < -2:
            score -= 25
        elif momentum < -1:
            score -= 15
        elif momentum < 0:
            score -= 5
        
        # 4. Trade Buy Ratio (25%)
        buy_ratio = trade_analysis.get('buy_ratio', 50)
        if buy_ratio > 70:
            score += 25
        elif buy_ratio > 60:
            score += 15
        elif buy_ratio > 55:
            score += 5
        elif buy_ratio < 30:
            score -= 25
        elif buy_ratio < 40:
            score -= 15
        elif buy_ratio < 45:
            score -= 5
        
        # Determine signal
        if score >= 85:
            signal = IndodaxSignal.STRONG_BUY
        elif score >= 65:
            signal = IndodaxSignal.BUY
        elif score <= 15:
            signal = IndodaxSignal.STRONG_SELL
        elif score <= 35:
            signal = IndodaxSignal.SELL
        else:
            signal = IndodaxSignal.NEUTRAL
        
        confidence = min(95, max(30, score))
        
        return signal, confidence
    
    def get_market_data(self, min_volume_idr: float = 0, top_n: int = 200) -> List[IndodaxToken]:
        """
        Get market data with analysis
        Uses ticker_all endpoint for single API call
        
        Args:
            min_volume_idr: Minimum 24h volume in IDR (default 0 = show all)
            top_n: Number of top tokens to return (default 200 = all)
        """
        # Check cache first
        if time.time() - self.cache_time < self.cache_duration and self.cache.get('market_data'):
            print("[INDODAX] Using cached data")
            return self.cache['market_data']
        
        print("[INDODAX] Fetching market data from ticker_all...")
        
        # Use ticker_all - single request for all data
        tickers = self.get_all_tickers()
        if not tickers:
            print("[INDODAX] No ticker data received")
            # Return cached data if available
            if self.cache.get('market_data'):
                return self.cache['market_data']
            return []
        
        tokens = []
        
        for pair_id, data in tickers.items():
            try:
                # Skip non-IDR pairs and stablecoins
                if not pair_id.endswith('_idr'):
                    continue
                if pair_id in ['usdt_idr', 'usdc_idr', 'busd_idr', 'dai_idr']:
                    continue
                
                # Extract symbol
                symbol = pair_id.replace('_idr', '').upper()
                
                # Parse data
                price_idr = float(data.get('last', 0))
                high_24h = float(data.get('high', 0))
                low_24h = float(data.get('low', 0))
                volume_idr = float(data.get('vol_idr', data.get('volIdr', 0)))
                volume_coin = float(data.get('vol_' + symbol.lower(), 0))
                buy_price = float(data.get('buy', 0))
                sell_price = float(data.get('sell', 0))
                
                # Filter by volume
                if volume_idr < min_volume_idr:
                    continue
                
                # Calculate metrics
                price_usd = price_idr / self.usd_idr_rate if self.usd_idr_rate > 0 else 0
                
                # Calculate 24h change
                if low_24h > 0:
                    # Estimate change from high/low
                    mid_price = (high_24h + low_24h) / 2
                    change_24h = ((price_idr - mid_price) / mid_price * 100) if mid_price > 0 else 0
                else:
                    change_24h = 0
                
                # Calculate spread
                spread_pct = ((sell_price - buy_price) / buy_price * 100) if buy_price > 0 else 0
                
                # Calculate volatility
                volatility = ((high_24h - low_24h) / low_24h * 100) if low_24h > 0 else 0
                
                token = IndodaxToken(
                    pair_id=pair_id,
                    symbol=symbol,
                    name=data.get('name', symbol),
                    price_idr=price_idr,
                    price_usd=price_usd,
                    change_24h=change_24h,
                    volume_24h_idr=volume_idr,
                    volume_24h_coin=volume_coin,
                    high_24h=high_24h,
                    low_24h=low_24h,
                    buy_price=buy_price,
                    sell_price=sell_price,
                    spread_pct=spread_pct,
                    volatility=volatility,
                    last_update=datetime.now().isoformat()
                )
                
                tokens.append(token)
                
            except Exception as e:
                print(f"[INDODAX] Error parsing {pair_id}: {e}")
                continue
        
        # Sort by volume
        tokens.sort(key=lambda x: x.volume_24h_idr, reverse=True)
        tokens = tokens[:top_n]
        
        # Quick analysis without additional API calls (to avoid rate limit)
        print(f"[INDODAX] Analyzing {len(tokens)} tokens (fast mode)...")
        
        for token in tokens:
            try:
                # Simple signal based on available data only
                score = 50
                
                # Price change factor
                if token.change_24h > 10:
                    score += 20
                elif token.change_24h > 5:
                    score += 12
                elif token.change_24h > 2:
                    score += 5
                elif token.change_24h < -10:
                    score -= 20
                elif token.change_24h < -5:
                    score -= 12
                elif token.change_24h < -2:
                    score -= 5
                
                # Volatility factor (high volatility = opportunity but risky)
                if 5 < token.volatility < 15:
                    score += 10  # Good volatility
                elif token.volatility > 20:
                    score -= 5  # Too volatile
                
                # Spread factor (lower spread = better liquidity)
                if token.spread_pct < 0.5:
                    score += 10
                elif token.spread_pct < 1:
                    score += 5
                elif token.spread_pct > 3:
                    score -= 10
                
                # Volume factor (high volume = more interest)
                if token.volume_24h_idr > 10_000_000_000:  # > 10B IDR
                    score += 10
                elif token.volume_24h_idr > 1_000_000_000:  # > 1B IDR
                    score += 5
                
                # Determine signal
                if score >= 80:
                    signal = IndodaxSignal.STRONG_BUY
                elif score >= 65:
                    signal = IndodaxSignal.BUY
                elif score <= 20:
                    signal = IndodaxSignal.STRONG_SELL
                elif score <= 35:
                    signal = IndodaxSignal.SELL
                else:
                    signal = IndodaxSignal.NEUTRAL
                
                token.signal = signal.value
                token.signal_type = signal.name
                token.confidence = min(95, max(30, score))
                token.buy_pressure = 50 + (token.change_24h / 2)  # Estimate
                
            except Exception as e:
                print(f"[INDODAX] Error analyzing {token.symbol}: {e}")
                token.signal = IndodaxSignal.NEUTRAL.value
                token.signal_type = IndodaxSignal.NEUTRAL.name
                token.confidence = 50
        
        # Sort by signal strength and confidence
        signal_order = {'STRONG_BUY': 0, 'BUY': 1, 'NEUTRAL': 2, 'SELL': 3, 'STRONG_SELL': 4}
        tokens.sort(key=lambda x: (signal_order.get(x.signal_type, 2), -x.confidence))
        
        # Cache results
        self.cache['market_data'] = tokens
        self.cache_time = time.time()
        
        print(f"[INDODAX] Found {len(tokens)} tokens")
        
        return tokens
    
    def to_dict(self, token: IndodaxToken) -> Dict:
        """Convert token to dictionary for API response"""
        return {
            'pair_id': token.pair_id,
            'symbol': token.symbol,
            'name': token.name,
            'price_idr': token.price_idr,
            'price_usd': token.price_usd,
            'change_24h': round(token.change_24h, 2),
            'volume_24h_idr': token.volume_24h_idr,
            'volume_24h_usd': token.volume_24h_idr / self.usd_idr_rate if self.usd_idr_rate > 0 else 0,
            'high_24h': token.high_24h,
            'low_24h': token.low_24h,
            'buy_price': token.buy_price,
            'sell_price': token.sell_price,
            'spread_pct': round(token.spread_pct, 3),
            'volatility': round(token.volatility, 2),
            'signal': token.signal,
            'signal_type': token.signal_type,
            'confidence': round(token.confidence, 1),
            'buy_pressure': round(token.buy_pressure, 1),
            'last_update': token.last_update
        }


# Global instance
indodax_api = IndodaxAPI()


def get_indodax_market() -> List[Dict]:
    """Get Indodax market data as list of dicts"""
    tokens = indodax_api.get_market_data()
    return [indodax_api.to_dict(t) for t in tokens]


# Quick test
if __name__ == "__main__":
    print("\n=== INDODAX API TEST ===")
    
    api = IndodaxAPI()
    
    # Get market data
    tokens = api.get_market_data(min_volume_idr=50_000_000, top_n=10)
    
    print(f"\nTop {len(tokens)} tokens by volume:")
    for t in tokens:
        print(f"  {t.symbol}: Rp {t.price_idr:,.0f} | {t.change_24h:+.2f}% | {t.signal} ({t.confidence:.0f}%)")
