"""
Binance API Integration V1.0
============================
Access global crypto market data from Binance (largest exchange by volume)

Features:
- Real-time price data
- 24h ticker statistics
- Historical klines (candlesticks)
- Order book depth
- WebSocket support (future)
- No API key required (public data)

Author: CryptoHunter Team
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import os


class BinanceAPI:
    """
    Binance API client for public market data
    No API key required
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        """Initialize Binance API client"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CryptoHunter/1.0'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        print("[BINANCE] API client initialized")
    
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _request(self, endpoint: str, params: Dict = None) -> Dict:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint
            params: Query parameters
        
        Returns:
            Response JSON
        """
        self._rate_limit()
        
        try:
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[BINANCE] Request error: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict:
        """
        Get 24h ticker statistics
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        
        Returns:
            Ticker data dictionary
        """
        binance_symbol = self._normalize_symbol(symbol)
        
        data = self._request("ticker/24hr", {"symbol": binance_symbol})
        
        if not data:
            return {}
        
        try:
            return {
                'symbol': symbol.upper(),
                'price': float(data.get('lastPrice', 0)),
                'price_change_24h': float(data.get('priceChange', 0)),
                'price_change_pct_24h': float(data.get('priceChangePercent', 0)),
                'volume_24h': float(data.get('volume', 0)),
                'quote_volume_24h': float(data.get('quoteVolume', 0)),
                'high_24h': float(data.get('highPrice', 0)),
                'low_24h': float(data.get('lowPrice', 0)),
                'open_price': float(data.get('openPrice', 0)),
                'bid_price': float(data.get('bidPrice', 0)),
                'ask_price': float(data.get('askPrice', 0)),
                'trades_count': int(data.get('count', 0)),
                'timestamp': datetime.now()
            }
        except (ValueError, KeyError) as e:
            print(f"[BINANCE] Error parsing ticker data: {e}")
            return {}
    
    def get_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 500
    ) -> pd.DataFrame:
        """
        Get historical candlestick data
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            interval: Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles (max 1000)
        
        Returns:
            DataFrame with OHLCV data
        """
        binance_symbol = self._normalize_symbol(symbol)
        
        data = self._request("klines", {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        })
        
        if not data:
            return pd.DataFrame()
        
        try:
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Keep only relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            return df
            
        except Exception as e:
            print(f"[BINANCE] Error parsing klines: {e}")
            return pd.DataFrame()
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get order book depth
        
        Args:
            symbol: Trading pair
            limit: Depth (5, 10, 20, 50, 100, 500, 1000)
        
        Returns:
            Order book with bids and asks
        """
        binance_symbol = self._normalize_symbol(symbol)
        
        data = self._request("depth", {
            "symbol": binance_symbol,
            "limit": limit
        })
        
        if not data:
            return {}
        
        try:
            bids = [[float(price), float(qty)] for price, qty in data.get('bids', [])]
            asks = [[float(price), float(qty)] for price, qty in data.get('asks', [])]
            
            return {
                'symbol': symbol.upper(),
                'bids': bids,  # [[price, quantity], ...]
                'asks': asks,
                'bid_depth': sum(qty for _, qty in bids),
                'ask_depth': sum(qty for _, qty in asks),
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0,
                'spread_pct': ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if bids and asks else 0,
                'timestamp': datetime.now()
            }
        except Exception as e:
            print(f"[BINANCE] Error parsing orderbook: {e}")
            return {}
    
    def get_price(self, symbol: str) -> float:
        """
        Get current price (simple)
        
        Args:
            symbol: Trading pair
        
        Returns:
            Current price
        """
        binance_symbol = self._normalize_symbol(symbol)
        
        data = self._request("ticker/price", {"symbol": binance_symbol})
        
        try:
            return float(data.get('price', 0))
        except:
            return 0.0
    
    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict:
        """
        Get exchange trading rules and symbol information
        
        Args:
            symbol: Optional specific symbol
        
        Returns:
            Exchange info
        """
        params = {}
        if symbol:
            params['symbol'] = self._normalize_symbol(symbol)
        
        return self._request("exchangeInfo", params)
    
    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize symbol to Binance format
        
        Args:
            symbol: Symbol like 'BTC', 'ETH', 'BTC-USDT'
        
        Returns:
            Binance format like 'BTCUSDT'
        """
        symbol = symbol.upper().replace('-', '').replace('_', '')
        
        # If already in correct format, return
        if 'USDT' in symbol or 'BUSD' in symbol:
            return symbol
        
        # Otherwise, append USDT
        return f"{symbol}USDT"
    
    def is_available(self) -> bool:
        """Check if Binance API is accessible"""
        try:
            data = self._request("ping")
            return data == {}  # Ping returns empty dict on success
        except:
            return False


# Convenience functions
def get_binance_price(symbol: str) -> float:
    """Get current price from Binance"""
    api = BinanceAPI()
    return api.get_price(symbol)


def get_binance_ticker(symbol: str) -> Dict:
    """Get 24h ticker from Binance"""
    api = BinanceAPI()
    return api.get_ticker(symbol)


def get_binance_klines(symbol: str, interval: str = '1h', limit: int = 500) -> pd.DataFrame:
    """Get historical candles from Binance"""
    api = BinanceAPI()
    return api.get_klines(symbol, interval, limit)


def get_binance_orderbook(symbol: str, limit: int = 20) -> Dict:
    """Get order book from Binance"""
    api = BinanceAPI()
    return api.get_orderbook(symbol, limit)


# Test
if __name__ == "__main__":
    print("\n=== BINANCE API TEST ===\n")
    
    api = BinanceAPI()
    
    # Test availability
    print(f"Binance API available: {api.is_available()}")
    
    # Test price
    print("\n1. Current Price:")
    price = api.get_price("BTC")
    print(f"BTC: ${price:,.2f}")
    
    # Test ticker
    print("\n2. 24h Ticker:")
    ticker = api.get_ticker("BTC")
    if ticker:
        print(f"Price: ${ticker['price']:,.2f}")
        print(f"24h Change: {ticker['price_change_pct_24h']:+.2f}%")
        print(f"24h Volume: {ticker['volume_24h']:,.2f} BTC")
        print(f"24h Quote Volume: ${ticker['quote_volume_24h']:,.0f}")
    
    # Test klines
    print("\n3. Historical Data (last 10 candles):")
    klines = api.get_klines("BTC", interval='1h', limit=10)
    if not klines.empty:
        print(klines.tail())
        print(f"\nTotal candles: {len(klines)}")
    
    # Test orderbook
    print("\n4. Order Book (top 5):")
    orderbook = api.get_orderbook("BTC", limit=5)
    if orderbook:
        print(f"Spread: ${orderbook['spread']:.2f} ({orderbook['spread_pct']:.3f}%)")
        print("\nBids:")
        for price, qty in orderbook['bids'][:5]:
            print(f"  ${price:,.2f} - {qty:.4f} BTC")
        print("\nAsks:")
        for price, qty in orderbook['asks'][:5]:
            print(f"  ${price:,.2f} - {qty:.4f} BTC")
    
    print("\n✅ Test complete!")
