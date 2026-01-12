"""
Crypto Data Aggregator V1.0
===========================
Multi-Source Big Data Aggregator untuk AI Training

Sources:
1. Binance API - Historical OHLCV (primary, most data)
2. CoinGecko API - Market data & historical
3. CryptoCompare API - Historical data
4. Indodax API - Indonesian market data

Menggabungkan semua data menjadi satu dataset untuk training AI
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AggregatedData:
    """Combined data from multiple sources"""
    symbol: str
    ohlcv: pd.DataFrame
    sources_used: List[str]
    total_candles: int
    date_range: Tuple[datetime, datetime]
    quality_score: float


class BinanceDataFetcher:
    """
    Fetch historical data from Binance (largest crypto exchange)
    Free API, no key needed for public data
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
    
    def get_klines(self, symbol: str, interval: str = '1h', limit: int = 1000) -> pd.DataFrame:
        """
        Get OHLCV klines from Binance
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
            interval: 1m, 5m, 15m, 1h, 4h, 1d
            limit: Max 1000 per request
        """
        try:
            # Normalize symbol for Binance (uppercase, USDT pair)
            binance_symbol = f"{symbol.upper()}USDT"
            
            url = f"{self.BASE_URL}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp')
                    
                    print(f"[BINANCE] Got {len(df)} candles for {binance_symbol}")
                    return df
                    
        except Exception as e:
            print(f"[BINANCE] Error fetching {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_historical_klines(self, symbol: str, interval: str = '1h', days: int = 30) -> pd.DataFrame:
        """Get historical klines for multiple days"""
        all_data = []
        binance_symbol = f"{symbol.upper()}USDT"
        
        try:
            # Calculate start time
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            url = f"{self.BASE_URL}/klines"
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    df = pd.DataFrame(data, columns=[
                        'timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                        'taker_buy_quote', 'ignore'
                    ])
                    
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['open'] = df['open'].astype(float)
                    df['high'] = df['high'].astype(float)
                    df['low'] = df['low'].astype(float)
                    df['close'] = df['close'].astype(float)
                    df['volume'] = df['volume'].astype(float)
                    
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp')
                    
                    print(f"[BINANCE] Got {len(df)} historical candles for {binance_symbol} ({days} days)")
                    return df
                    
        except Exception as e:
            print(f"[BINANCE] Error fetching historical {symbol}: {e}")
        
        return pd.DataFrame()


class CoinGeckoDataFetcher:
    """
    Fetch data from CoinGecko (free, comprehensive)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Map common symbols to CoinGecko IDs
    SYMBOL_TO_ID = {
        'BTC': 'bitcoin',
        'ETH': 'ethereum',
        'BNB': 'binancecoin',
        'SOL': 'solana',
        'XRP': 'ripple',
        'ADA': 'cardano',
        'DOGE': 'dogecoin',
        'SHIB': 'shiba-inu',
        'PEPE': 'pepe',
        'DOT': 'polkadot',
        'MATIC': 'matic-network',
        'AVAX': 'avalanche-2',
        'LINK': 'chainlink',
        'UNI': 'uniswap',
        'ATOM': 'cosmos',
        'LTC': 'litecoin',
        'TRX': 'tron',
        'NEAR': 'near',
        'APT': 'aptos',
        'ARB': 'arbitrum',
        'OP': 'optimism',
        'INJ': 'injective-protocol',
        'FTM': 'fantom',
        'SAND': 'the-sandbox',
        'MANA': 'decentraland',
        'GALA': 'gala',
        'AXS': 'axie-infinity',
        'FLOKI': 'floki',
        'BONK': 'bonk',
        'WIF': 'dogwifcoin',
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        })
        self.last_request = 0
        self.rate_limit = 1.5  # seconds between requests
    
    def _rate_limit(self):
        """Respect rate limits"""
        elapsed = time.time() - self.last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request = time.time()
    
    def get_coin_id(self, symbol: str) -> Optional[str]:
        """Get CoinGecko ID from symbol"""
        symbol_upper = symbol.upper()
        if symbol_upper in self.SYMBOL_TO_ID:
            return self.SYMBOL_TO_ID[symbol_upper]
        return symbol.lower()
    
    def get_market_chart(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """
        Get OHLC data from CoinGecko
        
        Args:
            symbol: Coin symbol (e.g., 'BTC', 'ETH')
            days: Number of days (1, 7, 14, 30, 90, 180, 365, max)
        """
        self._rate_limit()
        
        try:
            coin_id = self.get_coin_id(symbol)
            
            url = f"{self.BASE_URL}/coins/{coin_id}/ohlc"
            params = {
                'vs_currency': 'usd',
                'days': days
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data:
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df['volume'] = 0  # CoinGecko OHLC doesn't include volume
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp')
                    
                    print(f"[COINGECKO] Got {len(df)} candles for {symbol}")
                    return df
            else:
                print(f"[COINGECKO] Status {response.status_code} for {symbol}")
                    
        except Exception as e:
            print(f"[COINGECKO] Error fetching {symbol}: {e}")
        
        return pd.DataFrame()


class CryptoCompareDataFetcher:
    """
    Fetch data from CryptoCompare (free tier available)
    """
    
    BASE_URL = "https://min-api.cryptocompare.com/data/v2"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0'
        })
    
    def get_histohour(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """
        Get hourly OHLCV data
        
        Args:
            symbol: Coin symbol (e.g., 'BTC', 'ETH')
            limit: Number of hours (max 2000)
        """
        try:
            url = f"{self.BASE_URL}/histohour"
            params = {
                'fsym': symbol.upper(),
                'tsym': 'USD',
                'limit': min(limit, 2000)
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success' and data.get('Data', {}).get('Data'):
                    records = data['Data']['Data']
                    
                    df = pd.DataFrame(records)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df = df.rename(columns={
                        'open': 'open',
                        'high': 'high', 
                        'low': 'low',
                        'close': 'close',
                        'volumefrom': 'volume'
                    })
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp')
                    
                    print(f"[CRYPTOCOMPARE] Got {len(df)} hourly candles for {symbol}")
                    return df
                    
        except Exception as e:
            print(f"[CRYPTOCOMPARE] Error fetching {symbol}: {e}")
        
        return pd.DataFrame()
    
    def get_histoday(self, symbol: str, limit: int = 365) -> pd.DataFrame:
        """Get daily OHLCV data"""
        try:
            url = f"{self.BASE_URL}/histoday"
            params = {
                'fsym': symbol.upper(),
                'tsym': 'USD',
                'limit': min(limit, 2000)
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('Response') == 'Success' and data.get('Data', {}).get('Data'):
                    records = data['Data']['Data']
                    
                    df = pd.DataFrame(records)
                    df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                    df = df.rename(columns={'volumefrom': 'volume'})
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    df = df.sort_values('timestamp')
                    
                    print(f"[CRYPTOCOMPARE] Got {len(df)} daily candles for {symbol}")
                    return df
                    
        except Exception as e:
            print(f"[CRYPTOCOMPARE] Error fetching daily {symbol}: {e}")
        
        return pd.DataFrame()


class CryptoDataAggregator:
    """
    Main aggregator class that combines data from all sources
    """
    
    def __init__(self):
        self.binance = BinanceDataFetcher()
        self.coingecko = CoinGeckoDataFetcher()
        self.cryptocompare = CryptoCompareDataFetcher()
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
    
    def aggregate_data(self, symbol: str, interval: str = '1h', min_candles: int = 500) -> AggregatedData:
        """
        Aggregate data from multiple sources
        
        Priority:
        1. Binance (most reliable, most data)
        2. CryptoCompare (good backup)
        3. CoinGecko (additional data)
        
        Args:
            symbol: Coin symbol (e.g., 'BTC', 'PEPE')
            interval: Time interval ('1h', '4h', '1d')
            min_candles: Minimum candles needed
        """
        cache_key = f"{symbol}_{interval}"
        
        # Check cache
        if cache_key in self.cache:
            cached_time, cached_data = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                print(f"[AGGREGATOR] Using cached data for {symbol}")
                return cached_data
        
        print(f"\n[AGGREGATOR] Fetching big data for {symbol}...")
        
        all_dfs = []
        sources_used = []
        
        # 1. Try Binance first (1000 candles)
        try:
            if interval == '1h':
                binance_df = self.binance.get_historical_klines(symbol, '1h', days=45)
            elif interval == '4h':
                binance_df = self.binance.get_historical_klines(symbol, '4h', days=180)
            else:
                binance_df = self.binance.get_historical_klines(symbol, '1d', days=365)
            
            if not binance_df.empty:
                binance_df['source'] = 'binance'
                all_dfs.append(binance_df)
                sources_used.append('Binance')
        except Exception as e:
            print(f"[AGGREGATOR] Binance failed: {e}")
        
        # 2. Try CryptoCompare
        try:
            if interval in ['1h', '4h']:
                cc_df = self.cryptocompare.get_histohour(symbol, limit=1000)
            else:
                cc_df = self.cryptocompare.get_histoday(symbol, limit=365)
            
            if not cc_df.empty:
                cc_df['source'] = 'cryptocompare'
                all_dfs.append(cc_df)
                sources_used.append('CryptoCompare')
        except Exception as e:
            print(f"[AGGREGATOR] CryptoCompare failed: {e}")
        
        # 3. Try CoinGecko (has rate limits, so use sparingly)
        if len(all_dfs) == 0 or sum(len(df) for df in all_dfs) < min_candles:
            try:
                days = 90 if interval in ['1h', '4h'] else 365
                cg_df = self.coingecko.get_market_chart(symbol, days=days)
                
                if not cg_df.empty:
                    cg_df['source'] = 'coingecko'
                    all_dfs.append(cg_df)
                    sources_used.append('CoinGecko')
            except Exception as e:
                print(f"[AGGREGATOR] CoinGecko failed: {e}")
        
        if not all_dfs:
            print(f"[AGGREGATOR] No data found for {symbol}")
            return AggregatedData(
                symbol=symbol,
                ohlcv=pd.DataFrame(),
                sources_used=[],
                total_candles=0,
                date_range=(datetime.now(), datetime.now()),
                quality_score=0.0
            )
        
        # Combine all data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Remove duplicates based on timestamp, prefer Binance > CryptoCompare > CoinGecko
        source_priority = {'binance': 0, 'cryptocompare': 1, 'coingecko': 2}
        combined_df['source_priority'] = combined_df['source'].map(source_priority)
        combined_df = combined_df.sort_values(['timestamp', 'source_priority'])
        combined_df = combined_df.drop_duplicates(subset='timestamp', keep='first')
        combined_df = combined_df.drop(columns=['source', 'source_priority'])
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate quality score
        quality_score = min(100, (len(combined_df) / min_candles) * 100)
        if len(sources_used) >= 2:
            quality_score = min(100, quality_score + 10)
        
        # Get date range
        date_range = (
            combined_df['timestamp'].min().to_pydatetime() if len(combined_df) > 0 else datetime.now(),
            combined_df['timestamp'].max().to_pydatetime() if len(combined_df) > 0 else datetime.now()
        )
        
        result = AggregatedData(
            symbol=symbol,
            ohlcv=combined_df,
            sources_used=sources_used,
            total_candles=len(combined_df),
            date_range=date_range,
            quality_score=quality_score
        )
        
        print(f"[AGGREGATOR] Combined {len(combined_df)} candles from {', '.join(sources_used)}")
        print(f"[AGGREGATOR] Date range: {date_range[0].strftime('%Y-%m-%d')} to {date_range[1].strftime('%Y-%m-%d')}")
        print(f"[AGGREGATOR] Quality score: {quality_score:.1f}%")
        
        # Cache result
        self.cache[cache_key] = (time.time(), result)
        
        return result
    
    def get_training_data(self, symbol: str) -> pd.DataFrame:
        """
        Get optimal training data for AI models
        Combines hourly data for short-term and daily for long-term patterns
        """
        # Get hourly data (for recent patterns)
        hourly_data = self.aggregate_data(symbol, interval='1h', min_candles=500)
        
        if hourly_data.total_candles < 100:
            print(f"[AGGREGATOR] Warning: Only {hourly_data.total_candles} candles available for {symbol}")
        
        return hourly_data.ohlcv


# Global instance
data_aggregator = CryptoDataAggregator()


def get_aggregated_data(symbol: str, interval: str = '1h') -> AggregatedData:
    """API function to get aggregated data"""
    return data_aggregator.aggregate_data(symbol, interval)


def get_training_data(symbol: str) -> pd.DataFrame:
    """API function to get training-ready data"""
    return data_aggregator.get_training_data(symbol)


# Test
if __name__ == "__main__":
    print("\n=== CRYPTO DATA AGGREGATOR TEST ===\n")
    
    aggregator = CryptoDataAggregator()
    
    # Test with BTC
    result = aggregator.aggregate_data("BTC", interval="1h")
    
    print(f"\nResult for BTC:")
    print(f"  Sources: {result.sources_used}")
    print(f"  Candles: {result.total_candles}")
    print(f"  Quality: {result.quality_score:.1f}%")
    print(f"  Date range: {result.date_range}")
