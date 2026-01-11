"""
DexScreener API Module
======================
Modul untuk mengambil data dari DexScreener API (Gratis & Public)
- Mencari pair baru/trending
- Mengambil data historis untuk training
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from colorama import Fore, Style
import time


class DexScreenerAPI:
    """Client untuk DexScreener API"""
    
    BASE_URL = "https://api.dexscreener.com"
    
    # Chain IDs yang didukung
    SUPPORTED_CHAINS = [
        "ethereum", "bsc", "polygon", "arbitrum", "optimism",
        "avalanche", "fantom", "cronos", "base", "solana"
    ]
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        })
        self.last_request_time = 0
        self.min_request_interval = 1.5  # Minimum 1.5 seconds between requests
        self.cache = {}
        self.cache_ttl = 30  # Cache for 30 seconds
    
    def _rate_limit(self):
        """Ensure minimum interval between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """Helper untuk membuat HTTP request dengan retry dan exponential backoff"""
        
        # Check cache first
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cache_data
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()  # Rate limiting
                
                url = f"{self.BASE_URL}{endpoint}"
                response = self.session.get(url, params=params, timeout=15)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    print(f"{Fore.YELLOW}[RATE LIMIT] Waiting {wait_time}s before retry...{Style.RESET_ALL}")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                # Cache successful response
                self.cache[cache_key] = (time.time(), data)
                return data
                
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 1
                    print(f"{Fore.YELLOW}[RETRY] Attempt {attempt + 1}/{max_retries}, waiting {wait_time}s...{Style.RESET_ALL}")
                    time.sleep(wait_time)
                else:
                    print(f"{Fore.RED}[ERROR] API Request gagal setelah {max_retries} percobaan: {e}{Style.RESET_ALL}")
                    return None
        return None
    
    def search_new_pairs(self, chain: str = "all", min_liquidity: float = 1000) -> pd.DataFrame:
        """
        Mencari pair baru/trending dari DexScreener
        FAST MODE: Prioritas Search Endpoint (Langsung dapat data).
        """
        print(f"{Fore.CYAN}[INFO] Fast Scanning DexScreener...{Style.RESET_ALL}")
        
        pairs_list = []
        
        # 1. FASTEST: Search Endpoint (Returns full pair objects)
        # We query for popular terms to get a mix of trending/new
        queries = ["solana", "meme", "pump"]
        
        import concurrent.futures
        
        def fetch_search(q):
            try:
                data = self._make_request("/latest/dex/search", params={"q": q})
                if data and "pairs" in data:
                    return [self._parse_pair(p) for p in data["pairs"][:20]]
            except:
                pass
            return []

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(fetch_search, queries))
            for r in results:
                pairs_list.extend(r)
        
        # Remove duplicates
        seen = set()
        unique_pairs = []
        for p in pairs_list:
            if p['pair_address'] not in seen:
                seen.add(p['pair_address'])
                unique_pairs.append(p)
        
        # 2. If result is too small, try Boosts (Slower)
        if len(unique_pairs) < 10:
            print(f"{Fore.YELLOW}[INFO] Fast search yielded few results. Trying Boosts...{Style.RESET_ALL}")
            endpoint = "/token-boosts/latest/v1"
            data = self._make_request(endpoint)
            
            raw_candidates = []
            if data and isinstance(data, list):
                for item in data[:30]:
                     raw_candidates.append({"address": item.get("tokenAddress"), "chain": item.get("chainId")})
            
            # Fetch details for boosts
            boosted_pairs = self._fetch_details_concurrently(raw_candidates)
            unique_pairs.extend(boosted_pairs)
            
        df = pd.DataFrame(unique_pairs)
        return self._filter_pairs(df, min_liquidity)

    def _filter_pairs(self, df_or_list, min_liquidity):
        if isinstance(df_or_list, list):
             df = pd.DataFrame(df_or_list)
        else:
             df = df_or_list
             
        if not df.empty and "liquidity_usd" in df.columns:
            df = df[df["liquidity_usd"] >= min_liquidity]
            df = df.sort_values("volume_24h", ascending=False)
            print(f"{Fore.GREEN}[SUCCESS] Ditemukan {len(df)} pair potensial{Style.RESET_ALL}")
            return df
        
        print(f"{Fore.YELLOW}[WARNING] Tidak ada pair yang ditemukan{Style.RESET_ALL}")
        return pd.DataFrame()

    def _fetch_details_concurrently(self, candidates: List[Dict]) -> List[Dict]:
        """Fetch details in parallel threads"""
        import concurrent.futures
        
        results = []
        # Batasi max worker agar tidak kena ban IP (misal 10 worker)
        # Dan kita bypass rate limit sleep internal dengan hati-hati
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_token = {
                executor.submit(self._fetch_single_pair_no_wait, c["address"], c["chain"]): c 
                for c in candidates
            }
            
            for future in concurrent.futures.as_completed(future_to_token):
                try:
                    data = future.result()
                    if data:
                        results.extend(data)
                except Exception as e:
                    pass
        
        return results

    def _fetch_single_pair_no_wait(self, token_address, chain_id):
        """Fetch detail tanpa global sleep lock yang agresif"""
        endpoint = f"/tokens/v1/{chain_id}/{token_address}"
        try:
            # Direct request bypassing the heavy _make_request locking for speed
            # But we still need basic headers
            url = f"{self.BASE_URL}{endpoint}"
            resp = self.session.get(url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list):
                     return [self._parse_pair(p) for p in data[:1]] # Ambil top pair saja
            elif resp.status_code == 429:
                print("Rate limit hit in thread")
                time.sleep(1)
        except:
            pass
        return []

    def _get_trending_candidates_from_profiles(self) -> List[Dict]:
        endpoint = "/token-profiles/latest/v1"
        data = self._make_request(endpoint)
        candidates = []
        if data and isinstance(data, list):
            for item in data[:30]:
                candidates.append({
                    "address": item.get("tokenAddress"),
                    "chain": item.get("chainId", "solana")
                })
        return candidates
    
    def _parse_pair(self, pair: Dict) -> Dict:
        """Parse data pair menjadi format standar"""
        price_change = pair.get("priceChange", {})
        
        return {
            "pair_address": pair.get("pairAddress", ""),
            "chain_id": pair.get("chainId", ""),
            "dex_id": pair.get("dexId", ""),
            "base_token": pair.get("baseToken", {}).get("symbol", ""),
            "base_token_address": pair.get("baseToken", {}).get("address", ""),
            "quote_token": pair.get("quoteToken", {}).get("symbol", ""),
            "price_usd": float(pair.get("priceUsd", 0) or 0),
            "price_native": float(pair.get("priceNative", 0) or 0),
            "volume_5m": float(pair.get("volume", {}).get("m5", 0) or 0),
            "volume_1h": float(pair.get("volume", {}).get("h1", 0) or 0),
            "volume_6h": float(pair.get("volume", {}).get("h6", 0) or 0),
            "volume_24h": float(pair.get("volume", {}).get("h24", 0) or 0),
            "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0) or 0),
            "liquidity_base": float(pair.get("liquidity", {}).get("base", 0) or 0),
            "liquidity_quote": float(pair.get("liquidity", {}).get("quote", 0) or 0),
            "fdv": float(pair.get("fdv", 0) or 0),
            "market_cap": float(pair.get("marketCap", 0) or 0),
            "price_change_5m": float(price_change.get("m5", 0) or 0),
            "price_change_1h": float(price_change.get("h1", 0) or 0),
            "price_change_6h": float(price_change.get("h6", 0) or 0),
            "price_change_24h": float(price_change.get("h24", 0) or 0),
            "txns_buys_5m": int(pair.get("txns", {}).get("m5", {}).get("buys", 0) or 0),
            "txns_sells_5m": int(pair.get("txns", {}).get("m5", {}).get("sells", 0) or 0),
            "txns_buys_1h": int(pair.get("txns", {}).get("h1", {}).get("buys", 0) or 0),
            "txns_sells_1h": int(pair.get("txns", {}).get("h1", {}).get("sells", 0) or 0),
            "txns_buys_24h": int(pair.get("txns", {}).get("h24", {}).get("buys", 0) or 0),
            "txns_sells_24h": int(pair.get("txns", {}).get("h24", {}).get("sells", 0) or 0),
            "pair_created_at": pair.get("pairCreatedAt", None),
            "url": pair.get("url", ""),
        }
    
    def get_historical_data(self, pair_address: str, chain_id: str = "solana") -> pd.DataFrame:
        """
        Mengambil data historis untuk training
        
        Note: DexScreener tidak menyediakan endpoint candle gratis,
        jadi kita menggunakan data transaksi dan volume sebagai proxy
        
        Args:
            pair_address: Alamat pair
            chain_id: ID chain
            
        Returns:
            DataFrame dengan data historis (simulasi dari current data)
        """
        print(f"{Fore.CYAN}[INFO] Mengambil data historis untuk {pair_address[:10]}...{Style.RESET_ALL}")
        
        # Get current pair data
        endpoint = f"/latest/dex/pairs/{chain_id}/{pair_address}"
        data = self._make_request(endpoint)
        
        if not data or "pairs" not in data or len(data["pairs"]) == 0:
            print(f"{Fore.YELLOW}[WARNING] Data pair tidak ditemukan{Style.RESET_ALL}")
            return pd.DataFrame()
        
        pair = data["pairs"][0]
        
        # Generate synthetic historical data berdasarkan current metrics
        # Ini adalah simulasi karena DexScreener tidak menyediakan historical candles gratis
        current_price = float(pair.get("priceUsd", 0) or 0)
        volume_24h = float(pair.get("volume", {}).get("h24", 0) or 0)
        price_change_24h = float(pair.get("priceChange", {}).get("h24", 0) or 0)
        
        historical_data = self._generate_synthetic_history(
            current_price=current_price,
            volume_24h=volume_24h,
            price_change_24h=price_change_24h,
            hours=48  # 48 jam data
        )
        
        print(f"{Fore.GREEN}[SUCCESS] Generated {len(historical_data)} data points{Style.RESET_ALL}")
        return historical_data
    
    def _generate_synthetic_history(self, current_price: float, volume_24h: float, 
                                     price_change_24h: float, hours: int = 48) -> pd.DataFrame:
        """
        Generate synthetic historical data V2 - More realistic patterns
        Includes: trend, cycles, volume correlation, and micro-patterns
        """
        import numpy as np
        
        if current_price == 0:
            return pd.DataFrame()
        
        # Calculate starting price
        start_price = current_price / (1 + price_change_24h / 100) if price_change_24h != -100 else current_price
        
        # Generate timestamps
        now = datetime.now()
        timestamps = [now - timedelta(hours=hours-i) for i in range(hours)]
        
        np.random.seed(int(now.timestamp()) % 10000)  # Dynamic seed
        
        # === V2: Multi-component price generation ===
        
        # 1. Base trend (linear interpolation with acceleration)
        trend_power = 1.5 if price_change_24h > 0 else 0.8  # Accelerate if bullish
        t = np.linspace(0, 1, hours)
        base_trend = start_price + (current_price - start_price) * (t ** trend_power)
        
        # 2. Cyclical patterns (simulate intraday patterns)
        cycle_period = 8  # 8-hour cycle
        cycle_amplitude = current_price * 0.015  # 1.5% amplitude
        cycles = cycle_amplitude * np.sin(2 * np.pi * np.arange(hours) / cycle_period)
        
        # 3. Micro-volatility with clustering (GARCH-like)
        base_volatility = abs(price_change_24h) / 100 * 0.3 + 0.01
        volatility = np.zeros(hours)
        volatility[0] = base_volatility
        for i in range(1, hours):
            # Volatility clustering
            volatility[i] = 0.7 * volatility[i-1] + 0.3 * base_volatility * (1 + np.random.exponential(0.5))
        
        noise = np.random.normal(0, 1, hours) * volatility * current_price
        
        # 4. Sudden spikes (pump/dump simulation)
        spikes = np.zeros(hours)
        spike_prob = 0.1 if abs(price_change_24h) > 20 else 0.05
        for i in range(hours):
            if np.random.random() < spike_prob:
                spike_direction = 1 if price_change_24h > 0 else -1
                spikes[i] = spike_direction * current_price * np.random.uniform(0.01, 0.03)
        
        # Combine all components
        prices = base_trend + cycles + noise + spikes
        prices = np.maximum(prices, current_price * 0.3)  # Floor at 30%
        prices = np.minimum(prices, current_price * 2.0)  # Cap at 200%
        
        # Force last price to be close to current
        prices[-1] = current_price
        prices[-2] = current_price * np.random.uniform(0.98, 1.02)
        
        # === V2: Realistic OHLC generation ===
        opens = np.zeros(hours)
        highs = np.zeros(hours)
        lows = np.zeros(hours)
        closes = prices.copy()
        
        opens[0] = start_price
        for i in range(1, hours):
            opens[i] = closes[i-1]  # Open = previous close
        
        # High/Low with volume-correlated range
        for i in range(hours):
            candle_range = abs(closes[i] - opens[i]) * np.random.uniform(1.0, 1.5)
            highs[i] = max(opens[i], closes[i]) + candle_range * np.random.uniform(0.2, 0.5)
            lows[i] = min(opens[i], closes[i]) - candle_range * np.random.uniform(0.2, 0.5)
            lows[i] = max(lows[i], current_price * 0.2)  # Floor
        
        # === V2: Volume with price correlation ===
        avg_volume_per_hour = volume_24h / 24
        
        # Volume correlates with price change
        price_changes = np.abs(np.diff(prices, prepend=prices[0])) / prices * 100
        volume_multiplier = 1 + price_changes * 0.1  # More volume on bigger moves
        
        base_volumes = np.random.exponential(avg_volume_per_hour, hours)
        volumes = base_volumes * volume_multiplier
        
        # Recent hours have more volume (approaching now)
        recency_factor = np.linspace(0.5, 1.5, hours)
        volumes = volumes * recency_factor
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes
        })
        
        return df
    
    def get_token_info(self, token_address: str, chain_id: str = "solana") -> Optional[Dict]:
        """Mendapatkan informasi detail token"""
        endpoint = f"/tokens/v1/{chain_id}/{token_address}"
        data = self._make_request(endpoint)
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        return None


# === Utility Functions ===

def search_new_pairs(chain: str = "all", min_liquidity: float = 1000) -> pd.DataFrame:
    """Wrapper function untuk mencari pair baru"""
    api = DexScreenerAPI()
    return api.search_new_pairs(chain=chain, min_liquidity=min_liquidity)


def get_historical_data(pair_address: str, chain_id: str = "solana") -> pd.DataFrame:
    """Wrapper function untuk mendapatkan data historis"""
    api = DexScreenerAPI()
    return api.get_historical_data(pair_address=pair_address, chain_id=chain_id)


if __name__ == "__main__":
    # Test module
    print(f"\n{Fore.YELLOW}{'='*60}")
    print("  DEXSCREENER API MODULE TEST")
    print(f"{'='*60}{Style.RESET_ALL}\n")
    
    # Test search
    pairs = search_new_pairs(min_liquidity=5000)
    
    if not pairs.empty:
        print(f"\n{Fore.GREEN}Top 5 Pairs:{Style.RESET_ALL}")
        print(pairs[["base_token", "price_usd", "volume_24h", "liquidity_usd", "price_change_1h"]].head())
        
        # Test historical data
        if len(pairs) > 0:
            test_pair = pairs.iloc[0]
            hist = get_historical_data(
                pair_address=test_pair["pair_address"],
                chain_id=test_pair["chain_id"]
            )
            print(f"\n{Fore.GREEN}Historical Data Sample:{Style.RESET_ALL}")
            print(hist.head())
