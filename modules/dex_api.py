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
        self.min_request_interval = 0.2  # Reduced to 0.2s (~300 req/min) for speed
        self.cache = {}
        self.cache_ttl = 30  # Cache for 30 seconds
    
    def _rate_limit(self):
        """Ensure minimum interval between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, max_retries: int = 5) -> Optional[Dict]:
        """Helper untuk membuat HTTP request dengan retry dan exponential backoff yang robust"""
        
        # Check cache first
        cache_key = f"{endpoint}_{str(params)}"
        if cache_key in self.cache:
            cache_time, cache_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                return cache_data
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        backoff_delay = 1
        
        for attempt in range(max_retries):
            try:
                self._rate_limit()  # Rate limiting
                
                url = f"{self.BASE_URL}{endpoint}"
                response = self.session.get(url, params=params, timeout=10) # 10s timeout
                
                if response.status_code == 200:
                    data = response.json()
                    # Simpan ke cache
                    self.cache[cache_key] = (time.time(), data)
                    return data
                elif response.status_code == 429:
                    # Rate limit hit
                    print(f"{Fore.YELLOW}[WARN] Rate limit hit. Waiting {backoff_delay}s...{Style.RESET_ALL}")
                    time.sleep(backoff_delay)
                    backoff_delay *= 2
                    continue
                else:
                    print(f"{Fore.RED}[ERR] API Error {response.status_code}: {url}{Style.RESET_ALL}")
                    # For other non-200, non-429 errors, we don't retry based on status code, just log and exit
                    # If we want to retry on other HTTP errors, we'd need to add more logic here.
                    # For now, it will fall through and return None if not 200 or 429.
                    return None
                    
            except requests.exceptions.RequestException as e:
                # Network error (DNS, Timeout, Connection refused)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    print(f"{Fore.RED}[ERROR] API Request gagal setelah {max_retries} percobaan: {e}{Style.RESET_ALL}")
                    return None
        return None
    
    def search_new_pairs(self, chain: str = "all", min_liquidity: float = 1000) -> pd.DataFrame:
        """
        Mencari pair baru/trending dari DexScreener
        FAST MODE: Menggunakan Latest Boosts dan Token Profiles untuk mendapatkan koin BARU.
        """
        print(f"{Fore.CYAN}[INFO] Fast Scanning DexScreener for NEW tokens...{Style.RESET_ALL}")
        
        pairs_list = []
        
        import concurrent.futures
        
        # 1. UTAMA: Get LATEST Boosted Tokens (Biasanya koin baru yang dipromosikan)
        def fetch_boosts():
            try:
                endpoint = "/token-boosts/latest/v1"
                data = self._make_request(endpoint)
                if data and isinstance(data, list):
                    candidates = []
                    for item in data[:50]:  # Ambil 50 boosted tokens
                        candidates.append({
                            "address": item.get("tokenAddress"),
                            "chain": item.get("chainId")
                        })
                    return self._fetch_details_concurrently(candidates)
            except:
                pass
            return []
        
        # 2. Get Token Profiles (Koin dengan social/website = biasanya baru launch)
        def fetch_profiles():
            try:
                endpoint = "/token-profiles/latest/v1"
                data = self._make_request(endpoint)
                if data and isinstance(data, list):
                    candidates = []
                    for item in data[:50]:
                        candidates.append({
                            "address": item.get("tokenAddress"),
                            "chain": item.get("chainId", "solana")
                        })
                    return self._fetch_details_concurrently(candidates)
            except:
                pass
            return []

        # Execute fetchers in parallel - hanya boosts dan profiles
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_boosts = executor.submit(fetch_boosts)
            future_profiles = executor.submit(fetch_profiles)
            
            # Collect results
            try:
                pairs_list.extend(future_boosts.result(timeout=15))
            except:
                pass
            try:
                pairs_list.extend(future_profiles.result(timeout=15))
            except:
                pass
        
        # Remove duplicates
        seen = set()
        unique_pairs = []
        for p in pairs_list:
            pair_key = p.get('pair_address') or p.get('base_token_address')
            if pair_key and pair_key not in seen:
                seen.add(pair_key)
                unique_pairs.append(p)
        
        # Sort by creation time (newest first) if available, then by liquidity
        unique_pairs.sort(key=lambda x: (x.get('pair_created_at') or 0, x.get('liquidity_usd') or 0), reverse=True)
        
        print(f"{Fore.CYAN}[INFO] Found {len(unique_pairs)} unique tokens{Style.RESET_ALL}")
        
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
        Mengambil data historis REAL menggunakan GeckoTerminal API
        
        Args:
            pair_address: Alamat pair
            chain_id: ID chain (dexscreener format, will be mapped to geckoterminal)
            
        Returns:
            DataFrame dengan data historis OHLCV
        """
        print(f"{Fore.CYAN}[INFO] Mengambil data historis REAL (GeckoTerminal) untuk {pair_address[:10]}...{Style.RESET_ALL}")
        
        # Mapping chain ID DexScreener -> GeckoTerminal
        gecko_chain = self._map_chain_to_gecko(chain_id)
        
        # GeckoTerminal API Endpoint for OHLCV
        # https://api.geckoterminal.com/api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}
        # Timeframe 'hour' gives us hourly candles
        url = f"https://api.geckoterminal.com/api/v2/networks/{gecko_chain}/pools/{pair_address}/ohlcv/hour"
        
        params = {
            "limit": 100  # Get last 100 hours
        }
        
        try:
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                ohlcv_list = data.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
                
                if not ohlcv_list:
                    print(f"{Fore.YELLOW}[WARNING] Tidak ada data candle ditemukan di GeckoTerminal{Style.RESET_ALL}")
                    return pd.DataFrame()
                
                # Parse OHLCV list [timestamp, open, high, low, close, volume]
                # Timestamp is in seconds
                
                parsed_data = []
                for candle in ohlcv_list:
                    parsed_data.append({
                        "timestamp": datetime.fromtimestamp(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5])
                    })
                
                # Sort ascending (oldest first) for training
                df = pd.DataFrame(parsed_data).sort_values("timestamp")
                print(f"{Fore.GREEN}[SUCCESS] Berhasil mengambil {len(df)} candle asli dari GeckoTerminal{Style.RESET_ALL}")
                return df
                
            elif response.status_code == 404:
                print(f"{Fore.YELLOW}[WARNING] Pool tidak ditemukan di GeckoTerminal (Mungkin terlalu baru?){Style.RESET_ALL}")
                return pd.DataFrame()
            elif response.status_code == 429:
                print(f"{Fore.YELLOW}[WARN] GeckoTerminal Rate Limit Hit{Style.RESET_ALL}")
                return pd.DataFrame()
            else:
                print(f"{Fore.RED}[ERR] GeckoTerminal Error {response.status_code}{Style.RESET_ALL}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"{Fore.RED}[ERR] Gagal mengambil data GeckoTerminal: {e}{Style.RESET_ALL}")
            return pd.DataFrame()

    def _map_chain_to_gecko(self, chain_id: str) -> str:
        """Map DexScreener chain IDs to GeckoTerminal network IDs"""
        mapping = {
            "solana": "solana",
            "ethereum": "eth",
            "bsc": "bsc",
            "polygon": "polygon_pos",
            "arbitrum": "arbitrum",
            "optimism": "optimism",
            "avalanche": "avax",
            "base": "base",
            "fantom": "ftm",
            "cronos": "cronos"
        }
        return mapping.get(chain_id.lower(), chain_id.lower())

    # Legacy synthetic method removed/not used anymore
    
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
