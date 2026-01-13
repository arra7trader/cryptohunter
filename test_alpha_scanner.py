
"""
MANUAL TEST: Alpha Scanner Pipeline (Phase 3)
=============================================
1. Fetch latest tokens from DexScreener
2. Analyze with SNA V3 (Sentiment)
3. Fetch Real Candles from GeckoTerminal
4. Verify Security with RugCheck
"""

import sys
import os
import pandas as pd
from colorama import Fore, Style, init

# Init colorama
init()

# Force UTF-8 for Windows console
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.dex_api import search_new_pairs, get_historical_data
from modules.sna_analyzer import SNAAnalyzer

def test_alpha_pipeline():
    print(f"\n{Fore.YELLOW}=== ALPHA SCANNER PIPELINE TEST ==={Style.RESET_ALL}\n")
    
    # 1. Search for new pairs (Solana preferred for RugCheck)
    print(f"{Fore.CYAN}[1] Searching for new Solana pairs...{Style.RESET_ALL}")
    pairs_df = search_new_pairs(chain="solana", min_liquidity=5000)
    
    if pairs_df.empty:
        print(f"{Fore.RED}[FAIL] No pairs found. Exiting.{Style.RESET_ALL}")
        return

    print(f"{Fore.GREEN}[OK] Found {len(pairs_df)} candidates.{Style.RESET_ALL}")
    
    # Take top candidate
    candidate = pairs_df.iloc[0]
    token_address = candidate['base_token_address']
    symbol = candidate['base_token']
    chain = candidate['chain_id']
    
    print(f"\nTarget: {symbol} ({token_address}) on {chain}")
    
    # 2. Analyze Sentiment (SNA)
    print(f"\n{Fore.CYAN}[2] Analyzing Sentiment...{Style.RESET_ALL}")
    analyzer = SNAAnalyzer()
    # Enable Security Check
    result = analyzer.analyze_token(candidate.to_dict(), check_security=True)
    
    print(f"Alpha Score: {result.alpha_score:.1f}/100")
    print(f"Rating: {result.alpha_rating.value}")
    print(f"Safety: {result.is_safe} (Score: {result.safety_score})")
    
    if result.security_risks:
        print(f"Risks: {result.security_risks}")
    
    # 3. Fetch Real Candles
    print(f"\n{Fore.CYAN}[3] Fetching Real Historical Data (GeckoTerminal)...{Style.RESET_ALL}")
    pair_address = candidate['pair_address']
    df_hist = get_historical_data(pair_address, chain)
    
    if not df_hist.empty:
        print(f"{Fore.GREEN}[OK] Fetched {len(df_hist)} real candles.{Style.RESET_ALL}")
        print(df_hist.head(3))
        print("...")
        print(df_hist.tail(3))
    else:
        print(f"{Fore.RED}[FAIL] Could not fetch candles.{Style.RESET_ALL}")
        
    print(f"\n{Fore.YELLOW}=== TEST COMPLETE ==={Style.RESET_ALL}")

if __name__ == "__main__":
    test_alpha_pipeline()
