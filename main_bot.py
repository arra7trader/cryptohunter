"""
CryptoHunter Bot - Main Orchestrator
=====================================
Bot pencari koin micin/potensial dari DEX
Menggunakan DexScreener API + SNA Analysis + LSTM Prediction

Author: AI Data Scientist & Algo Trader
"""

import sys
import io
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from datetime import datetime
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Import modules
from modules.dex_api import DexScreenerAPI, search_new_pairs, get_historical_data
from modules.sna_analyzer import SNAAnalyzer, SNAResult, HypeLevel
from modules.enhanced_predictor import EnhancedPredictor


class CryptoHunterBot:
    """Main bot orchestrator V2.1 - Enhanced AI Engine"""
    
    VERSION = "2.1.0"
    
    def __init__(self, min_liquidity: float = 5000, min_sna_score: float = 40):
        self.min_liquidity = min_liquidity
        self.min_sna_score = min_sna_score
        self.dex_api = DexScreenerAPI()
        self.sna_analyzer = SNAAnalyzer()
        self.enhanced_predictor = EnhancedPredictor()  # New Enhanced Predictor
        self.results = []
    
    def print_banner(self):
        """Print startup banner"""
        banner = f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   {Fore.YELLOW}🔍 CRYPTOHUNTER BOT v{self.VERSION}{Fore.CYAN}                                      ║
║   {Fore.WHITE}Micin Hunter & Pump Predictor{Fore.CYAN}                                  ║
║                                                                  ║
║   {Fore.GREEN}▸ DexScreener API{Fore.CYAN}  {Fore.GREEN}▸ SNA Analysis{Fore.CYAN}  {Fore.GREEN}▸ Sentiment Aware{Fore.CYAN}       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
        print(banner)
    
    def run(self, top_n: int = 10):
        """Main execution flow"""
        self.print_banner()
        print(f"{Fore.WHITE}📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}\n")
        
        # Step 1: Search for potential coins
        print(f"{Fore.YELLOW}{'─'*60}")
        print(f"  STEP 1: Mencari Koin Potensial dari DEX")
        print(f"{'─'*60}{Style.RESET_ALL}\n")
        
        pairs_df = search_new_pairs(min_liquidity=self.min_liquidity)
        
        if pairs_df.empty:
            print(f"{Fore.RED}[ERROR] Tidak ada pair ditemukan. Coba lagi nanti.{Style.RESET_ALL}")
            return []
        
        # Step 2: SNA Analysis
        print(f"\n{Fore.YELLOW}{'─'*60}")
        print(f"  STEP 2: Analisis SNA (Drone Emprit Style)")
        print(f"{'─'*60}{Style.RESET_ALL}\n")
        
        sna_results = self.sna_analyzer.analyze_batch(pairs_df)
        
        # Filter by SNA score
        filtered = [r for r in sna_results if r.sna_score >= self.min_sna_score]
        
        if not filtered:
            print(f"{Fore.YELLOW}[INFO] Tidak ada token yang lolos filter SNA{Style.RESET_ALL}")
            filtered = sna_results[:5]  # Take top 5 anyway
        
        # Step 3: Enhanced AI Prediction
        print(f"\n{Fore.YELLOW}{'─'*60}")
        print(f"  STEP 3: Enhanced Prediction (AI + Sentiment + Global Data)")
        print(f"{'─'*60}{Style.RESET_ALL}\n")
        
        final_results = []
        
        for sna_result in filtered[:top_n]:
            try:
                # Prediction is now handled by EnhancedPredictor which fetches its own data/history when needed
                # However, our EnhancedPredictor expects a symbol.
                # SNA result gives us token symbol.
                
                print(f"Analyzing {sna_result.token_symbol}...")
                
                # We reuse the symbol from SNA result. Note: EnhancedPredictor handles data fetching internally.
                # But to preserve the specific pair found by DexScreener, maybe we should pass context?
                # For now EnhancedPredictor uses aggregated data for major coins or fetches anew.
                # Since this is a "Micin Hunter", these tokens might NOT be on Binance.
                # EnhancedPredictor gracefully handles missing Binance data.
                
                prediction = self.enhanced_predictor.predict(sna_result.token_symbol, use_binance=True)
                
                if not prediction:
                     print(f"{Fore.RED}   [SKIP] Prediction failed for {sna_result.token_symbol}{Style.RESET_ALL}")
                     continue
                
                # Get price and token address from pairs_df
                pair_row = pairs_df[pairs_df['base_token'] == sna_result.token_symbol].iloc[0]
                
                final_results.append({
                    "token": sna_result.token_symbol,
                    "chain": sna_result.chain_id,
                    "token_address": pair_row.get('base_token_address', 'N/A'),
                    "price_usd": pair_row.get('price_usd', 0),
                    "liquidity_usd": pair_row.get('liquidity_usd', 0),
                    "market_cap": pair_row.get('market_cap', 0),
                    "pair_url": pair_row.get('url', ''),
                    "sna_score": sna_result.sna_score,
                    "hype_level": sna_result.hype_level.value,
                    "volume_spike": sna_result.volume_spike_1h,
                    "buy_sell_ratio": sna_result.buy_sell_ratio,
                    "is_potential_pump": sna_result.is_potential_pump,
                    "prediction": prediction # This is now an EnhancedPrediction object
                })
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                print(f"{Fore.RED}[ERROR] {sna_result.token_symbol}: {e}{Style.RESET_ALL}")
                continue
        
        # Step 4: Final Report
        self._print_final_report(final_results)
        self.results = final_results
        return final_results
    
    def _print_final_report(self, results: list):
        """Print final analysis report"""
        print(f"\n{Fore.GREEN}{'═'*70}")
        print(f"  🎯 FINAL REPORT - TOP PUMP CANDIDATES")
        print(f"{'═'*70}{Style.RESET_ALL}\n")
        
        if not results:
            print(f"{Fore.YELLOW}Tidak ada hasil yang dapat ditampilkan.{Style.RESET_ALL}")
            return
        
        # Sort by prediction confidence
        results.sort(key=lambda x: x['prediction'].confidence, reverse=True)
        
        for i, r in enumerate(results, 1):
            pred = r['prediction'] # EnhancedPrediction object
            confidence = pred.confidence
            
            # Determine color based on confidence
            if confidence > 75:
                color = Fore.GREEN
                emoji = "🚀"
            elif confidence > 50:
                color = Fore.YELLOW
                emoji = "📈"
            else:
                color = Fore.WHITE
                emoji = "📊"
            
            pump_alert = f"{Fore.RED}🎯 PUMP ALERT!{Style.RESET_ALL}" if r['is_potential_pump'] else ""
            
            # Format price
            price = r.get('price_usd', 0)
            if price > 0.01:
                price_str = f"${price:.4f}"
            elif price > 0.000001:
                price_str = f"${price:.8f}"
            else:
                price_str = f"${price:.12f}"
            
            # Format liquidity & market cap
            liq = r.get('liquidity_usd', 0)
            mcap = r.get('market_cap', 0)
            liq_str = f"${liq:,.0f}" if liq > 0 else "N/A"
            mcap_str = f"${mcap:,.0f}" if mcap > 0 else "N/A"
            
            # Token address (shortened)
            token_addr = r.get('token_address', 'N/A')
            addr_short = f"{token_addr[:6]}...{token_addr[-4:]}" if len(str(token_addr)) > 12 else token_addr
            
            print(f"{color}{'─'*70}{Style.RESET_ALL}")
            print(f"{color}{emoji} #{i} {r['token']} ({r['chain'].upper()}) {pump_alert}")
            print(f"   ├─ 💰 Price      : {price_str}")
            print(f"   ├─ 📍 Token      : {addr_short}")
            print(f"   ├─ 💧 Liquidity  : {liq_str}")
            print(f"   ├─ 📊 MarketCap  : {mcap_str}")
            print(f"   ├─ SNA Score    : {r['sna_score']:.1f}/100")
            print(f"   ├─ Hype Level   : {r['hype_level']}")
            print(f"   ├─ AI Signal    : {pred.signal} ({confidence:.1f}%)")
            print(f"   ├─ Sentiment    : {pred.fear_greed_classification} ({pred.fear_greed_value})")
            print(f"   ├─ Binance Vol  : {pred.binance_volume_24h:,.2f}")
            print(f"   └─ Data Sources : {', '.join(pred.data_sources)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'═'*70}")
        print(f"  📊 SUMMARY")
        print(f"{'═'*70}{Style.RESET_ALL}")
        print(f"  Total Analyzed  : {len(results)}")
        print(f"  Pump Alerts     : {len([r for r in results if r['is_potential_pump']])}")
        print(f"  High Confidence : {len([r for r in results if r['prediction'].get('ensemble', {}).get('confidence', 0) > 75])}")
        print(f"\n{Fore.YELLOW}⚠️  DISCLAIMER: Ini bukan financial advice. DYOR!{Style.RESET_ALL}\n")


def main():
    """Entry point"""
    print(f"\n{Fore.CYAN}Initializing CryptoHunter Bot...{Style.RESET_ALL}\n")
    
    bot = CryptoHunterBot(
        min_liquidity=5000,
        min_sna_score=30
    )
    
    try:
        results = bot.run(top_n=10)
        
        if results:
            print(f"\n{Fore.GREEN}✅ Scan selesai! {len(results)} token dianalisis.{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW}⚠️ Tidak ada hasil. Coba lagi nanti.{Style.RESET_ALL}")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Bot dihentikan oleh user.{Style.RESET_ALL}")
    except Exception as e:
        print(f"\n{Fore.RED}[ERROR] {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
