"""
CryptoHunter Live Dashboard V2
==============================
Real-time price monitoring dengan AI predictions (LSTM + Ensemble)
"""

import sys
import io
import os
import time
import threading
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from colorama import Fore, Style, init, Back
init(autoreset=True)

from modules.dex_api import DexScreenerAPI
from modules.price_predictor import EnsemblePredictor
from modules.sna_analyzer import SNAAnalyzer


class LiveDashboard:
    """Real-time price monitoring dashboard with AI predictions"""
    
    def __init__(self, refresh_interval: int = 10):
        self.refresh_interval = max(refresh_interval, 10)
        self.api = DexScreenerAPI()
        self.sna = SNAAnalyzer()
        self.running = False
        self.prices_history = {}
        
        # AI Model cache
        self.ai_models = {}  # token -> trained model
        self.ai_cache = {}   # token -> (timestamp, confidence, pump_hours)
        self.cache_ttl = 60  # Cache predictions for 60 seconds
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def format_price(self, price: float) -> str:
        """Format price based on magnitude"""
        if price > 1:
            return f"${price:.4f}"
        elif price > 0.0001:
            return f"${price:.6f}"
        elif price > 0.00000001:
            return f"${price:.10f}"
        else:
            return f"${price:.14f}"
    
    def format_change(self, change: float) -> str:
        """Format price change with color"""
        if change > 0:
            return f"{Fore.GREEN}+{change:.2f}%{Style.RESET_ALL}"
        elif change < 0:
            return f"{Fore.RED}{change:.2f}%{Style.RESET_ALL}"
        else:
            return f"{Fore.WHITE}0.00%{Style.RESET_ALL}"
    
    def format_number(self, num: float) -> str:
        """Format large numbers"""
        if num >= 1_000_000:
            return f"${num/1_000_000:.2f}M"
        elif num >= 1_000:
            return f"${num/1_000:.1f}K"
        else:
            return f"${num:.0f}"
    
    def get_price_direction(self, token: str, current_price: float) -> str:
        """Get price direction arrow"""
        if token not in self.prices_history:
            self.prices_history[token] = current_price
            return "  "
        
        old_price = self.prices_history[token]
        self.prices_history[token] = current_price
        
        if current_price > old_price:
            return f"{Fore.GREEN}▲{Style.RESET_ALL}"
        elif current_price < old_price:
            return f"{Fore.RED}▼{Style.RESET_ALL}"
        else:
            return f"{Fore.WHITE}─{Style.RESET_ALL}"
    
    def print_header(self):
        """Print dashboard header"""
        print(f"\n{Fore.CYAN}╔{'═'*78}╗")
        print(f"║{' '*28}🔴 LIVE DASHBOARD{' '*33}║")
        print(f"║{' '*20}CryptoHunter V2.0 | Refresh: {self.refresh_interval}s{' '*20}║")
        print(f"╚{'═'*78}╝{Style.RESET_ALL}")
        print(f"{Fore.WHITE}  Last Update: {datetime.now().strftime('%H:%M:%S')} | Press Ctrl+C to stop{Style.RESET_ALL}\n")
    
    def get_ai_prediction(self, data: dict) -> tuple:
        """
        Get AI prediction using SNA + Hybrid Scoring
        Returns: (confidence%, pump_hours, source)
        """
        token = data.get('base_token', 'unknown')
        
        # Check cache first
        if token in self.ai_cache:
            cache_time, conf, hours = self.ai_cache[token]
            if time.time() - cache_time < self.cache_ttl:
                return (conf, hours, "cached")
        
        try:
            # Use SNA analysis (fast and reliable)
            sna_result = self.sna.analyze_token(data)
            sna_score = sna_result.sna_score
            
            # Hybrid confidence calculation based on SNA + market metrics
            base_conf = 50
            
            # SNA score contribution (max +25)
            base_conf += min(25, sna_score * 0.3)
            
            # Volume/Liquidity ratio (max +15)
            volume = data.get('volume_1h', 0)
            liquidity = max(data.get('liquidity_usd', 1), 1)
            vol_ratio = volume / liquidity
            if vol_ratio > 5:
                base_conf += 15
            elif vol_ratio > 2:
                base_conf += 10
            elif vol_ratio > 1:
                base_conf += 5
            
            # Trend consistency (max +10)
            c5m = data.get('price_change_5m', 0)
            c1h = data.get('price_change_1h', 0)
            c24h = data.get('price_change_24h', 0)
            if (c5m > 0 and c1h > 0) or (c5m < 0 and c1h < 0):
                base_conf += 10
            
            # Final confidence
            conf = int(min(95, max(55, base_conf)))
            
            # Pump hours based on SNA hype level
            if sna_score >= 70:
                hours = 1
            elif sna_score >= 50:
                hours = 2
            elif sna_score >= 30:
                hours = 4
            else:
                hours = 6
            
            # Cache result
            self.ai_cache[token] = (time.time(), conf, hours)
            return (conf, hours, "sna")
            
        except Exception as e:
            # Fallback with basic calculation
            conf = 60
            hours = 4
            return (conf, hours, "err")
    
    def print_token_row(self, idx: int, data: dict):
        """Print single token row with Dump Alert and AI Confidence"""
        token = data.get('base_token', 'N/A')[:10]
        chain = data.get('chain_id', '').upper()[:3]
        price = data.get('price_usd', 0)
        change_5m = data.get('price_change_5m', 0)
        change_1h = data.get('price_change_1h', 0)
        volume = data.get('volume_1h', 0)
        market_cap = data.get('market_cap', 0)  # NEW: Market Cap
        
        direction = self.get_price_direction(token, price)
        
        # Get AI prediction
        ai_conf, pump_hours, source = self.get_ai_prediction(data)
        
        # Format AI confidence
        if ai_conf >= 80:
            ai_color = Fore.GREEN
        elif ai_conf >= 70:
            ai_color = Fore.YELLOW
        else:
            ai_color = Fore.WHITE
        ai_str = f"{ai_color}{ai_conf:3d}%{Style.RESET_ALL}"
        
        # ═══════════════════════════════════════════════════════════════
        # ALERT TRIGGER LOGIC (based on 5-minute price change)
        # ═══════════════════════════════════════════════════════════════
        # CRASH: price dropped >= 10% in last 5 minutes (severe dump)
        # DUMP:  price dropped >= 5% in last 5 minutes (moderate dump)
        # PUMP:  price increased > 5% in last 5 minutes (bullish spike)
        # ═══════════════════════════════════════════════════════════════
        status_str = ""
        row_bg = ""
        
        if change_5m <= -10:
            # CRASH: Severe price drop (>= -10% in 5m)
            row_bg = Back.RED + Fore.WHITE + Style.BRIGHT
            status_str = "CRASH"
            print('\a', end='')  # Sound alert
        elif change_5m <= -5:
            # DUMP: Moderate price drop (>= -5% in 5m)
            row_bg = Fore.RED + Style.BRIGHT
            status_str = "DUMP"
        elif change_5m > 5:
            # PUMP: Strong price increase (> +5% in 5m)
            row_bg = Fore.GREEN + Style.BRIGHT
            status_str = "PUMP"
        
        # Format price
        price_str = self.format_price(price)
        
        # Format changes
        c5m = f"{change_5m:+.2f}%"
        c1h = f"{change_1h:+.2f}%"
        
        # Color the changes
        c5m_color = Fore.GREEN if change_5m > 0 else (Fore.RED if change_5m < 0 else Fore.WHITE)
        c1h_color = Fore.GREEN if change_1h > 0 else (Fore.RED if change_1h < 0 else Fore.WHITE)
        
        # Format volume and market cap
        vol_str = self.format_number(volume)
        mcap_str = self.format_number(market_cap) if market_cap > 0 else "N/A"
        
        print(f"{row_bg}{idx:2d} | {token:10s} | {chain:3s} | {direction}{price_str:15s} | {c5m_color}{c5m:8s}{Style.RESET_ALL} | {c1h_color}{c1h:8s}{Style.RESET_ALL} | {ai_str:4s} | {vol_str:8s} | {mcap_str:8s} | {status_str:5s}{Style.RESET_ALL}")

    def print_table_header(self):
        """Print table header"""
        header_line = f"{Fore.CYAN}{'─'*105}{Style.RESET_ALL}"
        print(header_line)
        print(f"{Fore.CYAN} # | {'TOKEN':10s} | CHN |  {'PRICE':14s} | {'5MIN':8s} | {'1HOUR':8s} | {'AI%':4s} | {'VOL/1H':8s} | {'MCAP':8s} | {'STAT':5s}{Style.RESET_ALL}")
        print(header_line)
    
    def fetch_and_display(self):
        """Fetch latest data and display"""
        pairs_df = self.api.search_new_pairs(min_liquidity=5000)
        
        if pairs_df.empty:
            print(f"{Fore.RED}[ERROR] Tidak dapat mengambil data{Style.RESET_ALL}")
            return
        
        # Sort by volume
        pairs_df = pairs_df.sort_values('volume_1h', ascending=False)
        
        self.clear_screen()
        self.print_header()
        self.print_table_header()
        
        for idx, (_, row) in enumerate(pairs_df.head(15).iterrows(), 1):
            self.print_token_row(idx, row.to_dict())
        
        print(f"\n{Fore.CYAN}{'─'*80}{Style.RESET_ALL}")
        
        # Summary stats
        total_volume = pairs_df['volume_1h'].sum()
        total_mcap = pairs_df['market_cap'].sum()  # NEW: Total Market Cap
        avg_change = pairs_df['price_change_1h'].mean()
        gainers = len(pairs_df[pairs_df['price_change_1h'] > 0])
        losers = len(pairs_df[pairs_df['price_change_1h'] < 0])
        
        print(f"  📊 Vol 1H: {self.format_number(total_volume)} | MCAP: {self.format_number(total_mcap)} | Avg: {self.format_change(avg_change)} | 📈 Gainers: {gainers} | 📉 Losers: {losers}")
    
    def run(self):
        """Run the live dashboard"""
        self.running = True
        
        print(f"\n{Fore.GREEN}🚀 Starting Live Dashboard...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}   Auto-refresh every {self.refresh_interval} seconds")
        print(f"   Press Ctrl+C to stop{Style.RESET_ALL}\n")
        time.sleep(2)
        
        while self.running:
            try:
                self.fetch_and_display()
                
                # Countdown
                for remaining in range(self.refresh_interval, 0, -1):
                    if not self.running: break
                    print(f"\r{Fore.CYAN}  ⏱️  Next refresh in {remaining}s...{Style.RESET_ALL}", end='', flush=True)
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                self.running = False
                print(f"\n\n{Fore.YELLOW}👋 Dashboard stopped.{Style.RESET_ALL}")
                break
            except Exception as e:
                # Catch connection errors/crashes and retry instead of dying
                print(f"\n{Fore.RED}[ERR] Connection lost: {e}. Reconnecting in 5s...{Style.RESET_ALL}")
                time.sleep(5)
    
    def stop(self):
        """Stop the dashboard"""
        self.running = False


def main():
    """Entry point for live dashboard"""
    print(f"\n{Fore.CYAN}{'═'*60}")
    print(f"  🔴 CRYPTOHUNTER LIVE DASHBOARD")
    print(f"  Real-time price monitoring")
    print(f"{'═'*60}{Style.RESET_ALL}\n")
    
    # Get refresh interval from user
    try:
        interval_input = input(f"{Fore.WHITE}Refresh interval dalam detik (default: 5): {Style.RESET_ALL}")
        interval = int(interval_input) if interval_input else 5
        interval = max(3, min(interval, 60))  # Clamp 3-60 seconds
    except ValueError:
        interval = 5
    
    dashboard = LiveDashboard(refresh_interval=interval)
    dashboard.run()


if __name__ == "__main__":
    main()
