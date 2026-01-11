"""
SNA Analyzer V2 - Enhanced Drone Emprit Style
==============================================
- More metrics: Transaction velocity, Unique wallets proxy
- Weighted scoring system
- Better hype detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from colorama import Fore, Style
from datetime import datetime


class HypeLevel(Enum):
    DORMANT = "💤 Dormant"
    WARMING = "🌡️ Warming"
    HEATING = "🔥 Heating"
    EXPLODING = "🚀 Exploding"
    PARABOLIC = "💎 Parabolic"


@dataclass
class SNAResult:
    token_symbol: str
    token_address: str
    chain_id: str
    volume_5m: float
    volume_1h: float
    volume_6h: float
    volume_24h: float
    volume_spike_1h: float
    volume_spike_5m: float
    buy_sell_ratio: float
    liquidity_score: float
    hype_level: HypeLevel
    sna_score: float
    price_change_1h: float
    market_cap: float
    is_potential_pump: bool
    # New V2 metrics
    tx_velocity: float  # Transaction speed
    momentum_score: float  # Price momentum
    whale_indicator: float  # Large transaction indicator


class SNAAnalyzer:
    """Enhanced SNA Analyzer V2 with more metrics"""
    
    VOLUME_SPIKE_THRESHOLD = 400  # Lowered for better sensitivity
    BUY_SELL_RATIO_MIN = 1.1
    LIQUIDITY_MIN = 5000
    
    def __init__(self):
        self.analyzed_tokens: List[SNAResult] = []
    
    def analyze_token(self, token_data: Dict) -> SNAResult:
        # Extract volumes
        volume_5m = float(token_data.get("volume_5m", 0))
        volume_1h = float(token_data.get("volume_1h", 0))
        volume_6h = float(token_data.get("volume_6h", 0))
        volume_24h = float(token_data.get("volume_24h", 0))
        
        # Calculate average hourly volume
        avg_hourly_6h = volume_6h / 6 if volume_6h > 0 else 1
        avg_hourly_24h = volume_24h / 24 if volume_24h > 0 else 1
        avg_hourly = max(avg_hourly_6h, avg_hourly_24h, 1)
        
        # Volume spikes
        volume_spike_1h = ((volume_1h / avg_hourly) - 1) * 100
        avg_5m = volume_1h / 12 if volume_1h > 0 else 1
        volume_spike_5m = ((volume_5m / avg_5m) - 1) * 100
        
        # Buy/sell ratio
        buys_1h = int(token_data.get("txns_buys_1h", 0))
        sells_1h = int(token_data.get("txns_sells_1h", 0))
        buys_5m = int(token_data.get("txns_buys_5m", 0))
        sells_5m = int(token_data.get("txns_sells_5m", 0))
        buy_sell_ratio = buys_1h / max(sells_1h, 1)
        
        # Transaction velocity (txns per minute)
        tx_velocity = (buys_1h + sells_1h) / 60
        tx_velocity_5m = (buys_5m + sells_5m) / 5
        
        # Whale indicator (large avg transaction size)
        total_txns = buys_1h + sells_1h
        avg_tx_size = volume_1h / max(total_txns, 1)
        whale_indicator = min(avg_tx_size / 1000, 100)  # Normalize
        
        # Liquidity score
        liquidity_usd = float(token_data.get("liquidity_usd", 0))
        liquidity_score = self._calc_liquidity_score(liquidity_usd)
        
        # Price momentum
        price_change_1h = float(token_data.get("price_change_1h", 0))
        price_change_5m = float(token_data.get("price_change_5m", 0))
        price_change_24h = float(token_data.get("price_change_24h", 0))
        momentum_score = self._calc_momentum_score(price_change_5m, price_change_1h, price_change_24h)
        
        # Hype level
        hype_level = self._get_hype_level(volume_spike_1h, tx_velocity, buy_sell_ratio)
        
        # SNA Score V2 (weighted formula)
        sna_score = self._calc_sna_score_v2(
            volume_spike_1h=volume_spike_1h,
            volume_spike_5m=volume_spike_5m,
            buy_sell_ratio=buy_sell_ratio,
            liquidity_score=liquidity_score,
            price_change_1h=price_change_1h,
            tx_velocity=tx_velocity,
            momentum_score=momentum_score,
            whale_indicator=whale_indicator
        )
        
        # Potential pump detection (improved)
        is_pump = (
            (volume_spike_1h >= self.VOLUME_SPIKE_THRESHOLD or 
             volume_spike_5m >= 200 or
             (tx_velocity > 2 and buy_sell_ratio > 1.5)) and
            buy_sell_ratio >= self.BUY_SELL_RATIO_MIN and
            liquidity_usd >= self.LIQUIDITY_MIN and
            momentum_score > 30
        )
        
        return SNAResult(
            token_symbol=token_data.get("base_token", "UNKNOWN"),
            token_address=token_data.get("base_token_address", ""),
            chain_id=token_data.get("chain_id", ""),
            volume_5m=volume_5m, volume_1h=volume_1h, 
            volume_6h=volume_6h, volume_24h=volume_24h,
            volume_spike_1h=volume_spike_1h, volume_spike_5m=volume_spike_5m,
            buy_sell_ratio=buy_sell_ratio, liquidity_score=liquidity_score,
            hype_level=hype_level, sna_score=sna_score,
            price_change_1h=price_change_1h,
            market_cap=float(token_data.get("market_cap", 0)),
            is_potential_pump=is_pump,
            tx_velocity=tx_velocity,
            momentum_score=momentum_score,
            whale_indicator=whale_indicator
        )
    
    def analyze_batch(self, tokens_df: pd.DataFrame) -> List[SNAResult]:
        print(f"{Fore.CYAN}[SNA-V2] Analyzing {len(tokens_df)} tokens with enhanced metrics...{Style.RESET_ALL}")
        results = [self.analyze_token(row.to_dict()) for _, row in tokens_df.iterrows()]
        results.sort(key=lambda x: x.sna_score, reverse=True)
        pumps = [r for r in results if r.is_potential_pump]
        print(f"{Fore.GREEN}[SNA-V2] Found {len(pumps)} potential PUMP candidates{Style.RESET_ALL}")
        return results
    
    def filter_potential_pumps(self, results: List[SNAResult], min_score: float = 40) -> List[SNAResult]:
        return [r for r in results if r.is_potential_pump and r.sna_score >= min_score]
    
    def _calc_liquidity_score(self, liq: float) -> float:
        """Logarithmic liquidity score 0-100"""
        if liq <= 0:
            return 0
        log_liq = np.log10(max(liq, 1))
        # $1k = 30, $10k = 40, $100k = 60, $1M = 80
        return min(log_liq * 20, 100)
    
    def _calc_momentum_score(self, pc_5m: float, pc_1h: float, pc_24h: float) -> float:
        """Calculate price momentum score 0-100"""
        # Recent momentum more important
        score = pc_5m * 0.5 + pc_1h * 0.35 + pc_24h * 0.15
        # Normalize to 0-100
        return min(max(score + 50, 0), 100)
    
    def _get_hype_level(self, spike: float, velocity: float, ratio: float) -> HypeLevel:
        """Multi-factor hype level detection"""
        hype_score = spike * 0.5 + velocity * 100 + (ratio - 1) * 50
        
        if hype_score > 600:
            return HypeLevel.PARABOLIC
        elif hype_score > 400:
            return HypeLevel.EXPLODING
        elif hype_score > 200:
            return HypeLevel.HEATING
        elif hype_score > 50:
            return HypeLevel.WARMING
        return HypeLevel.DORMANT
    
    def _calc_sna_score_v2(self, volume_spike_1h: float, volume_spike_5m: float,
                           buy_sell_ratio: float, liquidity_score: float,
                           price_change_1h: float, tx_velocity: float,
                           momentum_score: float, whale_indicator: float) -> float:
        """
        Enhanced SNA Score Formula V2 (0-100)
        
        Weights:
        - 25% Volume Spike
        - 20% Buy/Sell Ratio  
        - 15% Liquidity
        - 15% Momentum
        - 15% Transaction Velocity
        - 10% Whale Activity
        """
        # Volume spike score (0-25)
        vol_score = min(max(volume_spike_1h, volume_spike_5m) / 20, 25)
        
        # Buy/sell ratio score (0-20)
        ratio_score = min((buy_sell_ratio - 1) * 10, 20) if buy_sell_ratio > 1 else 0
        
        # Liquidity score (0-15)
        liq_score = liquidity_score * 0.15
        
        # Momentum score (0-15)
        mom_score = momentum_score * 0.15
        
        # Transaction velocity score (0-15)
        tx_score = min(tx_velocity * 5, 15)
        
        # Whale score (0-10)
        whale_score = min(whale_indicator * 0.1, 10)
        
        total = vol_score + ratio_score + liq_score + mom_score + tx_score + whale_score
        return min(total, 100)
    
    def print_report(self, results: List[SNAResult], top_n: int = 10):
        print(f"\n{Fore.YELLOW}{'='*75}")
        print(f"  🔍 SNA V2 REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*75}{Style.RESET_ALL}\n")
        
        for i, r in enumerate(results[:top_n], 1):
            color = Fore.GREEN if r.sna_score > 60 else Fore.YELLOW if r.sna_score > 40 else Fore.WHITE
            pump = f"{Fore.RED}🎯 PUMP!{Style.RESET_ALL}" if r.is_potential_pump else ""
            
            print(f"{color}#{i:2d} {r.token_symbol:12s} | SNA: {r.sna_score:5.1f} | {r.hype_level.value}")
            print(f"    Spike: {r.volume_spike_1h:>6.0f}% | B/S: {r.buy_sell_ratio:.2f} | TxVel: {r.tx_velocity:.1f}/min | Mom: {r.momentum_score:.0f} {pump}{Style.RESET_ALL}")


def detect_volume_spike(token_data: Dict, threshold: float = 400) -> Tuple[bool, float]:
    result = SNAAnalyzer().analyze_token(token_data)
    return result.volume_spike_1h >= threshold, result.volume_spike_1h
