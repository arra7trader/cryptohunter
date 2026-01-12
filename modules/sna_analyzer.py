"""
SNA Analyzer V3 - Alpha Hunter Edition
=======================================
Enhanced detection for REAL alpha coins:
- Token age analysis
- Buy pressure indicator
- Volume acceleration
- Liquidity sweet spot
- Smart money detection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
from colorama import Fore, Style
from datetime import datetime, timedelta


class HypeLevel(Enum):
    DORMANT = "💤 Dormant"
    WARMING = "🌡️ Warming"
    HEATING = "🔥 Heating"
    EXPLODING = "🚀 Exploding"
    PARABOLIC = "💎 Parabolic"


class AlphaRating(Enum):
    """Alpha quality rating"""
    S_TIER = "🏆 S-TIER"  # 90+ score - VERY HIGH probability
    A_TIER = "⭐ A-TIER"  # 80-89 - HIGH probability
    B_TIER = "✅ B-TIER"  # 70-79 - GOOD probability
    C_TIER = "📊 C-TIER"  # 60-69 - MODERATE probability
    D_TIER = "⚠️ D-TIER"  # Below 60 - LOW probability


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
    # V2 metrics
    tx_velocity: float
    momentum_score: float
    whale_indicator: float
    # V3 Alpha metrics
    alpha_rating: AlphaRating
    buy_pressure: float  # 0-100, higher = more buyers
    volume_acceleration: float  # Rate of volume increase
    token_age_hours: float  # Age in hours
    liquidity_ratio: float  # Liq vs MCap ratio
    alpha_score: float  # Final alpha hunting score


class SNAAnalyzer:
    """Enhanced SNA Analyzer V3 - Alpha Hunter Edition"""
    
    # V3 Alpha criteria - STRICT for high accuracy
    ALPHA_CRITERIA = {
        'min_buy_pressure': 55,      # At least 55% buyers
        'min_volume_spike': 100,     # 100% volume increase
        'min_liquidity': 3000,       # Minimum $3K liquidity
        'max_liquidity': 500000,     # Max $500K (easy to pump)
        'max_market_cap': 1000000,   # Max $1M mcap (small cap)
        'min_tx_velocity': 0.5,      # At least 0.5 tx/min
        'max_token_age_hours': 72,   # Token < 72 hours old
        'ideal_liq_ratio': (0.1, 0.5),  # Liq should be 10-50% of MCap
    }
    
    VOLUME_SPIKE_THRESHOLD = 100
    BUY_SELL_RATIO_MIN = 1.1
    LIQUIDITY_MIN = 3000
    
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
        
        # Buy/sell analysis
        buys_1h = int(token_data.get("txns_buys_1h", 0))
        sells_1h = int(token_data.get("txns_sells_1h", 0))
        buys_5m = int(token_data.get("txns_buys_5m", 0))
        sells_5m = int(token_data.get("txns_sells_5m", 0))
        buy_sell_ratio = buys_1h / max(sells_1h, 1)
        
        # BUY PRESSURE (V3) - Percentage of buy transactions
        total_txns_1h = buys_1h + sells_1h
        buy_pressure = (buys_1h / max(total_txns_1h, 1)) * 100
        
        # Transaction velocity
        tx_velocity = total_txns_1h / 60
        tx_velocity_5m = (buys_5m + sells_5m) / 5
        
        # Volume acceleration (V3) - How fast volume is growing
        if volume_6h > 0:
            recent_vol = volume_1h
            older_vol = (volume_6h - volume_1h) / 5  # Avg of other 5 hours
            volume_acceleration = ((recent_vol / max(older_vol, 1)) - 1) * 100
        else:
            volume_acceleration = 0
        
        # Whale indicator
        avg_tx_size = volume_1h / max(total_txns_1h, 1)
        whale_indicator = min(avg_tx_size / 1000, 100)
        
        # Liquidity analysis
        liquidity_usd = float(token_data.get("liquidity_usd", 0))
        liquidity_score = self._calc_liquidity_score(liquidity_usd)
        market_cap = float(token_data.get("market_cap", 0))
        
        # Liquidity ratio (V3) - Healthy is 10-50% of mcap
        liquidity_ratio = (liquidity_usd / max(market_cap, 1)) * 100
        
        # Token age (V3)
        pair_created_at = token_data.get("pair_created_at")
        token_age_hours = self._calc_token_age(pair_created_at)
        
        # Price momentum
        price_change_1h = float(token_data.get("price_change_1h", 0))
        price_change_5m = float(token_data.get("price_change_5m", 0))
        price_change_24h = float(token_data.get("price_change_24h", 0))
        momentum_score = self._calc_momentum_score(price_change_5m, price_change_1h, price_change_24h)
        
        # Hype level
        hype_level = self._get_hype_level(volume_spike_1h, tx_velocity, buy_sell_ratio)
        
        # SNA Score V2
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
        
        # ALPHA SCORE V3 - More comprehensive scoring
        alpha_score = self._calc_alpha_score(
            buy_pressure=buy_pressure,
            volume_spike=max(volume_spike_1h, volume_spike_5m),
            volume_acceleration=volume_acceleration,
            liquidity_usd=liquidity_usd,
            market_cap=market_cap,
            liquidity_ratio=liquidity_ratio,
            token_age_hours=token_age_hours,
            tx_velocity=tx_velocity,
            momentum_score=momentum_score,
            whale_indicator=whale_indicator
        )
        
        # Alpha rating
        alpha_rating = self._get_alpha_rating(alpha_score)
        
        # Potential pump detection V3 - STRICTER
        is_pump = self._detect_alpha_pump(
            alpha_score=alpha_score,
            buy_pressure=buy_pressure,
            volume_spike=max(volume_spike_1h, volume_spike_5m),
            liquidity_usd=liquidity_usd,
            market_cap=market_cap,
            token_age_hours=token_age_hours,
            tx_velocity=tx_velocity
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
            market_cap=market_cap,
            is_potential_pump=is_pump,
            tx_velocity=tx_velocity,
            momentum_score=momentum_score,
            whale_indicator=whale_indicator,
            # V3 fields
            alpha_rating=alpha_rating,
            buy_pressure=buy_pressure,
            volume_acceleration=volume_acceleration,
            token_age_hours=token_age_hours,
            liquidity_ratio=liquidity_ratio,
            alpha_score=alpha_score
        )
    
    def _calc_token_age(self, pair_created_at) -> float:
        """Calculate token age in hours"""
        if not pair_created_at:
            return 999  # Unknown = treat as old
        try:
            if isinstance(pair_created_at, (int, float)):
                created_time = datetime.fromtimestamp(pair_created_at / 1000)
            else:
                return 999
            age_hours = (datetime.now() - created_time).total_seconds() / 3600
            return max(age_hours, 0)
        except:
            return 999
    
    def _calc_alpha_score(self, buy_pressure: float, volume_spike: float,
                          volume_acceleration: float, liquidity_usd: float,
                          market_cap: float, liquidity_ratio: float,
                          token_age_hours: float, tx_velocity: float,
                          momentum_score: float, whale_indicator: float) -> float:
        """
        Alpha Score V3 (0-100) - Comprehensive scoring for pump potential
        
        Weights:
        - 25% Buy Pressure (most important - more buyers = pump incoming)
        - 20% Volume Dynamics (spike + acceleration)
        - 15% Token Freshness (newer = more volatile)
        - 15% Liquidity Sweet Spot
        - 15% Momentum & Velocity
        - 10% Whale Activity
        """
        score = 0
        
        # 1. Buy Pressure Score (0-25)
        # 50% = neutral, 60%+ = bullish, 70%+ = very bullish
        if buy_pressure >= 70:
            bp_score = 25
        elif buy_pressure >= 60:
            bp_score = 20
        elif buy_pressure >= 55:
            bp_score = 15
        elif buy_pressure >= 50:
            bp_score = 10
        else:
            bp_score = 5
        score += bp_score
        
        # 2. Volume Dynamics Score (0-20)
        vol_score = 0
        if volume_spike >= 500:
            vol_score += 12
        elif volume_spike >= 200:
            vol_score += 9
        elif volume_spike >= 100:
            vol_score += 6
        elif volume_spike >= 50:
            vol_score += 3
            
        if volume_acceleration >= 200:
            vol_score += 8
        elif volume_acceleration >= 100:
            vol_score += 6
        elif volume_acceleration >= 50:
            vol_score += 4
        score += min(vol_score, 20)
        
        # 3. Token Freshness Score (0-15)
        # Newer tokens are more volatile and can pump harder
        if token_age_hours <= 6:
            age_score = 15  # Very fresh - high volatility
        elif token_age_hours <= 24:
            age_score = 12  # Fresh
        elif token_age_hours <= 48:
            age_score = 9   # Recent
        elif token_age_hours <= 72:
            age_score = 6   # Still new
        else:
            age_score = 3   # Established
        score += age_score
        
        # 4. Liquidity Sweet Spot Score (0-15)
        # Too low = can't trade, too high = hard to pump
        C = self.ALPHA_CRITERIA
        if C['min_liquidity'] <= liquidity_usd <= C['max_liquidity']:
            if 5000 <= liquidity_usd <= 50000:  # Ideal range
                liq_score = 15
            elif 3000 <= liquidity_usd <= 100000:
                liq_score = 12
            else:
                liq_score = 8
        elif liquidity_usd < C['min_liquidity']:
            liq_score = 3  # Too risky
        else:
            liq_score = 5  # Too established
        
        # Bonus for good liquidity ratio
        if 10 <= liquidity_ratio <= 50:
            liq_score = min(liq_score + 3, 15)
        score += liq_score
        
        # 5. Momentum & Velocity Score (0-15)
        mv_score = 0
        mv_score += min(momentum_score * 0.1, 8)
        mv_score += min(tx_velocity * 3, 7)
        score += min(mv_score, 15)
        
        # 6. Whale Activity Score (0-10)
        # Some whale activity is good, too much is risky
        if 10 <= whale_indicator <= 50:
            whale_score = 10  # Healthy whale interest
        elif whale_indicator < 10:
            whale_score = 5   # Low interest
        else:
            whale_score = 3   # Too concentrated
        score += whale_score
        
        return min(score, 100)
    
    def _detect_alpha_pump(self, alpha_score: float, buy_pressure: float,
                          volume_spike: float, liquidity_usd: float,
                          market_cap: float, token_age_hours: float,
                          tx_velocity: float) -> bool:
        """
        Strict alpha pump detection - only flag high confidence signals
        """
        C = self.ALPHA_CRITERIA
        
        # Must meet ALL core criteria
        core_criteria = (
            alpha_score >= 70 and  # High alpha score
            buy_pressure >= C['min_buy_pressure'] and  # More buyers than sellers
            liquidity_usd >= C['min_liquidity'] and  # Minimum liquidity
            tx_velocity >= C['min_tx_velocity']  # Active trading
        )
        
        if not core_criteria:
            return False
        
        # Additional boost criteria (need at least 2)
        boost_count = 0
        if volume_spike >= C['min_volume_spike']:
            boost_count += 1
        if liquidity_usd <= C['max_liquidity']:
            boost_count += 1
        if market_cap <= C['max_market_cap']:
            boost_count += 1
        if token_age_hours <= C['max_token_age_hours']:
            boost_count += 1
        if buy_pressure >= 60:
            boost_count += 1
        
        return boost_count >= 2
    
    def _get_alpha_rating(self, alpha_score: float) -> AlphaRating:
        """Get alpha tier rating"""
        if alpha_score >= 90:
            return AlphaRating.S_TIER
        elif alpha_score >= 80:
            return AlphaRating.A_TIER
        elif alpha_score >= 70:
            return AlphaRating.B_TIER
        elif alpha_score >= 60:
            return AlphaRating.C_TIER
        return AlphaRating.D_TIER
    
    def analyze_batch(self, tokens_df: pd.DataFrame) -> List[SNAResult]:
        print(f"{Fore.CYAN}[SNA-V3] Analyzing {len(tokens_df)} tokens with Alpha Hunter...{Style.RESET_ALL}")
        results = [self.analyze_token(row.to_dict()) for _, row in tokens_df.iterrows()]
        # Sort by alpha_score instead of sna_score
        results.sort(key=lambda x: x.alpha_score, reverse=True)
        pumps = [r for r in results if r.is_potential_pump]
        print(f"{Fore.GREEN}[SNA-V3] Found {len(pumps)} ALPHA candidates (70%+ confidence){Style.RESET_ALL}")
        return results
    
    def filter_potential_pumps(self, results: List[SNAResult], min_score: float = 70) -> List[SNAResult]:
        return [r for r in results if r.is_potential_pump and r.alpha_score >= min_score]
    
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
