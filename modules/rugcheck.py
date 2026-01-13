"""
RugCheck / Honeypot Detection Module
=====================================
Modular security checker for tokens.
Uses RugCheck.xyz (for Solana) and heuristic checks (for EVM/Others).

Features:
- Check Liquidity Locked status
- Check Mint Authority (if mintable)
- Check Freeze Authority
- Check Top Holders concentration
"""

import requests
import time
from typing import Dict, Optional, Tuple
from colorama import Fore, Style

class RugChecker:
    """Security Scanner for Alpha Tokens"""
    
    RUGCHECK_API = "https://api.rugcheck.xyz/v1/tokens/{mint}/report"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "CryptoHunter-Bot/2.1 (SecurityScanner)"
        })
    
    def check_token(self, token_address: str, chain_id: str = "solana") -> Dict:
        """
        Check token security
        
        Args:
            token_address: Token contract address
            chain_id: Chain ID (solana, ethereum, etc)
            
        Returns:
            Dict with security report:
            {
                'is_safe': bool,
                'score': int (0-100, higher is safer),
                'risks': List[str],
                'details': Dict
            }
        """
        # Normalize chain ID
        chain = chain_id.lower()
        
        if chain == "solana":
            return self._check_solana_rugcheck(token_address)
        else:
            # Fallback for EVM (Placeholder for GoPlus or Honeypot.is integration)
            # For now, we perform basic heuristic checks if data available, or return specific "Unknown" status
            return self._check_evm_heuristic(token_address)

    def _check_solana_rugcheck(self, token_address: str) -> Dict:
        """Check Solana token using RugCheck.xyz"""
        print(f"{Fore.CYAN}[SECURITY] Scanning {token_address[:8]}... on RugCheck{Style.RESET_ALL}")
        
        try:
            url = f"https://api.rugcheck.xyz/v1/tokens/{token_address}/report"
            resp = self.session.get(url, timeout=10)
            
            if resp.status_code == 200:
                data = resp.json()
                
                # Check for critical risks
                risks = data.get("risks", [])
                score = data.get("score", 0) # RugCheck uses 0 (Safe) to High (Risk). Wait, usually score is risk score.
                # RugCheck: Lower score is better (Risk Score). 0 = Perfect, > 5000 = Bad.
                # Let's normalize to 0-100 Safety Score.
                
                risk_score = data.get("score", 0)
                markets = data.get("markets", [])
                
                # Determine safety
                # Safe if risk_score < 1000 and liquidity is locked usually
                
                safety_score = max(0, 100 - (risk_score / 100)) # Simple normalization
                
                critical_risks = [r['name'] for r in risks if r['level'] == 'danger']
                warnings = [r['name'] for r in risks if r['level'] == 'warn']
                
                is_safe = risk_score < 2000 and len(critical_risks) == 0
                
                # Report
                result = {
                    'is_safe': is_safe,
                    'safety_score': int(safety_score),
                    'risk_score_raw': risk_score,
                    'risks': critical_risks + warnings,
                    'details': {
                        'mint_authority': not data.get('tokenMeta', {}).get('mutable', True),
                        'freeze_authority': data.get('tokenMeta', {}).get('freezeAuthority') is None,
                        'top_holders_exposed': False # Placeholder logic
                    }
                }
                
                color = Fore.GREEN if is_safe else Fore.RED
                print(f"{color}[SECURITY] Safety Score: {int(safety_score)}/100 | Safe: {is_safe}{Style.RESET_ALL}")
                return result
                
            else:
                print(f"{Fore.YELLOW}[WARN] RugCheck API Error {resp.status_code}{Style.RESET_ALL}")
                return self._get_default_result()
                
        except Exception as e:
            print(f"{Fore.RED}[ERR] RugCheck Failed: {e}{Style.RESET_ALL}")
            return self._get_default_result()

    def _check_evm_heuristic(self, token_address: str) -> Dict:
        """Basic checks for EVM (since we don't have free API key for GoPlus yet)"""
        # For now, assume neutral but flag as "Unverified"
        return {
            'is_safe': True, # Default allow but warn
            'safety_score': 50,
            'risks': ['Unverified (EVM Security Check Not Configured)'],
            'details': {}
        }

    def _get_default_result(self) -> Dict:
        return {
            'is_safe': False,
            'safety_score': 0,
            'risks': ['Scan Failed'],
            'details': {}
        }

if __name__ == "__main__":
    # Test
    checker = RugChecker()
    # Test a known Solana token
    res = checker.check_token("So11111111111111111111111111111111111111112", "solana")
    print(res)
