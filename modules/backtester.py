"""
CryptoHunter Backtesting Framework V1.0
========================================
Systematic backtesting engine to validate AI predictions against historical data.

Features:
- Historical data replay
- Walk-forward validation
- Performance metrics (Sharpe, win rate, max drawdown)
- Model comparison (Prophet vs LSTM vs Ensemble)
- P&L tracking
- Visualization support

Author: CryptoHunter Team
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os


class TradeAction(Enum):
    """Trade action types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Trade:
    """Single trade record"""
    timestamp: datetime
    action: TradeAction
    price: float
    amount: float
    predicted_price: float
    actual_price: float
    profit_loss: float
    profit_loss_pct: float


@dataclass
class BacktestResult:
    """Backtest results summary"""
    symbol: str
    model_name: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_profit_per_trade: float
    avg_loss_per_trade: float
    profit_factor: float
    directional_accuracy: float
    trades: List[Trade]


class Backtester:
    """
    Backtesting engine for validating predictions
    """
    
    def __init__(
        self,
        initial_capital: float = 10000,
        position_size_pct: float = 0.1,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in USD
            position_size_pct: Position size as % of capital (0.1 = 10%)
            commission: Trading commission (0.001 = 0.1%)
            slippage: Price slippage (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage = slippage
        
        # State
        self.capital = initial_capital
        self.position = 0  # Current position size
        self.entry_price = 0
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        print(f"[BACKTESTER] Initialized with ${initial_capital:,.2f} capital")
    
    def fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Fetch historical price data
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC', 'ETH')
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Try Binance first (will implement next)
            # For now, use aggregator
            try:
                from modules.crypto_data_aggregator import get_aggregated_data
            except ImportError:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from modules.crypto_data_aggregator import get_aggregated_data
            
            print(f"[BACKTESTER] Fetching {symbol} data from {start_date} to {end_date}")
            
            # Get data
            aggregated = get_aggregated_data(symbol, interval='1h')
            
            if not aggregated or aggregated.total_candles < 100:
                print(f"[BACKTESTER] Insufficient data for {symbol}")
                return pd.DataFrame()
            
            df = aggregated.ohlcv.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by date range
            mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
            filtered_df = df.loc[mask].copy()
            
            print(f"[BACKTESTER] Loaded {len(filtered_df)} candles for {symbol}")
            return filtered_df
            
        except Exception as e:
            print(f"[BACKTESTER] Error fetching data: {e}")
            return pd.DataFrame()
    
    def generate_predictions(
        self,
        df: pd.DataFrame,
        model_name: str = "ensemble"
    ) -> List[Dict]:
        """
        Generate predictions for historical data
        
        Args:
            df: Historical OHLCV data
            model_name: Model to use ('prophet', 'lstm', 'ensemble')
        
        Returns:
            List of predictions
        """
        predictions = []
        
        try:
            from modules.timegpt_prophet_forecaster import UnifiedForecaster
            forecaster = UnifiedForecaster()
            
            # Walk forward through history (every 24 hours)
            for i in range(100, len(df), 24):  # Start at 100 for enough history
                # Get data up to current point
                historical_data = df.iloc[:i].copy()
                
                # Get actual future price (24h ahead)
                if i + 24 < len(df):
                    actual_price_24h = df.iloc[i + 24]['close']
                else:
                    break
                
                current_price = historical_data.iloc[-1]['close']
                current_time = historical_data.iloc[-1]['timestamp']
                
                # Generate prediction
                if model_name == "prophet":
                    result = forecaster.forecast_with_prophet(historical_data)
                elif model_name == "lstm":
                    result = forecaster.forecast_with_lstm(historical_data)
                elif model_name == "enhanced":
                    # Get base ensemble prediction
                    base_result = forecaster.forecast_ensemble(historical_data)
                    if base_result:
                        # Fetch historical F&G if not already done
                        if not hasattr(self, 'fg_history'):
                             from modules.fear_greed import FearGreedIndex
                             fg_api = FearGreedIndex()
                             # Get enough history to cover backtest period
                             days_needed = (df.iloc[-1]['timestamp'] - df.iloc[0]['timestamp']).days + 10
                             self.fg_history = fg_api.get_historical(days=max(30, days_needed))
                             print(f"[BACKTESTER] Fetched {len(self.fg_history)} days of Fear & Greed history")

                        # Find matching F&G value
                        current_date = current_time.date()
                        # F&G data is daily, finding the record for this date
                        # Assuming fg_history is a DataFrame with 'timestamp' and 'value'
                        # timestamp in fg_history is usually 00:00:00 of that day
                        
                        # Simple lookup
                        fg_val = 50 # Default neutral
                        try:
                            # Filter for same day
                            day_match = self.fg_history[
                                self.fg_history['timestamp'].dt.date == current_date
                            ]
                            if not day_match.empty:
                                fg_val = day_match.iloc[0]['value']
                        except Exception:
                            pass
                            
                        # Apply adjustment
                        from modules.fear_greed import FearGreedIndex
                        fg_api = FearGreedIndex()
                        adjusted = fg_api.adjust_prediction(
                            base_result.signal.value, 
                            confidence=base_result.confidence / 100
                        )
                        
                        # Create pseudo-result with adjusted values
                        result = base_result
                        result.signal = result.signal.__class__(adjusted['adjusted_signal']) # Enum conversion might be needed if strictly typed, but let's assume string or compatible
                        result.confidence = adjusted['confidence']
                        
                    else:
                        result = None

                else:  # ensemble
                    result = forecaster.forecast_ensemble(historical_data)
                
                if result:
                    predictions.append({
                        'timestamp': current_time,
                        'current_price': current_price,
                        'predicted_price_24h': result.predicted_price_24h,
                        'actual_price_24h': actual_price_24h,
                        'predicted_change': result.change_24h_pct,
                        'actual_change': ((actual_price_24h - current_price) / current_price) * 100,
                        'signal': str(result.signal.value if hasattr(result.signal, 'value') else result.signal),
                        'confidence': result.confidence
                    })
            
            print(f"[BACKTESTER] Total predictions: {len(predictions)}")
            return predictions
            
        except Exception as e:
            print(f"[BACKTESTER] Error generating predictions: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def simulate_trading(
        self,
        predictions: List[Dict]
    ) -> List[Trade]:
        """
        Simulate trading based on predictions
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            List of executed trades
        """
        trades = []
        
        for pred in predictions:
            current_price = pred['current_price']
            predicted_change = pred['predicted_change']
            actual_price_24h = pred['actual_price_24h']
            
            # Trading logic: Buy if predicted up >2%, Sell if predicted down >2%
            if predicted_change > 2 and self.position == 0:
                # BUY
                action = TradeAction.BUY
                entry_price = current_price * (1 + self.slippage)  # Slippage on entry
                amount = (self.capital * self.position_size_pct) / entry_price
                cost = amount * entry_price * (1 + self.commission)
                
                self.position = amount
                self.entry_price = entry_price
                self.capital -= cost
                
                # Calculate P&L (realized at actual future price)
                exit_price = actual_price_24h * (1 - self.slippage)
                exit_value = amount * exit_price * (1 - self.commission)
                profit_loss = exit_value - cost
                profit_loss_pct = (profit_loss / cost) * 100
                
                # Close position
                self.capital += exit_value
                self.position = 0
                
                trade = Trade(
                    timestamp=pred['timestamp'],
                    action=action,
                    price=entry_price,
                    amount=amount,
                    predicted_price=pred['predicted_price_24h'],
                    actual_price=actual_price_24h,
                    profit_loss=profit_loss,
                    profit_loss_pct=profit_loss_pct
                )
                trades.append(trade)
                
            elif predicted_change < -2 and self.position == 0:
                # SHORT (or skip if no shorting)
                # For simplicity, we'll skip short trades in this version
                pass
            
            # Track equity
            total_equity = self.capital + (self.position * current_price if self.position > 0 else 0)
            self.equity_curve.append({
                'timestamp': pred['timestamp'],
                'equity': total_equity
            })
        
        print(f"[BACKTESTER] Executed {len(trades)} trades")
        return trades
    
    def calculate_metrics(self, trades: List[Trade]) -> Dict:
        """
        Calculate performance metrics
        
        Args:
            trades: List of executed trades
        
        Returns:
            Dictionary of metrics
        """
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'total_return_pct': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'avg_profit_per_trade': 0.0,
                'avg_loss_per_trade': 0.0,
                'profit_factor': 0.0,
                'directional_accuracy': 0.0,
                'final_capital': round(self.capital, 2)
            }
        
        # Basic stats
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.profit_loss > 0]
        losing_trades = [t for t in trades if t.profit_loss < 0]
        
        win_rate = (len(winning_trades) / total_trades) * 100 if total_trades > 0 else 0
        
        # Returns
        total_return = sum(t.profit_loss for t in trades)
        total_return_pct = ((self.capital - self.initial_capital) / self.initial_capital) * 100
        
        # Average profits/losses
        avg_profit = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.profit_loss for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        total_profits = sum(t.profit_loss for t in winning_trades)
        total_losses = abs(sum(t.profit_loss for t in losing_trades))
        profit_factor = total_profits / total_losses if total_losses > 0 else 0
        
        # Sharpe Ratio (simplified)
        returns = [t.profit_loss_pct for t in trades]
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sortino Ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        downside_std = np.std(downside_returns) if downside_returns else 1
        sortino_ratio = (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Max Drawdown
        equity_values = [e['equity'] for e in self.equity_curve]
        if equity_values:
            peak = equity_values[0]
            max_dd = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                dd = peak - equity
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = (max_dd / peak) * 100 if peak > 0 else 0
        else:
            max_dd = 0
            max_dd_pct = 0
        
        # Directional Accuracy
        correct_direction = sum(
            1 for t in trades
            if (t.actual_price > t.price and t.action == TradeAction.BUY) or
               (t.actual_price < t.price and t.action == TradeAction.SELL)
        )
        directional_accuracy = (correct_direction / total_trades) * 100 if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return, 2),
            'total_return_pct': round(total_return_pct, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'sortino_ratio': round(sortino_ratio, 2),
            'max_drawdown': round(max_dd, 2),
            'max_drawdown_pct': round(max_dd_pct, 2),
            'avg_profit_per_trade': round(avg_profit, 2),
            'avg_loss_per_trade': round(avg_loss, 2),
            'profit_factor': round(profit_factor, 2),
            'directional_accuracy': round(directional_accuracy, 2),
            'final_capital': round(self.capital, 2)
        }
    
    def run_backtest(
        self,
        symbol: str,
        model_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """
        Run complete backtest
        
        Args:
            symbol: Crypto symbol
            model_name: Model to test
            start_date: Backtest start date
            end_date: Backtest end date
        
        Returns:
            BacktestResult object
        """
        print(f"\n[BACKTESTER] ========== Starting Backtest ==========")
        print(f"Symbol: {symbol}")
        print(f"Model: {model_name}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []
        
        # Fetch data
        df = self.fetch_historical_data(symbol, start_date, end_date)
        if df.empty:
            print("[BACKTESTER] No data available")
            return None
        
        # Generate predictions
        predictions = self.generate_predictions(df, model_name)
        if not predictions:
            print("[BACKTESTER] No predictions generated")
            return None
        
        # Simulate trading
        trades = self.simulate_trading(predictions)
        
        # Calculate metrics
        metrics = self.calculate_metrics(trades)
        
        # Print summary
        print(f"\n[BACKTESTER] ========== Results ==========")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        # Create result object
        result = BacktestResult(
            symbol=symbol,
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            **metrics,
            trades=trades
        )
        
        return result
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        result_dict = {
            'symbol': result.symbol,
            'model_name': result.model_name,
            'start_date': result.start_date.isoformat(),
            'end_date': result.end_date.isoformat(),
            'metrics': {
                'total_trades': result.total_trades,
                'win_rate': result.win_rate,
                'total_return_pct': result.total_return_pct,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown_pct': result.max_drawdown_pct,
                'directional_accuracy': result.directional_accuracy
            },
            'trades': [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'action': t.action.value,
                    'price': t.price,
                    'profit_loss_pct': t.profit_loss_pct
                }
                for t in result.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"[BACKTESTER] Results saved to {filepath}")


# Test
if __name__ == "__main__":
    print("\\n=== BACKTESTER TEST ===\\n")
    
    backtester = Backtester(initial_capital=10000)
    
    # Test backtest (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    result = backtester.run_backtest(
        symbol="BTC",
        model_name="prophet",
        start_date=start_date,
        end_date=end_date
    )
    
    if result:
        # Save results
        backtester.save_results(result, "data/backtest_results/btc_prophet_30d.json")
        print("\\nTest complete!")
