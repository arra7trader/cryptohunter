
import asyncio
import sys
import os
import io
from datetime import datetime, timedelta

# Force UTF-8 for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.backtester import Backtester

def test_enhanced_backtest():
    print("\n=== ENHANCED BACKTEST TEST ===")
    
    backtester = Backtester(initial_capital=10000)
    
    # Test backtest (last 7 days to be quick)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    print(f"Running backtest from {start_date.date()} to {end_date.date()}")
    
    try:
        result = backtester.run_backtest(
            symbol="BTC",
            model_name="enhanced",
            start_date=start_date,
            end_date=end_date
        )
        
        if result:
            print("\n✅ Backtest Successful!")
            print(f"Total Return: {result.total_return_pct:.2f}%")
            print(f"Win Rate: {result.win_rate:.2f}%")
            print(f"Trades: {result.total_trades}")
            
            # Save results
            output_file = "data/backtest_results/btc_enhanced_test.json"
            backtester.save_results(result, output_file)
            print(f"Saved to {output_file}")
        else:
            print("\n❌ Backtest returned no result (possibly no data or insufficient history)")
            
    except Exception as e:
        print(f"\n❌ Prediction Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_backtest()
