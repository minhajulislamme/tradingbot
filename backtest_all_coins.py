#!/usr/bin/env python3
"""
Backtest script for LAYER and other cryptocurrencies
"""
import sys
import os
import logging
import argparse
from datetime import datetime

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the backtest module
from main import run_backtest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('backtest_all_coins')

def main():
    parser = argparse.ArgumentParser(description='Run backtests for multiple cryptocurrencies')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--coins', nargs='+', default=None, help='Specific coins to backtest (e.g., LAYER BTC ETH)')
    parser.add_argument('--timeframes', nargs='+', default=['15m'], help='Timeframes to test (e.g., 5m 15m 1h)')
    parser.add_argument('--strategy', type=str, default='LayerDynamicGrid', help='Strategy to use for backtest')
    args = parser.parse_args()
    
    # Define the coins to test with LAYER as the default focus
    all_coins = ['LAYERUSDT', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'SHIBUSDT']
    
    # If specific coins are provided, filter the list
    if args.coins:
        test_coins = []
        for coin in args.coins:
            coin_upper = coin.upper()
            if not coin_upper.endswith('USDT'):
                coin_upper += 'USDT'
            if coin_upper in all_coins or coin_upper == 'LAYERUSDT':
                test_coins.append(coin_upper)
            else:
                logger.warning(f"Coin not found: {coin_upper}")
        coins_to_test = test_coins
    else:
        # Default to just LAYER if no coins specified
        coins_to_test = ['LAYERUSDT']
    
    # Dictionary to store backtest results
    results = {}
    
    # Calculate start date
    start_date = f"{args.days} days ago"
    
    # Run backtest for each coin
    logger.info(f"Starting backtest for {len(coins_to_test)} coins over {args.days} days using {args.strategy} strategy")
    for symbol in coins_to_test:
        logger.info(f"Backtesting {symbol}")
        
        coin_results = {}
        
        # Select appropriate strategy for the coin
        strategy_name = args.strategy
        # Default to LayerDynamicGrid for LAYER
        if symbol == 'LAYERUSDT':
            strategy_name = 'LayerDynamicGrid'
        
        # Run backtest for each timeframe
        for timeframe in args.timeframes:
            logger.info(f"Running {symbol} on {timeframe} timeframe with {strategy_name} strategy")
            result = run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                strategy_name=strategy_name,
                save_results=True
            )
            
            if result:
                coin_results[timeframe] = {
                    'total_return': result.get('total_return', 0),
                    'win_rate': result.get('win_rate', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'total_trades': result.get('total_trades', 0)
                }
                
                logger.info(f"{symbol} {timeframe} backtest: Return: {result.get('total_return', 0):.2f}%, "
                           f"Win Rate: {result.get('win_rate', 0):.2f}%, "
                           f"Max DD: {result.get('max_drawdown', 0):.2f}%, "
                           f"Sharpe: {result.get('sharpe_ratio', 0):.2f}")
            else:
                logger.error(f"Backtest failed for {symbol} on {timeframe}")
        
        results[symbol] = coin_results
    
    # Print summary of results
    logger.info("\n\nBacktest Results Summary:")
    logger.info("=" * 80)
    logger.info(f"{'Symbol':<10} {'Timeframe':<10} {'Return%':<10} {'Win Rate':<10} {'Max DD%':<10} {'Sharpe':<10} {'Trades':<10}")
    logger.info("=" * 80)
    
    for symbol in results:
        for timeframe, metrics in results[symbol].items():
            logger.info(f"{symbol:<10} {timeframe:<10} "
                      f"{metrics['total_return']:<10.2f} "
                      f"{metrics['win_rate']:<10.2f} "
                      f"{metrics['max_drawdown']:<10.2f} "
                      f"{metrics['sharpe_ratio']:<10.2f} "
                      f"{metrics['total_trades']:<10}")
    
    logger.info("\nBacktesting completed!")

if __name__ == "__main__":
    main()