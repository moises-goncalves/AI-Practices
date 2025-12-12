"""
Stock data download module using Yahoo Finance API

Usage:
    python download_data.py --ticker AAPL --start 2015-01-01 --end 2023-12-31
    python download_data.py --ticker GOOGL --period 5y
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_yfinance():
    """Check and install yfinance if needed."""
    try:
        import yfinance as yf
        return yf
    except ImportError:
        logger.error("yfinance not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'yfinance'])
        import yfinance as yf
        return yf


def download_stock_data(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    period: str = '5y',
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Download stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
        start: Start date in 'YYYY-MM-DD' format
        end: End date in 'YYYY-MM-DD' format
        period: Period to download (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max)
        output_path: Path to save the CSV file

    Returns:
        DataFrame with OHLCV data
    """
    yf = setup_yfinance()

    logger.info(f"Downloading stock data for {ticker}")

    try:
        stock = yf.Ticker(ticker)

        if start and end:
            df = stock.history(start=start, end=end)
            logger.info(f"Date range: {start} to {end}")
        else:
            df = stock.history(period=period)
            logger.info(f"Period: {period}")

        if df.empty:
            raise ValueError(f"No data found for ticker {ticker}")

        df.reset_index(inplace=True)

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df = df[required_columns]

        logger.info(f"Downloaded {len(df)} records")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        logger.info(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Data saved to: {output_path}")

        return df

    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download stock data from Yahoo Finance')

    parser.add_argument('--ticker', type=str, default='AAPL',
                       help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--start', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--period', type=str, default='5y',
                       help='Period to download (default: 5y)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path')

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    data_dir = Path(__file__).parent

    if args.output is None:
        args.output = data_dir / f'{args.ticker}_stock_data.csv'

    print("="*60)
    print("Stock Data Download")
    print("="*60)
    print(f"Ticker: {args.ticker}")
    if args.start and args.end:
        print(f"Date range: {args.start} to {args.end}")
    else:
        print(f"Period: {args.period}")
    print(f"Output: {args.output}")
    print("="*60)

    try:
        df = download_stock_data(
            ticker=args.ticker,
            start=args.start,
            end=args.end,
            period=args.period,
            output_path=args.output
        )

        print("\n" + "="*60)
        print("Download Successful!")
        print("="*60)
        print(f"\nRecords: {len(df)}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"\nFirst 5 rows:")
        print(df.head())
        print(f"\nData statistics:")
        print(df.describe())

        print(f"\nNext steps:")
        print(f"  1. Check the data file: {args.output}")
        print(f"  2. Run training: python ../src/train.py --data_path {args.output}")

    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
