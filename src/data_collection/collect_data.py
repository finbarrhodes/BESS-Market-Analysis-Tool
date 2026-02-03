"""
Main data collection script.

This script orchestrates the collection of all market data from both
National Grid ESO and Elexon BMRS APIs.

Usage:
    python src/data_collection/collect_data.py --start 2024-01-01 --end 2024-01-07
    python src/data_collection/collect_data.py --config config.yaml
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_collection import NESOCollector, ElexonBMRSCollector
from src.utils import load_config, setup_logging
from loguru import logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Collect GB BESS market data from various sources'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD)',
        default=None
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file',
        default='config.yaml'
    )
    
    parser.add_argument(
        '--sources',
        type=str,
        nargs='+',
        choices=['neso', 'elexon', 'all'],
        default=['all'],
        help='Data sources to collect from'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save data to files (useful for testing)'
    )
    
    return parser.parse_args()


def collect_neso_data(start_date: str, end_date: str, save: bool = True):
    """
    Collect data from National Grid ESO.
    
    Args:
        start_date: Start date string
        end_date: End date string
        save: Whether to save data
    """
    logger.info("=" * 80)
    logger.info("COLLECTING NATIONAL GRID ESO DATA")
    logger.info("=" * 80)
    
    try:
        collector = NESOCollector()
        data = collector.collect_all_markets(start_date, end_date, save=save)

        # Print summary
        logger.info("\nNESO Collection Summary:")
        for market, df in data.items():
            logger.info(f"  {market}: {len(df)} records")

        return data

    except Exception as e:
        logger.error(f"Failed to collect NESO data: {e}")
        return {}


def collect_elexon_data(start_date: str, end_date: str, save: bool = True):
    """
    Collect data from Elexon BMRS.
    
    Args:
        start_date: Start date string
        end_date: End date string
        save: Whether to save data
    """
    logger.info("=" * 80)
    logger.info("COLLECTING ELEXON BMRS DATA")
    logger.info("=" * 80)
    
    try:
        collector = ElexonBMRSCollector()
        data = collector.collect_all_markets(start_date, end_date, save=save)
        
        # Print summary
        logger.info("\nElexon BMRS Collection Summary:")
        for market, df in data.items():
            logger.info(f"  {market}: {len(df)} records")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to collect Elexon BMRS data: {e}")

        return {}


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    setup_logging(config)
    
    # Get date range
    if args.start and args.end:
        start_date = args.start
        end_date = args.end
    else:
        # Use dates from config
        data_config = config['data_collection']
        start_date = data_config['start_date']
        end_date = data_config['end_date']
    
    logger.info(f"Collecting data from {start_date} to {end_date}")
    
    save_data = not args.no_save
    
    # Collect data based on sources
    all_data = {}
    
    if 'all' in args.sources or 'neso' in args.sources:
        neso_data = collect_neso_data(start_date, end_date, save=save_data)
        all_data['neso'] = neso_data
    
    if 'all' in args.sources or 'elexon' in args.sources:
        elexon_data = collect_elexon_data(start_date, end_date, save=save_data)
        all_data['elexon'] = elexon_data
    
    # Final summary
    logger.info("=" * 80)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 80)
    
    total_records = 0
    for source, markets in all_data.items():
        logger.info(f"\n{source.upper()}:")
        for market, df in markets.items():
            records = len(df)
            total_records += records
            logger.info(f"  {market}: {records:,} records")
    
    logger.info(f"\nTotal records collected: {total_records:,}")
    
    if save_data:
        logger.info(f"\nData saved to: data/raw/")
    
    return all_data


if __name__ == "__main__":
    main()
