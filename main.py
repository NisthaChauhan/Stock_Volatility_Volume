'''
"""
Main Runner - Stock Volatility vs Volume Analysis
Orchestrates all modules for complete analysis pipeline
"""

import os
import sys
from ticker_validator import TickerValidator
from regression_models import ModelFitter
from visualization import ModelVisualizer
from model_selector import ModelSelector


class StockVolatilityAnalyzer:
    """Main class to run complete analysis pipeline"""
    
    def __init__(self):
        self.validator=TickerValidator()
        self.fitter=ModelFitter()
        self.selector=ModelSelector()
        self.visualizer=None
        
        self.ticker=None
        self.X=None
        self.y=None
        self.results=None
    
    def run_analysis(self, ticker_symbol, years=5, create_plots=True, 
                     output_dir='results', metric='rmse'):
        """
        Runs complete analysis pipeline
        
        Args:
            ticker_symbol (str): Stock ticker (e.g., 'TSLA', 'AAPL')
            years (int): Years of historical data
            create_plots (bool): Whether to create visualization plots
            output_dir (str): Directory for output files
            metric (str): Primary metric for model selection ('rmse', 'mae', 'r2')
        """
        print("\n" + "="*80)
        print("STOCK VOLATILITY vs VOLUME ANALYSIS")
        print("="*80)
        
        # Step 1: Validate and fetch data
        print("\n[STEP 1/4] Validating ticker and fetching data...")
        self.X, self.y=self.validator.get_full_pipeline(ticker_symbol, years)
        
        if self.X is None or self.y is None:
            print("\nAnalysis failed.")
            return False
        
        self.ticker=self.validator.ticker
        print(f"Analyzing {len(self.X)} data points for {self.ticker}")
        
        # Step 2: Fit all models
        print("\n[STEP 2/4] Fitting regression models...")
        self.results=self.fitter.fit_all_models(self.X, self.y)
        
        print("\n[STEP 3/4] Analyzing model performance...")
        self.selector.print_rankings(self.results, self.X, metric=metric)
        print(self.selector.get_summary(self.results, self.X))

        # Step 4: Create simplified visualizations (only fitting equations)
        if create_plots:
            print("\n[STEP 4/4] Creating visualizations...")
            self.visualizer=ModelVisualizer(self.ticker)
            # Show only the comparison grid (fitting equations in subplots)
            self.visualizer.plot_comparison_grid(self.X, self.y, self.results)
            
        # Export results to CSV
        csv_path=os.path.join(output_dir, f'{self.ticker}_results.csv')
        self.selector.export_results(self.results, self.X, self.ticker, csv_path)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80 + "\n")
        
        return True

    def get_best_model_details(self):
        """Returns details of the best performing model"""
        if self.results is None:
            return None
        
        best_name, best_result=self.fitter.get_best_model(self.results)
        return {
            'ticker': self.ticker,
            'model': best_name,
            'rmse': best_result['rmse'],
            'r2': best_result['r2'],
            'mae': best_result['mae'],
            'params': best_result['params']
        }


def interactive_mode():
    """Runs the analyzer in simplified interactive mode"""
    print("\n" + "="*80)
    print("🎯 STOCK VOLATILITY ANALYZER")
    print("="*80)
    
    analyzer=StockVolatilityAnalyzer()
    
    while True:
        # Get ticker from user
        ticker=input("\nEnter stock ticker (e.g., TSLA, AAPL) or 'q' to quit: ").strip()
        
        if ticker.lower() == 'q':
            print("\n👋 Thank you for using Stock Volatility Analyzer!\n")
            break
        
        if not ticker:
            print("⚠️  Please enter a valid ticker symbol.")
            continue

        # Validate ticker first
        print(f"\n🔍 Validating ticker: {ticker}")
        if not analyzer.validator.validate_ticker(ticker):
            print(f"Ticker '{ticker}' is invalid. Please try again.")
            continue
        else:
            print(f"Ticker '{ticker}' is valid!")
        
        # Get years of data
        try:
            years_input=input("\nEnter years of historical data [default: 5]: ").strip()
            years=int(years_input) if years_input else 5
        except ValueError:
            print("⚠️  Invalid input. Using default 5 years.")
            years=5
        
        # Get primary metric
        print("\nChoose primary metric for model selection:")
        print("  1. RMSE (Root Mean Squared Error) - default")
        print("  2. MAE (Mean Absolute Error)")
        print("  3. R² (Coefficient of Determination)")
        metric_input=input("Enter choice (1/2/3) [default: 1]: ").strip()
        
        metric_map={'1': 'rmse', '2': 'mae', '3': 'r2'}
        metric=metric_map.get(metric_input, 'rmse')
        
        # Run analysis
        success=analyzer.run_analysis(ticker, years=years, 
                                       create_plots=True,
                                       metric=metric)
        
        if success:
            # Show best model details
            best=analyzer.get_best_model_details()
            if best:
                print(f"\nBEST MODEL EQUATION:")
                print(f"   Model: {best['model']}")
                print(f"   Parameters: {[f'{p:.6e}' for p in best['params']]}")
        
        # Ask if continue
        continue_input=input("\nAnalyze another stock? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("\n👋 Thank you for using Stock Volatility Analyzer!\n")
            break


def main():
    """Main entry point"""
    import argparse
    
    parser=argparse.ArgumentParser(
        description='Stock Volatility vs Volume Analysis Tool'
    )
    parser.add_argument(
        '-t', '--ticker',
        type=str,
        help='Single ticker to analyze (e.g., TSLA)'
    )
    parser.add_argument(
        '-b', '--batch',
        nargs='+',
        help='Multiple tickers for batch analysis (e.g., TSLA AAPL MSFT)'
    )
    parser.add_argument(
        '-y', '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )
    parser.add_argument(
        '-m', '--metric',
        choices=['rmse', 'mae', 'r2'],
        default='rmse',
        help='Primary metric for model selection (default: rmse)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    args=parser.parse_args()
    
    if args.batch:
        # Batch mode
        print("Batch mode not implemented in this version.")
    elif args.ticker:
        # Single ticker mode
        analyzer=StockVolatilityAnalyzer()
        analyzer.run_analysis(args.ticker, years=args.years,
                            output_dir=args.output, metric=args.metric)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()
'''
"""
Main Runner - Stock Volatility vs Volume Analysis
Orchestrates all modules for complete analysis pipeline
"""

import os
import sys
from ticker_validator import TickerValidator
from regression_models import ModelFitter
from visualization import ModelVisualizer
from model_selector import ModelSelector


class StockVolatilityAnalyzer:
    """Main class to run complete analysis pipeline"""
    
    def __init__(self):
        self.validator=TickerValidator()
        self.fitter=ModelFitter()
        self.selector=ModelSelector()
        self.visualizer=None
        
        self.ticker=None
        self.X=None
        self.y=None
        self.results=None
    
    def run_analysis(self, ticker_symbol, years=5, create_plots=True, 
                     output_dir='results', metric='rmse'):
        """
        Runs complete analysis pipeline
        
        Args:
            ticker_symbol (str): Stock ticker (e.g., 'TSLA', 'AAPL')
            years (int): Years of historical data
            create_plots (bool): Whether to create visualization plots
            output_dir (str): Directory for output files
            metric (str): Primary metric for model selection ('rmse', 'mae', 'r2')
        """
        print("\n" + "="*80)
        print("STOCK VOLATILITY vs VOLUME ANALYSIS")
        print("="*80)
        
        # Step 1: Validate and fetch data
        print("\n[STEP 1/4] Validating ticker and fetching data...")
        self.X, self.y=self.validator.get_full_pipeline(ticker_symbol, years)
        
        if self.X is None or self.y is None:
            print("\nAnalysis failed.")
            return False
        
        self.ticker=self.validator.ticker
        print(f"Analyzing {len(self.X)} data points for {self.ticker}")
        
        # Step 2: Fit all models
        print("\n[STEP 2/4] Fitting regression models...")
        self.results=self.fitter.fit_all_models(self.X, self.y)
        
        print("\n[STEP 3/4] Analyzing model performance...")
        self.selector.print_rankings(self.results, self.X, metric=metric)
        print(self.selector.get_summary(self.results, self.X))

        # Step 4: Create simplified visualizations (only fitting equations)
        if create_plots:
            print("\n[STEP 4/4] Creating visualizations...")
            self.visualizer=ModelVisualizer(self.ticker)
            # Show only the comparison grid (fitting equations in subplots)
            self.visualizer.plot_comparison_grid(self.X, self.y, self.results)
            
        # Export results to CSV
        csv_path=os.path.join(output_dir, f'{self.ticker}_results.csv')
        self.selector.export_results(self.results, self.X, self.ticker, csv_path)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80 + "\n")
        
        return True

    def get_best_model_details(self):
        """Returns details of the best performing model"""
        if self.results is None:
            return None
        
        best_name, best_result=self.fitter.get_best_model(self.results)
        return {
            'ticker': self.ticker,
            'model': best_name,
            'rmse': best_result['rmse'],
            'r2': best_result['r2'],
            'mae': best_result['mae'],
            'params': best_result['params']
        }


def interactive_mode():
    """Runs the analyzer in simplified interactive mode"""
    print("\n" + "="*80)
    print("STOCK VOLATILITY ANALYZER")
    print("="*80)
    
    analyzer=StockVolatilityAnalyzer()
    years=5
    metric='rmse'
    
    while True:
        ticker=input("\nEnter stock ticker (e.g., TSLA, AAPL) or 'q' to quit: ").strip()
        
        if ticker.lower() == 'q':
            print("\nThank you for using Stock Volatility Analyzer!\n")
            break
        
        if not ticker:
            print("Please enter a valid ticker symbol.")
            continue

        # Validate ticker first
        print(f"\nValidating ticker: {ticker}")
        if not analyzer.validator.validate_ticker(ticker):
            print(f"Ticker '{ticker}' is invalid. Please try again.")
            continue
        else:
            print(f"Ticker '{ticker}' is valid!")
        
        # Run analysis with static defaults (5 years, RMSE metric)
        success=analyzer.run_analysis(ticker, years=years, 
                                       create_plots=True,
                                       metric=metric)
        
        if success:
            # Show best model details
            best=analyzer.get_best_model_details()
            if best:
                print(f"\nBEST MODEL EQUATION:")
                print(f"   Model: {best['model']}")
                print(f"   Parameters: {[f'{p:.6e}' for p in best['params']]}")
        
        # Ask if continue
        continue_input=input("\nAnalyze another stock? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("\nThank you for using Stock Volatility Analyzer!\n")
            break


def main():
    """Main entry point"""
    import argparse
    
    parser=argparse.ArgumentParser(
        description='Stock Volatility vs Volume Analysis Tool'
    )
    parser.add_argument(
        '-t', '--ticker',
        type=str,
        help='Single ticker to analyze (e.g., TSLA)'
    )
    parser.add_argument(
        '-b', '--batch',
        nargs='+',
        help='Multiple tickers for batch analysis (e.g., TSLA AAPL MSFT)'
    )
    parser.add_argument(
        '-y', '--years',
        type=int,
        default=5,
        help='Years of historical data (default: 5)'
    )
    parser.add_argument(
        '-m', '--metric',
        choices=['rmse', 'mae', 'r2'],
        default='rmse',
        help='Primary metric for model selection (default: rmse)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='results',
        help='Output directory (default: results)'
    )
    
    args=parser.parse_args()
    
    if args.batch:
        # Batch mode
        print("Batch mode not implemented in this version.")
    elif args.ticker:
        # Single ticker mode
        analyzer=StockVolatilityAnalyzer()
        analyzer.run_analysis(args.ticker, years=args.years,
                            output_dir=args.output, metric=args.metric)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()