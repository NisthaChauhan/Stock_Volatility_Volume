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
        self.validator = TickerValidator()
        self.fitter = ModelFitter()
        self.selector = ModelSelector()
        self.visualizer = None
        
        self.ticker = None
        self.X = None
        self.y = None
        self.results = None
    
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
        print("📈 STOCK VOLATILITY vs VOLUME ANALYSIS")
        print("="*80)
        
        # Step 1: Validate and fetch data
        print("\n[STEP 1/4] Validating ticker and fetching data...")
        self.X, self.y = self.validator.get_full_pipeline(ticker_symbol, years)
        
        if self.X is None or self.y is None:
            print("\n❌ Analysis failed. Exiting...")
            return False
        
        self.ticker = self.validator.ticker
        
        # Step 2: Fit all models
        print("\n[STEP 2/4] Fitting regression models...")
        self.results = self.fitter.fit_all_models(self.X, self.y)
        
        print("\n[STEP 3/4] Analyzing model performance...")
        # ← FIXED: pass self.X
        self.selector.print_rankings(self.results, self.X, metric=metric)
        # ← FIXED: pass self.X
        print(self.selector.get_recommendation(self.results, self.X))

        # Step 4: Create visualizations
        '''if create_plots:
            print("\n[STEP 4/4] Creating visualizations...")
            plots_dir = os.path.join(output_dir, self.ticker)
            os.makedirs(plots_dir, exist_ok=True)   # make sure folder exists
            self.visualizer = ModelVisualizer(self.ticker)
            self.visualizer.plot_all_models(self.X, self.y, self.results, plots_dir)
            
            comparison_path = os.path.join(plots_dir, f'{self.ticker}_comparison.png')
            self.visualizer.plot_comparison_summary(self.X, self.y, self.results, 
                                                   comparison_path)
        '''
        if create_plots:
            print("\n[STEP 4/4] Displaying visualizations...")
            self.visualizer = ModelVisualizer(self.ticker)
            self.visualizer.create_full_report(self.X, self.y, self.results)
        # Export results to CSV
        csv_path = os.path.join(output_dir, f'{self.ticker}_results.csv')
        self.selector.export_results(self.results, self.X, self.ticker, csv_path)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80 + "\n")
        
        return True
    
    '''def get_best_model_details(self):
        """Returns details of the best performing model"""
        if self.results is None:
            return None
        
        best_name, best_result = self.selector.get_best_model(self.results)
        return {
            'ticker': self.ticker,
            'model': best_name,
            'rmse': best_result['rmse'],
            'r2': best_result['r2'],
            'mae': best_result['mae'],
            'params': best_result['params']
        }'''

def get_best_model_details(self):
        """Returns details of the best performing model"""
        if self.results is None:
            return None
        
        # This method already works – it uses self.results only
        best_name, best_result = self.selector.get_best_model(self.results)
        return {
            'ticker': self.ticker,
            'model': best_name,
            'rmse': best_result['rmse'],
            'r2': best_result['r2'],
            'mae': best_result['mae'],
            'params': best_result['params']
        }
def interactive_mode():
    """Runs the analyzer in interactive mode"""
    print("\n" + "="*80)
    print("🎯 WELCOME TO STOCK VOLATILITY ANALYZER")
    print("="*80)
    print("\nThis tool analyzes the relationship between stock volatility and volume.")
    print("It fits 6 different mathematical models and finds the best fit.\n")
    
    analyzer = StockVolatilityAnalyzer()
    
    while True:
        # Get ticker from user
        ticker = input("\nEnter stock ticker (e.g., TSLA, AAPL) or 'q' to quit: ").strip()
        
        if ticker.lower() == 'q':
            print("\n👋 Thank you for using Stock Volatility Analyzer!\n")
            break
        
        if not ticker:
            print("⚠️  Please enter a valid ticker symbol.")
            continue
        
        # Get years of data
        try:
            years_input = input("Years of historical data [default: 5]: ").strip()
            years = int(years_input) if years_input else 5
        except ValueError:
            print("⚠️  Invalid input. Using default 5 years.")
            years = 5
        
        # Ask about plots
        plots_input = input("Create visualization plots? (y/n) [default: y]: ").strip().lower()
        create_plots = plots_input != 'n'
        
        # Ask about metric
        print("\nChoose primary metric for model selection:")
        print("  1. RMSE (Root Mean Squared Error) - default")
        print("  2. MAE (Mean Absolute Error)")
        print("  3. R² (Coefficient of Determination)")
        metric_input = input("Enter choice (1/2/3): ").strip()
        
        metric_map = {'1': 'rmse', '2': 'mae', '3': 'r2'}
        metric = metric_map.get(metric_input, 'rmse')
        
        # Run analysis
        success = analyzer.run_analysis(ticker, years=years, 
                                       create_plots=create_plots,
                                       metric=metric)
        
        if success:
            # Ask if user wants details
            details_input = input("\nShow best model equation details? (y/n): ").strip().lower()
            if details_input == 'y':
                best = analyzer.get_best_model_details()
                if best:
                    print(f"\n{'='*80}")
                    print(f"🏆 BEST MODEL DETAILS FOR {best['ticker']}")
                    print(f"{'='*80}")
                    print(f"Model: {best['model']}")
                    print(f"RMSE:  {best['rmse']:.8f}")
                    print(f"R²:    {best['r2']:.6f}")
                    print(f"MAE:   {best['mae']:.8f}")
                    print(f"Parameters: {[f'{p:.6e}' for p in best['params']]}")
                    print(f"{'='*80}\n")
        
        # Ask if continue
        continue_input = input("\nAnalyze another stock? (y/n): ").strip().lower()
        if continue_input != 'y':
            print("\n👋 Thank you for using Stock Volatility Analyzer!\n")
            break


def batch_mode(tickers, years=5, output_dir='results', metric='rmse'):
    """
    Runs analysis for multiple tickers in batch mode
    
    Args:
        tickers (list): List of ticker symbols
        years (int): Years of historical data
        output_dir (str): Output directory
        metric (str): Primary metric for model selection
    """
    print("\n" + "="*80)
    print(f"🔄 BATCH MODE: Analyzing {len(tickers)} stocks")
    print("="*80 + "\n")
    
    analyzer = StockVolatilityAnalyzer()
    batch_results = []
    
    for idx, ticker in enumerate(tickers, 1):
        print(f"\n[{idx}/{len(tickers)}] Processing {ticker}...")
        
        success = analyzer.run_analysis(ticker, years=years, 
                                       create_plots=True,
                                       output_dir=output_dir,
                                       metric=metric)
        
        if success:
            best = analyzer.get_best_model_details()
            batch_results.append(best)
    
    # Print summary
    print("\n" + "="*80)
    print("📊 BATCH ANALYSIS SUMMARY")
    print("="*80)
    
    for result in batch_results:
        print(f"\n{result['ticker']}:")
        print(f"  Best Model: {result['model']}")
        print(f"  RMSE: {result['rmse']:.8f}")
        print(f"  R²:   {result['r2']:.6f}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
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
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode
        batch_mode(args.batch, years=args.years, 
                  output_dir=args.output, metric=args.metric)
    elif args.ticker:
        # Single ticker mode
        analyzer = StockVolatilityAnalyzer()
        analyzer.run_analysis(args.ticker, years=args.years,
                            output_dir=args.output, metric=args.metric)
    else:
        # Interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()