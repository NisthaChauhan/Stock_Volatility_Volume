'''"""
Module 4: Advanced Model Selector
Comprehensive model selection and ranking system
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

class ModelSelector:
    """Advanced model selection and ranking system"""
    
    def __init__(self):
        self.rankings = None
        self.comparison_df = None
    
    def calculate_aic(self, n, rmse, k):
        """
        Calculate Akaike Information Criterion (AIC)
        Lower is better
        
        Args:
            n (int): Number of data points
            rmse (float): Root mean squared error
            k (int): Number of parameters
            
        Returns:
            float: AIC value
        """
        mse = rmse ** 2
        aic = n * np.log(mse) + 2 * k
        return aic
    
    def calculate_bic(self, n, rmse, k):
        """
        Calculate Bayesian Information Criterion (BIC)
        Lower is better, penalizes complexity more than AIC
        
        Args:
            n (int): Number of data points
            rmse (float): Root mean squared error
            k (int): Number of parameters
            
        Returns:
            float: BIC value
        """
        mse = rmse ** 2
        bic = n * np.log(mse) + k * np.log(n)
        return bic
    
    def calculate_adjusted_r2(self, r2, n, k):
        """
        Calculate Adjusted R² (accounts for number of parameters)
        
        Args:
            r2 (float): R-squared value
            n (int): Number of data points
            k (int): Number of parameters
            
        Returns:
            float: Adjusted R²
        """
        if n - k - 1 <= 0:
            return r2
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        return adj_r2
    
    def rank_models(self, results, X, primary_metric='rmse', 
                    include_ic=True):
        """
        Comprehensive model ranking with multiple metrics
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data (for calculating IC)
            primary_metric (str): Primary metric for ranking
            include_ic (bool): Include AIC/BIC calculations
            
        Returns:
            pd.DataFrame: Ranked models with all metrics
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("⚠️  No valid models to rank")
            return None
        
        n = len(X)
        data = []
        
        for name, result in valid_results.items():
            k = len(result['params'])  # Number of parameters
            
            row = {
                'Model': name,
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'Params_Count': k,
                'Adj_R²': self.calculate_adjusted_r2(result['r2'], n, k)
            }
            
            if include_ic:
                row['AIC'] = self.calculate_aic(n, result['rmse'], k)
                row['BIC'] = self.calculate_bic(n, result['rmse'], k)
            
            row['Parameters'] = result['params']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by primary metric
        ascending = primary_metric.upper() not in ['R²', 'ADJ_R²']
        sort_col = primary_metric.upper().replace('_', ' ')
        if sort_col == 'ADJ R²':
            sort_col = 'Adj_R²'
        
        df = df.sort_values(by=sort_col, ascending=ascending)
        df = df.reset_index(drop=True)
        
        # Add rank column
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        self.rankings = df
        return df
    
    def get_best_model(self, results, metric='rmse'):
        """
        Returns the single best model based on specified metric
        
        Args:
            results (dict): Results from ModelFitter
            metric (str): Metric to optimize
            
        Returns:
            tuple: (model_name, model_details)
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        metric_lower = metric.lower()
        
        if metric_lower in ['rmse', 'mae', 'aic', 'bic']:
            # Lower is better
            best_name = min(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_lower, 
                                                             float('inf')))
        elif metric_lower in ['r2', 'adj_r2']:
            # Higher is better
            metric_key = 'r2' if metric_lower == 'r2' else 'adj_r2'
            best_name = max(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_key, 
                                                             float('-inf')))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_name, valid_results[best_name]
    
    def print_rankings(self, results, X, metric='rmse', top_n=None):
        """
        Prints a beautifully formatted ranking table
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            metric (str): Primary metric for ranking
            top_n (int): Show only top N models (None = all)
        """
        df = self.rank_models(results, X, primary_metric=metric)
        
        if df is None:
            return
        
        if top_n:
            df_display = df.head(top_n).copy()
            title = f"TOP {top_n} MODEL RANKINGS"
        else:
            df_display = df.copy()
            title = "COMPLETE MODEL RANKINGS"
        
        # Format for display
        display_df = df_display[[
            'Rank', 'Model', 'RMSE', 'R²', 'MAE', 'Adj_R²', 
            'AIC', 'BIC', 'Params_Count'
        ]].copy()
        
        # Round numeric columns
        display_df['RMSE'] = display_df['RMSE'].map('{:.8f}'.format)
        display_df['R²'] = display_df['R²'].map('{:.6f}'.format)
        display_df['MAE'] = display_df['MAE'].map('{:.8f}'.format)
        display_df['Adj_R²'] = display_df['Adj_R²'].map('{:.6f}'.format)
        display_df['AIC'] = display_df['AIC'].map('{:.2f}'.format)
        display_df['BIC'] = display_df['BIC'].map('{:.2f}'.format)
        
        print(f"\n{'='*100}")
        print(f"📊 {title} (sorted by {metric.upper()})")
        print(f"{'='*100}")
        
        # Print as table
        print(tabulate(display_df, headers='keys', tablefmt='pretty', 
                      showindex=False))
        
        print(f"{'='*100}")
        
        # Highlight best model
        best_row = df.iloc[0]
        print(f"\n🏆 BEST MODEL: {best_row['Model']}")
        print(f"   Primary Metric ({metric.upper()}): {best_row[metric.upper().replace('_', ' ')]:.8f}")
        print(f"   R²: {best_row['R²']:.6f} | Adj R²: {best_row['Adj_R²']:.6f}")
        print(f"   Parameters: {best_row['Params_Count']}")
        print(f"{'='*100}\n")
    
    def print_detailed_ranking(self, results, X):
        """
        Prints detailed information for each model
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return
        
        print(f"\n{'='*100}")
        print(f"📋 DETAILED MODEL ANALYSIS")
        print(f"{'='*100}\n")
        
        for _, row in df.iterrows():
            print(f"{'─'*100}")
            print(f"Rank #{row['Rank']}: {row['Model']}")
            print(f"{'─'*100}")
            print(f"  Performance Metrics:")
            print(f"    • RMSE:       {row['RMSE']:.10f}")
            print(f"    • MAE:        {row['MAE']:.10f}")
            print(f"    • R²:         {row['R²']:.6f}")
            print(f"    • Adjusted R²: {row['Adj_R²']:.6f}")
            print(f"\n  Information Criteria:")
            print(f"    • AIC:        {row['AIC']:.2f}")
            print(f"    • BIC:        {row['BIC']:.2f}")
            print(f"\n  Model Complexity:")
            print(f"    • Parameters: {row['Params_Count']}")
            print(f"    • Values:     {[f'{p:.4e}' for p in row['Parameters']]}")
            print()
        
        print(f"{'='*100}\n")
    
    def compare_metrics(self, results, X):
        """
        Creates a comprehensive comparison with normalized scores
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            
        Returns:
            pd.DataFrame: Comparison table with normalized scores
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return None
        
        # Calculate normalized scores (0-1 scale)
        # For metrics where lower is better
        for col in ['RMSE', 'MAE', 'AIC', 'BIC']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_norm'] = 1 - (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_norm'] = 1.0
        
        # For metrics where higher is better
        for col in ['R²', 'Adj_R²']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_norm'] = 1.0
        
        # Overall score (weighted average)
        df['Overall_Score'] = (
            0.30 * df['RMSE_norm'] +
            0.20 * df['MAE_norm'] +
            0.20 * df['R²_norm'] +
            0.15 * df['Adj_R²_norm'] +
            0.10 * df['AIC_norm'] +
            0.05 * df['BIC_norm']
        )
        
        # Re-rank by overall score
        df = df.sort_values('Overall_Score', ascending=False).reset_index(drop=True)
        df['Rank'] = range(1, len(df) + 1)
        
        self.comparison_df = df
        
        return df[['Rank', 'Model', 'RMSE', 'R²', 'Adj_R²', 'AIC', 'BIC', 
                   'Overall_Score']]
    
    def get_recommendation(self, results, X, context='general'):
        """
        Provides intelligent recommendation based on context
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            context (str): Analysis context
            
        Returns:
            str: Recommendation text
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return "No valid models available for recommendation."
        
        recommendation = f"\n{'='*100}\n"
        recommendation += "🎯 INTELLIGENT MODEL RECOMMENDATION\n"
        recommendation += f"{'='*100}\n\n"
        
        if context == 'prediction':
            best_rmse = df.iloc[0]
            recommendation += f"🔮 FOR PREDICTION ACCURACY:\n"
            recommendation += f"   → {best_rmse['Model']}\n"
            recommendation += f"     RMSE: {best_rmse['RMSE']:.8f}\n"
            recommendation += f"     R²: {best_rmse['R²']:.6f}\n"
            recommendation += f"     This model minimizes prediction error.\n\n"
        
        elif context == 'interpretation':
            # Prefer simpler models with good performance
            simple_models = ['Linear', 'Square Root', 'Logarithmic', 'Quadratic']
            simple_df = df[df['Model'].isin(simple_models)]
            
            if not simple_df.empty:
                best_simple = simple_df.iloc[0]
                recommendation += f"📖 FOR INTERPRETABILITY:\n"
                recommendation += f"   → {best_simple['Model']}\n"
                recommendation += f"     RMSE: {best_simple['RMSE']:.8f}\n"
                recommendation += f"     R²: {best_simple['R²']:.6f}\n"
                recommendation += f"     Parameters: {best_simple['Params_Count']}\n"
                recommendation += f"     This model balances performance and simplicity.\n\n"
            else:
                recommendation += f"📖 FOR INTERPRETABILITY:\n"
                recommendation += f"   → Consider {df.iloc[0]['Model']} (best overall)\n\n"
        
        elif context == 'balanced':
            # Use AIC or BIC for balance
            best_aic = df.sort_values('AIC').iloc[0]
            recommendation += f"⚖️  FOR BALANCED APPROACH (AIC-based):\n"
            recommendation += f"   → {best_aic['Model']}\n"
            recommendation += f"     RMSE: {best_aic['RMSE']:.8f}\n"
            recommendation += f"     AIC: {best_aic['AIC']:.2f}\n"
            recommendation += f"     This model optimally balances fit and complexity.\n\n"
        
        else:  # general
            best_overall = df.iloc[0]
            comparison_df = self.compare_metrics(results, X)
            best_composite = comparison_df.iloc[0]
            
            recommendation += f"🌟 OVERALL BEST (by RMSE):\n"
            recommendation += f"   → {best_overall['Model']}\n"
            recommendation += f"     RMSE: {best_overall['RMSE']:.8f}\n"
            recommendation += f"     R²: {best_overall['R²']:.6f}\n"
            recommendation += f"     Adj R²: {best_overall['Adj_R²']:.6f}\n\n"
            
            if best_composite['Model'] != best_overall['Model']:
                recommendation += f"🎖️  BEST COMPOSITE SCORE:\n"
                recommendation += f"   → {best_composite['Model']}\n"
                recommendation += f"     Overall Score: {best_composite['Overall_Score']:.4f}\n"
                recommendation += f"     This model performs well across all metrics.\n\n"
        
        # Add top 3 summary
        recommendation += f"📊 TOP 3 MODELS SUMMARY:\n"
        for i, row in df.head(3).iterrows():
            recommendation += f"   {row['Rank']}. {row['Model']}: "
            recommendation += f"RMSE={row['RMSE']:.6f}, R²={row['R²']:.4f}\n"
        
        recommendation += f"\n{'='*100}\n"
        
        return recommendation
    
    def export_results(self, results, X, ticker, output_file='model_results.csv'):
        """
        Exports comprehensive results to CSV
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            ticker (str): Stock ticker symbol
            output_file (str): Output filename
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return
        
        # Add ticker column
        df.insert(0, 'Ticker', ticker)
        
        # Convert params list to string for CSV
        df['Parameters'] = df['Parameters'].apply(
            lambda x: '|'.join([f'{p:.8e}' for p in x])
        )
        
        df.to_csv(output_file, index=False)
        print(f"✅ Results exported to '{output_file}'")
    
    def generate_summary_report(self, results, X, ticker):
        """
        Generates a text summary report
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            ticker (str): Stock ticker symbol
            
        Returns:
            str: Summary report text
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return "No results available for summary."
        
        report = f"\n{'='*100}\n"
        report += f"📈 STOCK VOLATILITY ANALYSIS SUMMARY REPORT\n"
        report += f"{'='*100}\n\n"
        report += f"Ticker: {ticker}\n"
        report += f"Total Models Tested: {len(df)}\n"
        report += f"Data Points: {len(X)}\n"
        report += f"\n{'─'*100}\n\n"
        
        # Best performers
        report += "🏆 BEST PERFORMERS:\n\n"
        report += f"  Best RMSE:    {df.iloc[0]['Model']} ({df.iloc[0]['RMSE']:.8f})\n"
        
        best_r2 = df.sort_values('R²', ascending=False).iloc[0]
        report += f"  Best R²:      {best_r2['Model']} ({best_r2['R²']:.6f})\n"
        
        best_aic = df.sort_values('AIC').iloc[0]
        report += f"  Best AIC:     {best_aic['Model']} ({best_aic['AIC']:.2f})\n"
        
        # Simplest good model
        simple_df = df[df['Params_Count'] <= 2]
        if not simple_df.empty:
            best_simple = simple_df.iloc[0]
            report += f"  Best Simple:  {best_simple['Model']} ({best_simple['RMSE']:.8f})\n"
        
        report += f"\n{'─'*100}\n\n"
        report += self.get_recommendation(results, X, 'general')
        
        return report


# Example usage
if __name__ == "__main__":
    # Mock results for testing
    np.random.seed(42)
    X_test = np.random.randn(200) * 1e6
    
    mock_results = {
        'Linear': {'rmse': 0.0105, 'r2': 0.45, 'mae': 0.0087, 'params': [1e-9, 0.001]},
        'Quadratic': {'rmse': 0.0103, 'r2': 0.48, 'mae': 0.0085, 'params': [1e-18, 2e-9, 0.001]},
        'Cubic': {'rmse': 0.0102, 'r2': 0.50, 'mae': 0.0084, 'params': [1e-27, 1e-18, 2e-9, 0.001]},
    }
    
    selector = ModelSelector()
    selector.print_rankings(mock_results, X_test, top_n=5)
    print(selector.get_recommendation(mock_results, X_test))'''

"""
Module 4: Advanced Model Selector
Comprehensive model selection and ranking system
"""

import numpy as np
import pandas as pd
from tabulate import tabulate

class ModelSelector:
    """Advanced model selection and ranking system"""
    
    def __init__(self):
        self.rankings = None
        self.comparison_df = None
    
    def calculate_aic(self, n, rmse, k):
        """
        Calculate Akaike Information Criterion (AIC)
        Lower is better
        
        Args:
            n (int): Number of data points
            rmse (float): Root mean squared error
            k (int): Number of parameters
            
        Returns:
            float: AIC value
        """
        mse = rmse ** 2
        aic = n * np.log(mse) + 2 * k
        return aic
    
    def calculate_bic(self, n, rmse, k):
        """
        Calculate Bayesian Information Criterion (BIC)
        Lower is better, penalizes complexity more than AIC
        
        Args:
            n (int): Number of data points
            rmse (float): Root mean squared error
            k (int): Number of parameters
            
        Returns:
            float: BIC value
        """
        mse = rmse ** 2
        bic = n * np.log(mse) + k * np.log(n)
        return bic
    
    def calculate_adjusted_r2(self, r2, n, k):
        """
        Calculate Adjusted R² (accounts for number of parameters)
        
        Args:
            r2 (float): R-squared value
            n (int): Number of data points
            k (int): Number of parameters
            
        Returns:
            float: Adjusted R²
        """
        if n - k - 1 <= 0:
            return r2
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        return adj_r2
    
    def rank_models(self, results, X, primary_metric='rmse'):
        """
        Comprehensive model ranking with multiple metrics
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data (for calculating IC)
            primary_metric (str): Primary metric for ranking
            
        Returns:
            pd.DataFrame: Ranked models with all metrics
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("⚠️  No valid models to rank")
            return None
        
        n = len(X)
        data = []
        
        for name, result in valid_results.items():
            k = len(result['params'])  # Number of parameters
            
            row = {
                'Model': name,
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'Params_Count': k,
                'Adj_R²': self.calculate_adjusted_r2(result['r2'], n, k)
            }
            
            row['Parameters'] = result['params']
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Sort by primary metric
        ascending = primary_metric.upper() not in ['R²', 'ADJ_R²']
        sort_col = primary_metric.upper().replace('_', ' ')
        if sort_col == 'ADJ R²':
            sort_col = 'Adj_R²'
        
        df = df.sort_values(by=sort_col, ascending=ascending)
        df = df.reset_index(drop=True)
        
        # Add rank column
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        self.rankings = df
        return df
    
    def get_best_model(self, results, metric='rmse'):
        """
        Returns the single best model based on specified metric
        
        Args:
            results (dict): Results from ModelFitter
            metric (str): Metric to optimize
            
        Returns:
            tuple: (model_name, model_details)
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        metric_lower = metric.lower()
        
        if metric_lower in ['rmse', 'mae']:
            # Lower is better
            best_name = min(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_lower, 
                                                             float('inf')))
        elif metric_lower in ['r2']:
            # Higher is better
            best_name = max(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_lower, 
                                                             float('-inf')))
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_name, valid_results[best_name]
    
    def print_rankings(self, results, X, metric='rmse', top_n=None):
        """
        Prints a beautifully formatted ranking table
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            metric (str): Primary metric for ranking
            top_n (int): Show only top N models (None = all)
        """
        df = self.rank_models(results, X, primary_metric=metric)
        
        if df is None:
            return
        
        if top_n:
            df_display = df.head(top_n).copy()
            title = f"TOP {top_n} MODEL RANKINGS"
        else:
            df_display = df.copy()
            title = "MODEL RANKINGS"
        
        # Format for display - only show Rank, Model, RMSE, R², MAE
        display_df = df_display[['Rank', 'Model', 'RMSE', 'R²', 'MAE']].copy()
        
        # Round numeric columns
        display_df['RMSE'] = display_df['RMSE'].map('{:.8f}'.format)
        display_df['R²'] = display_df['R²'].map('{:.6f}'.format)
        display_df['MAE'] = display_df['MAE'].map('{:.8f}'.format)
        
        print(f"\n{'='*80}")
        print(f"📊 {title} (sorted by {metric.upper()})")
        print(f"{'='*80}")
        
        # Print as table
        print(tabulate(display_df, headers='keys', tablefmt='pretty', 
                      showindex=False))
        
        print(f"{'='*80}")
    
    def get_summary(self, results, X):
        """
        Provides summary based on context
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            
        Returns:
            str: Summary text
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return "No valid models available for summary."
        
        summary = f"\n{'='*80}\n"
        summary += "📊 SUMMARY\n"
        summary += f"{'='*80}\n\n"
        
        best_overall = df.iloc[0]
        
        summary += f"🏆 BEST MODEL: {best_overall['Model']}\n"
        summary += f"   RMSE: {best_overall['RMSE']:.8f}\n"
        summary += f"   R²:   {best_overall['R²']:.6f}\n"
        summary += f"   MAE:  {best_overall['MAE']:.8f}\n\n"
        
        # Add top 3 summary
        summary += f"TOP 3 MODELS:\n"
        for i, row in df.head(3).iterrows():
            summary += f"   {row['Rank']}. {row['Model']}: "
            summary += f"RMSE={row['RMSE']:.6f}, R²={row['R²']:.4f}\n"
        
        summary += f"\n{'='*80}\n"
        
        return summary
    
    def export_results(self, results, X, ticker, output_file='model_results.csv'):
        """
        Exports comprehensive results to CSV
        
        Args:
            results (dict): Results from ModelFitter
            X (np.array): Input data
            ticker (str): Stock ticker symbol
            output_file (str): Output filename
        """
        df = self.rank_models(results, X)
        
        if df is None:
            return
        
        # Add ticker column
        df.insert(0, 'Ticker', ticker)
        
        # Convert params list to string for CSV
        df['Parameters'] = df['Parameters'].apply(
            lambda x: '|'.join([f'{p:.8e}' for p in x])
        )
        
        df.to_csv(output_file, index=False)
        print(f"✅ Results exported to '{output_file}'")


# Example usage
if __name__ == "__main__":
    # Mock results for testing
    np.random.seed(42)
    X_test = np.random.randn(200) * 1e6
    
    mock_results = {
        'Linear': {'rmse': 0.0105, 'r2': 0.45, 'mae': 0.0087, 'params': [1e-9, 0.001]},
        'Quadratic': {'rmse': 0.0103, 'r2': 0.48, 'mae': 0.0085, 'params': [1e-18, 2e-9, 0.001]},
        'Cubic': {'rmse': 0.0102, 'r2': 0.50, 'mae': 0.0084, 'params': [1e-27, 1e-18, 2e-9, 0.001]},
    }
    
    selector = ModelSelector()
    selector.print_rankings(mock_results, X_test, top_n=5)
    print(selector.get_summary(mock_results, X_test))