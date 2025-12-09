'''
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
        self.rankings=None
        self.comparison_df=None
    
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
        mse=rmse ** 2
        aic=n * np.log(mse) + 2 * k
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
        mse=rmse ** 2
        bic=n * np.log(mse) + k * np.log(n)
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
        adj_r2=1 - (1 - r2) * (n - 1) / (n - k - 1)
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
        valid_results={k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("⚠️  No valid models to rank")
            return None
        
        n=len(X)
        data=[]
        
        for name, result in valid_results.items():
            k=len(result['params'])  # Number of parameters
            
            row={
                'Model': name,
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'Params_Count': k,
                'Adj_R²': self.calculate_adjusted_r2(result['r2'], n, k)
            }
            
            row['Parameters']=result['params']
            
            data.append(row)
        
        df=pd.DataFrame(data)
        
        # Sort by primary metric
        ascending=primary_metric.upper() not in ['R²', 'ADJ_R²']
        sort_col=primary_metric.upper().replace('_', ' ')
        if sort_col == 'ADJ R²':
            sort_col='Adj_R²'
        
        df=df.sort_values(by=sort_col, ascending=ascending)
        df=df.reset_index(drop=True)
        
        # Add rank column
        df.insert(0, 'Rank', range(1, len(df) + 1))
        
        self.rankings=df
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
        valid_results={k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        metric_lower=metric.lower()
        
        if metric_lower in ['rmse', 'mae']:
            # Lower is better
            best_name=min(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_lower, 
                                                             float('inf')))
        elif metric_lower in ['r2']:
            # Higher is better
            best_name=max(valid_results.keys(), 
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
            top_n (int): Show only top N models (None=all)
        """
        df=self.rank_models(results, X, primary_metric=metric)
        
        if df is None:
            return
        
        if top_n:
            df_display=df.head(top_n).copy()
            title=f"TOP {top_n} MODEL RANKINGS"
        else:
            df_display=df.copy()
            title="MODEL RANKINGS"
        
        # Format for display - only show Rank, Model, RMSE, R², MAE
        display_df=df_display[['Rank', 'Model', 'RMSE', 'R²', 'MAE']].copy()
        
        # Round numeric columns
        display_df['RMSE']=display_df['RMSE'].map('{:.8f}'.format)
        display_df['R²']=display_df['R²'].map('{:.6f}'.format)
        display_df['MAE']=display_df['MAE'].map('{:.8f}'.format)
        
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
        df=self.rank_models(results, X)
        
        if df is None:
            return "No valid models available for summary."
        
        summary=f"\n{'='*80}\n"
        summary += "📊 SUMMARY\n"
        summary += f"{'='*80}\n\n"
        
        best_overall=df.iloc[0]
        
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
        df=self.rank_models(results, X)
        
        if df is None:
            return
        
        # Add ticker column
        df.insert(0, 'Ticker', ticker)
        
        # Convert params list to string for CSV
        df['Parameters']=df['Parameters'].apply(
            lambda x: '|'.join([f'{p:.8e}' for p in x])
        )
        
        df.to_csv(output_file, index=False)
        print(f"✅ Results exported to '{output_file}'")


# Example usage
if __name__ == "__main__":
    # Mock results for testing
    np.random.seed(42)
    X_test=np.random.randn(200) * 1e6
    
    mock_results={
        'Linear': {'rmse': 0.0105, 'r2': 0.45, 'mae': 0.0087, 'params': [1e-9, 0.001]},
        'Quadratic': {'rmse': 0.0103, 'r2': 0.48, 'mae': 0.0085, 'params': [1e-18, 2e-9, 0.001]},
        'Cubic': {'rmse': 0.0102, 'r2': 0.50, 'mae': 0.0084, 'params': [1e-27, 1e-18, 2e-9, 0.001]},
    }
    
    selector=ModelSelector()
    selector.print_rankings(mock_results, X_test, top_n=5)
    print(selector.get_summary(mock_results, X_test))

'''

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
        self.rankings=None
        self.comparison_df=None
    
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
        mse=rmse ** 2
        aic=n * np.log(mse) + 2 * k
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
        mse=rmse ** 2
        bic=n * np.log(mse) + k * np.log(n)
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
        adj_r2=1 - (1 - r2) * (n - 1) / (n - k - 1)
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
        valid_results={k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("⚠️  No valid models to rank")
            return None
        
        n=len(X)
        data=[]
        
        for name, result in valid_results.items():
            k=len(result['params'])  # Number of parameters
            
            row={
                'Model': name,
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'MAE': result['mae'],
                'Params_Count': k,
                'Adj_R²': self.calculate_adjusted_r2(result['r2'], n, k)
            }
            
            row['Parameters']=result['params']
            
            data.append(row)
        
        df=pd.DataFrame(data)
        
        # Sort by primary metric
        ascending=primary_metric.upper() not in ['R²', 'ADJ_R²']
        sort_col=primary_metric.upper().replace('_', ' ')
        if sort_col == 'ADJ R²':
            sort_col='Adj_R²'
        
        df=df.sort_values(by=sort_col, ascending=ascending)
        df=df.reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        self.rankings=df
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
        valid_results={k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        metric_lower=metric.lower()
        
        if metric_lower in ['rmse', 'mae']:
            # Lower is better
            best_name=min(valid_results.keys(), 
                          key=lambda k: valid_results[k].get(metric_lower, 
                                                             float('inf')))
        elif metric_lower in ['r2']:
            # Higher is better
            best_name=max(valid_results.keys(), 
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
            top_n (int): Show only top N models (None=all)
        """
        df=self.rank_models(results, X, primary_metric=metric)
        if df is None:
            return
        
        if top_n:
            df_display=df.head(top_n).copy()
            title=f"TOP {top_n} MODEL RANKINGS"
        else:
            df_display=df.copy()
            title="MODEL RANKINGS"
        display_df=df_display[['Rank', 'Model', 'RMSE', 'R²', 'MAE']].copy()
        display_df['RMSE']=display_df['RMSE'].map('{:.8f}'.format)
        display_df['R²']=display_df['R²'].map('{:.6f}'.format)
        display_df['MAE']=display_df['MAE'].map('{:.8f}'.format)
        
        print(f"\n{'='*80}")
        print(f"{title} (sorted by {metric.upper()})")
        print(f"{'='*80}")
        
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
        df=self.rank_models(results, X)
        
        if df is None:
            return "No valid models available for summary."
        
        summary=f"\n{'='*80}\n"
        summary += "SUMMARY\n"
        summary += f"{'='*80}\n\n"
        
        best_overall=df.iloc[0]
        
        summary += f"BEST MODEL: {best_overall['Model']}\n"
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
        df=self.rank_models(results, X)
        
        if df is None:
            return
        
        # Add ticker column
        df.insert(0, 'Ticker', ticker)
        
        # Convert params list to string for CSV
        df['Parameters']=df['Parameters'].apply(
            lambda x: '|'.join([f'{p:.8e}' for p in x])
        )
        
        df.to_csv(output_file, index=False)
        print(f"Results exported to '{output_file}'")


if __name__ == "__main__":
    # Mock results for testing
    np.random.seed(42)
    X_test=np.random.randn(200) * 1e6
    
    mock_results={
        'Linear': {'rmse': 0.0105, 'r2': 0.45, 'mae': 0.0087, 'params': [1e-9, 0.001]},
        'Quadratic': {'rmse': 0.0103, 'r2': 0.48, 'mae': 0.0085, 'params': [1e-18, 2e-9, 0.001]},
        'Cubic': {'rmse': 0.0102, 'r2': 0.50, 'mae': 0.0084, 'params': [1e-27, 1e-18, 2e-9, 0.001]},
    }
    
    selector=ModelSelector()
    selector.print_rankings(mock_results, X_test, top_n=5)
    print(selector.get_summary(mock_results, X_test))