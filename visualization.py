"""
Module 3: Visualization
Creates simplified plots for regression analysis (fitting equations only)
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from regression_models import RegressionModels

class ModelVisualizer:
    """Creates simplified visualizations for regression analysis"""
    
    def __init__(self, ticker_symbol):
        self.ticker=ticker_symbol
        self.latex_equations=RegressionModels.get_latex_equations()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_comparison_grid(self, X, y, results, save_path=None):
        """
        Creates a large comparison grid showing all models (fitting equations only)
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            save_path (str): Optional path to save the figure
        """
        valid_results={k: v for k, v in results.items() if v is not None}
        n_models=len(valid_results)
        
        if n_models == 0:
            print(" No valid models to plot")
            return
        
        # Create subplot grid (4 columns)
        n_cols=4
        n_rows=(n_models + n_cols - 1) // n_cols
        
        fig, axes=plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle(f'{self.ticker} - Model Comparison (All {n_models} Models)', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Flatten axes array for easy iteration
        if n_rows == 1:
            axes=axes.reshape(1, -1)
        axes_flat=axes.flatten()
        
        # Sort by RMSE for better visualization
        sorted_results=sorted(valid_results.items(), 
                               key=lambda x: x[1]['rmse'])
        
        for idx, (model_name, result) in enumerate(sorted_results):
            ax=axes_flat[idx]
            
            # Scatter plot of actual data
            ax.scatter(X, y, alpha=0.3, s=20, color='steelblue', 
                      edgecolors='none', label='Actual Data')
            
            # Fitted curve
            X_smooth=np.linspace(X.min(), X.max(), 300)
            y_smooth=result['model_func'](X_smooth, *result['params'])
            ax.plot(X_smooth, y_smooth, 'r-', linewidth=2.5, alpha=0.9, 
                   label='Fitted Curve')
            
            # Title with metrics and rank
            title_text=f"#{idx+1}: {model_name}\n"
            title_text += f"RMSE: {result['rmse']:.6f} | R²: {result['r2']:.4f}"
            ax.set_title(title_text, fontsize=11, fontweight='bold', pad=10)
            
            ax.set_xlabel('Volume Delta (ΔV)', fontsize=10)
            ax.set_ylabel('Volatility Delta (ΔVol)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
            
            # Add equation as text below the plot
            equation=self.latex_equations.get(model_name, "")
            ax.text(0.5, -0.25, equation, transform=ax.transAxes,
                   fontsize=9, ha='center', style='italic')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.98])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved comparison grid: {os.path.basename(save_path)}")
        
        plt.show()
        plt.close()
    
    def plot_top_models(self, X, y, results, top_n=6, save_path=None):
        """
        Creates a focused comparison of the top N best models
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            top_n (int): Number of top models to display
            save_path (str): Optional path to save the figure
        """
        valid_results={k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) == 0:
            print("⚠️  No valid models to plot")
            return
        
        # Get top N models by RMSE
        sorted_results=sorted(valid_results.items(), 
                               key=lambda x: x[1]['rmse'])[:top_n]
        
        n_models=len(sorted_results)
        n_cols=2
        n_rows=(n_models + n_cols - 1) // n_cols
        
        fig, axes=plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        fig.suptitle(f'{self.ticker} - Top {n_models} Best Performing Models', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if n_rows == 1:
            axes=axes.reshape(1, -1)
        axes_flat=axes.flatten()
        
        for idx, (model_name, result) in enumerate(sorted_results):
            ax=axes_flat[idx]
            
            # Scatter plot
            ax.scatter(X, y, alpha=0.3, s=25, color='steelblue', 
                      edgecolors='none', label='Actual Data')
            
            # Fitted curve
            X_smooth=np.linspace(X.min(), X.max(), 400)
            y_smooth=result['model_func'](X_smooth, *result['params'])
            ax.plot(X_smooth, y_smooth, 'r-', linewidth=3, label='Fitted Curve')
            
            # Title and metrics
            ax.set_title(f"#{idx+1}: {model_name}", 
                        fontsize=13, fontweight='bold', pad=12)
            
            # Metrics box
            metrics_text=f"RMSE: {result['rmse']:.6f}\nR²: {result['r2']:.5f}\nMAE: {result['mae']:.6f}"
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.9, edgecolor='black'))
            
            # Equation
            equation=self.latex_equations.get(model_name, "")
            ax.text(0.5, -0.2, equation, transform=ax.transAxes,
                   fontsize=10, ha='center', style='italic')
            
            ax.set_xlabel('Volume Delta (ΔV)', fontsize=11)
            ax.set_ylabel('Volatility Delta (ΔVol)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        plt.show()
        plt.close()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X_test=np.random.randn(300) * 1e6
    y_test=0.5e-9 * X_test + 0.001 + np.random.randn(300) * 0.01
    
    # Mock results
    mock_result={
        'params': [0.5e-9, 0.001],
        'rmse': 0.01,
        'r2': 0.5,
        'mae': 0.008,
        'y_pred': 0.5e-9 * X_test + 0.001,
        'model_func': lambda x, a, b: a*x + b
    }
    
    mock_results={'Linear': mock_result, 'Quadratic': mock_result}
    
    viz=ModelVisualizer('TEST')
    viz.plot_comparison_grid(X_test, y_test, mock_results)