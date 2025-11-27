"""
Module 3: Visualization
Creates comprehensive plots for regression analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from regression_models import RegressionModels

class ModelVisualizer:
    """Creates visualizations for regression analysis"""
    
    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol
        self.latex_equations = RegressionModels.get_latex_equations()
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
    def plot_individual_model(self, X, y, model_name, model_result, save_path=None):
        """
        Creates a detailed plot for a single model
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            model_name (str): Name of the model
            model_result (dict): Results from ModelFitter
            save_path (str): Optional path to save the figure
        """
        if model_result is None:
            print(f"⚠️  No results available for {model_name}")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # === LEFT PLOT: Data + Fitted Curve ===
        # Scatter plot of actual data
        ax1.scatter(X, y, alpha=0.4, s=15, label='Actual Data', 
                   color='steelblue', edgecolors='none')
        
        # Generate smooth fitted curve
        X_smooth = np.linspace(X.min(), X.max(), 500)
        y_smooth = model_result['model_func'](X_smooth, *model_result['params'])
        
        # Plot fitted curve
        ax1.plot(X_smooth, y_smooth, 'r-', linewidth=2.5, 
                label='Fitted Curve', alpha=0.9)
        
        # Labels and title
        ax1.set_xlabel('Volume Delta (ΔV)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Volatility Delta (ΔVol)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{model_name} Model - Fit Visualization', 
                     fontsize=13, fontweight='bold', pad=15)
        
        # Add equation as text
        equation_text = self.latex_equations.get(model_name, "")
        ax1.text(0.5, -0.15, equation_text, transform=ax1.transAxes,
                fontsize=11, ha='center', style='italic')
        
        ax1.legend(loc='best', framealpha=0.9)
        ax1.grid(True, alpha=0.3)
        
        # === RIGHT PLOT: Residuals ===
        residuals = y - model_result['y_pred']
        
        # Residual scatter
        ax2.scatter(model_result['y_pred'], residuals, alpha=0.4, s=15,
                   color='coral', edgecolors='none')
        ax2.axhline(y=0, color='darkred', linestyle='--', linewidth=2, 
                   label='Zero Line')
        
        # Add +/- 2 std bands
        std_resid = np.std(residuals)
        ax2.axhline(y=2*std_resid, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax2.axhline(y=-2*std_resid, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        
        ax2.set_xlabel('Predicted ΔVol', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax2.set_title('Residual Analysis', fontsize=13, fontweight='bold', pad=15)
        ax2.legend(loc='best', framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        # === Main Title ===
        fig.suptitle(f'{self.ticker} - {model_name} Regression Analysis', 
                    fontsize=15, fontweight='bold', y=1.02)
        
        # === Metrics Text Box ===
        textstr = f"Performance Metrics:\n"
        textstr += f"{'─'*30}\n"
        textstr += f"RMSE:  {model_result['rmse']:.8f}\n"
        textstr += f"R²:    {model_result['r2']:.6f}\n"
        textstr += f"MAE:   {model_result['mae']:.8f}\n"
        textstr += f"{'─'*30}\n"
        textstr += f"Parameters:\n"
        
        param_labels = ['a', 'b', 'c', 'd', 'e']
        for i, p in enumerate(model_result['params']):
            label = param_labels[i] if i < len(param_labels) else f'p{i+1}'
            textstr += f"  {label} = {p:.4e}\n"
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, 
                    edgecolor='black', linewidth=1.5)
        fig.text(0.98, 0.5, textstr, transform=fig.transFigure,
                fontsize=9, verticalalignment='center', 
                bbox=props, family='monospace')
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.98])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"  💾 Saved: {os.path.basename(save_path)}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_all_models(self, X, y, results, output_dir='plots'):
        """
        Creates separate detailed plots for all fitted models
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            output_dir (str): Directory to save plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Creating individual plots for {self.ticker}...")
        print(f"   Output directory: {output_dir}/")
        
        successful = 0
        failed = 0
        
        for model_name, result in results.items():
            if result is not None:
                filename = f"{self.ticker}_{model_name.replace(' ', '_')}.png"
                save_path = os.path.join(output_dir, filename)
                self.plot_individual_model(X, y, model_name, result, save_path)
                successful += 1
            else:
                failed += 1
        
        print(f"✅ Created {successful} plots successfully")
        if failed > 0:
            print(f"⚠️  {failed} models failed to converge")
    
    def plot_comparison_grid(self, X, y, results, save_path=None):
        """
        Creates a large comparison grid showing all models
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            save_path (str): Optional path to save the figure
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        n_models = len(valid_results)
        
        if n_models == 0:
            print("⚠️  No valid models to plot")
            return
        
        # Create subplot grid (4 columns)
        n_cols = 4
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        fig.suptitle(f'{self.ticker} - Model Comparison (All {n_models} Models)', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        # Flatten axes array for easy iteration
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        # Sort by RMSE for better visualization
        sorted_results = sorted(valid_results.items(), 
                               key=lambda x: x[1]['rmse'])
        
        for idx, (model_name, result) in enumerate(sorted_results):
            ax = axes_flat[idx]
            
            # Scatter plot (smaller points for cleaner look)
            ax.scatter(X, y, alpha=0.25, s=8, color='steelblue', 
                      edgecolors='none', rasterized=True)
            
            # Fitted curve
            X_smooth = np.linspace(X.min(), X.max(), 300)
            y_smooth = result['model_func'](X_smooth, *result['params'])
            ax.plot(X_smooth, y_smooth, 'r-', linewidth=2.5, alpha=0.9)
            
            # Title with metrics
            title_text = f"{model_name}\n"
            title_text += f"RMSE: {result['rmse']:.6f} | R²: {result['r2']:.4f}"
            ax.set_title(title_text, fontsize=10, fontweight='bold', pad=8)
            
            ax.set_xlabel('ΔV', fontsize=9)
            ax.set_ylabel('ΔVol', fontsize=9)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.tick_params(labelsize=8)
            
            # Add rank badge
            rank_text = f"#{idx+1}"
            ax.text(0.05, 0.95, rank_text, transform=ax.transAxes,
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8),
                   verticalalignment='top')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  💾 Saved comparison grid: {os.path.basename(save_path)}")
        else:
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
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) == 0:
            print("⚠️  No valid models to plot")
            return
        
        # Get top N models by RMSE
        sorted_results = sorted(valid_results.items(), 
                               key=lambda x: x[1]['rmse'])[:top_n]
        
        n_models = len(sorted_results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        fig.suptitle(f'{self.ticker} - Top {n_models} Best Performing Models', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        for idx, (model_name, result) in enumerate(sorted_results):
            ax = axes_flat[idx]
            
            # Scatter plot
            ax.scatter(X, y, alpha=0.3, s=12, color='steelblue', 
                      edgecolors='none')
            
            # Fitted curve
            X_smooth = np.linspace(X.min(), X.max(), 400)
            y_smooth = result['model_func'](X_smooth, *result['params'])
            ax.plot(X_smooth, y_smooth, 'r-', linewidth=2.5)
            
            # Title and equation
            ax.set_title(f"#{idx+1}: {model_name}", 
                        fontsize=12, fontweight='bold', pad=10)
            
            equation = self.latex_equations.get(model_name, "")
            ax.text(0.5, -0.18, equation, transform=ax.transAxes,
                   fontsize=9, ha='center', style='italic')
            
            # Metrics box
            metrics_text = f"RMSE: {result['rmse']:.6f}\nR²: {result['r2']:.5f}"
            ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top', 
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.9, edgecolor='black'))
            
            ax.set_xlabel('Volume Delta (ΔV)', fontsize=10)
            ax.set_ylabel('Volatility Delta (ΔVol)', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0.02, 1, 0.99])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  💾 Saved top models: {os.path.basename(save_path)}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_residual_analysis(self, X, y, results, save_path=None):
        """
        Creates comprehensive residual analysis for all models
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            save_path (str): Optional path to save the figure
        """
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if not valid_results:
            print("⚠️  No valid models for residual analysis")
            return
        
        # Get top 6 models
        sorted_results = sorted(valid_results.items(), 
                               key=lambda x: x[1]['rmse'])[:6]
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle(f'{self.ticker} - Residual Analysis (Top 6 Models)', 
                    fontsize=16, fontweight='bold')
        
        axes_flat = axes.flatten()
        
        for idx, (model_name, result) in enumerate(sorted_results):
            ax = axes_flat[idx]
            
            residuals = y - result['y_pred']
            
            # Q-Q plot style residual visualization
            ax.scatter(model_result['y_pred'], residuals, 
                      alpha=0.4, s=15, color='coral', edgecolors='none')
            ax.axhline(y=0, color='darkred', linestyle='--', linewidth=2)
            
            # Add confidence bands
            std_resid = np.std(residuals)
            ax.axhline(y=2*std_resid, color='gray', linestyle=':', alpha=0.7)
            ax.axhline(y=-2*std_resid, color='gray', linestyle=':', alpha=0.7)
            ax.fill_between(ax.get_xlim(), -2*std_resid, 2*std_resid, 
                           alpha=0.1, color='gray')
            
            ax.set_title(f"{model_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel('Predicted Values', fontsize=9)
            ax.set_ylabel('Residuals', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            stats_text = f"Std: {std_resid:.4e}\nMean: {np.mean(residuals):.4e}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"  💾 Saved residual analysis: {os.path.basename(save_path)}")
        else:
            plt.show()
        
        plt.close()
    
    def create_full_report(self, X, y, results, output_dir='plots'):
        """
        Creates all visualization types in one go
        
        Args:
            X (np.array): Volume_delta data
            y (np.array): Volatility_delta data
            results (dict): Results from ModelFitter
            output_dir (str): Directory to save all plots
        """
        print(f"\n📊 Generating complete visualization report for {self.ticker}...")
        
        # 1. Individual model plots
        self.plot_all_models(X, y, results, output_dir)
        
        # 2. Comparison grid
        grid_path = os.path.join(output_dir, f'{self.ticker}_comparison_grid.png')
        self.plot_comparison_grid(X, y, results, grid_path)
        
        # 3. Top models focus
        top_path = os.path.join(output_dir, f'{self.ticker}_top_models.png')
        self.plot_top_models(X, y, results, top_n=6, save_path=top_path)
        
        # 4. Residual analysis
        resid_path = os.path.join(output_dir, f'{self.ticker}_residual_analysis.png')
        self.plot_residual_analysis(X, y, results, resid_path)
        
        print(f"\n✅ Complete visualization report generated!")
        print(f"   Location: {output_dir}/")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X_test = np.random.randn(300) * 1e6
    y_test = 0.5e-9 * X_test + 0.001 + np.random.randn(300) * 0.01
    
    # Mock results
    mock_result = {
        'params': [0.5e-9, 0.001],
        'rmse': 0.01,
        'r2': 0.5,
        'mae': 0.008,
        'y_pred': 0.5e-9 * X_test + 0.001,
        'model_func': lambda x, a, b: a*x + b
    }
    
    mock_results = {'Linear': mock_result, 'Quadratic': mock_result}
    
    viz = ModelVisualizer('TEST')
    viz.create_full_report(X_test, y_test, mock_results, 'test_plots')