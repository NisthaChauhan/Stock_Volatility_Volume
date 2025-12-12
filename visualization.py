"""
Module 3: Visualization
Clean plots with improved layout (table removed)
"""

import matplotlib.pyplot as plt
import numpy as np
import os

class ModelVisualizer:
    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol
        plt.style.use('seaborn-v0_8-darkgrid')

    def plot_comparison_grid(self, X, y, results, save_path=None):
        valid_results = {k: v for k, v in results.items() if v is not None}
        if not valid_results:
            print("No models to plot")
            return

        n_models = len(valid_results)
        cols = 4
        rows = (n_models + cols - 1) // cols

        # Create figure without extra space for table
        fig = plt.figure(figsize=(20, 5*rows))
        
        # Create grid spec without table row
        gs = fig.add_gridspec(rows, cols, hspace=0.35, wspace=0.25)
        
        fig.suptitle(f'{self.ticker} - All Models Comparison ({n_models} models)', 
                     fontsize=18, fontweight='bold', y=0.995)

        # Sort by RMSE (best to worst)
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['rmse'])

        # Create subplots for models
        axes = []
        for i in range(rows):
            for j in range(cols):
                ax = fig.add_subplot(gs[i, j])
                axes.append(ax)

        # Plot each model
        for i, (name, res) in enumerate(sorted_results):
            if i >= len(axes):
                break
                
            ax = axes[i]
            ax.scatter(X, y, alpha=0.4, s=20, color='steelblue', label='Data')
            
            X_line = np.linspace(X.min(), X.max(), 300)
            y_line = res['model_func'](X_line, *res['params'])
            ax.plot(X_line, y_line, 'r-', lw=2.5, label='Fit')

            # Clean title with just model name
            ax.set_title(f"{name}", fontsize=12, fontweight='bold', pad=10)
            ax.set_xlabel('Volume Δ (shares)', fontsize=10)
            ax.set_ylabel('Volatility Δ', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.25, linestyle='--')

        # Hide unused subplots
        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        plt.close()

    def plot_top_models(self, X, y, results, top_n=6, save_path=None):
        valid = {k: v for k, v in results.items() if v is not None}
        top = sorted(valid.items(), key=lambda x: x[1]['rmse'])[:top_n]

        cols = 2
        rows = (len(top) + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        fig.suptitle(f'{self.ticker} - Top {len(top)} Models', fontsize=16, fontweight='bold')

        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for i, (name, res) in enumerate(top):
            ax = axes[i]
            ax.scatter(X, y, alpha=0.4, s=20, color='steelblue')
            Xl = np.linspace(X.min(), X.max(), 400)
            yl = res['model_func'](Xl, *res['params'])
            ax.plot(Xl, yl, 'r-', lw=3)

            ax.set_title(f"{name}", fontsize=14, fontweight='bold')
            txt = f"RMSE: {res['rmse']:.6f}\nR²: {res['r2']:.5f}\nMAE: {res['mae']:.6f}"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="wheat"))

            ax.set_xlabel('Volume Δ (shares)')
            ax.set_ylabel('Volatility Δ')
            ax.grid(True, alpha=0.3)

        for j in range(i+1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()