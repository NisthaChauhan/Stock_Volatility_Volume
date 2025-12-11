'''"""
Module 2: Regression Models
Defines mathematical models for volatility-volume relationship
"""

import numpy as np
from scipy.optimize import curve_fit

class RegressionModels:
    """Collection of mathematical models for fitting volatility vs volume"""
    
    def linear_model(self, x, a, b):
        """Linear model: Δvol=a·Δv + b"""
        return a * x + b
    
    def quadratic_model(self, x, a, b, c):
        """Quadratic model: Δvol=a·(Δv)² + b·Δv + c"""
        return a * x**2 + b * x + c
    
    def sqrt_model(self, x, a, b):
        """Square root model: Δvol=a·√|Δv|·sign(Δv) + b"""
        return a * np.sqrt(np.abs(x)) * np.sign(x) + b
    
    def logarithmic_model(self, x, a, b):
        """Logarithmic model: Δvol=a·log(|Δv| + 1)·sign(Δv) + b"""
        return a * np.log(np.abs(x) + 1) * np.sign(x) + b
    
    def regime_model(self, x, a, b, c):
        """Regime mixture model: Δvol=a·Δv + b·(Δv)³ + c"""
        return a * x + b * x**3 + c
    
    def exp_decay_model(self, x, a, b, c):
        """Exponential decay model: Δvol=a·(1 - e^(-b·|Δv|))·sign(Δv) + c"""
        return a * (1 - np.exp(-b * np.abs(x))) * np.sign(x) + c
    
    def cubic_model(self, x, a, b, c, d):
        """Cubic model: Δvol=a·(Δv)³ + b·(Δv)² + c·Δv + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def inverse_model(self, x, a, b, c):
        """Inverse model: Δvol=a/(|Δv| + b) · sign(Δv) + c"""
        return a / (np.abs(x) + b) * np.sign(x) + c
    
    def power_law_model(self, x, a, b, c):
        """Power law model: Δvol=a·|Δv|^b · sign(Δv) + c"""
        return a * np.power(np.abs(x) + 1, b) * np.sign(x) + c
    
    def tanh_model(self, x, a, b, c):
        """Hyperbolic tangent model: Δvol=a·tanh(b·Δv) + c"""
        return a * np.tanh(b * x) + c
    
    def sigmoid_model(self, x, a, b, c, d):
        """Sigmoid model: Δvol=a/(1 + e^(-b·(Δv - c))) + d"""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def gaussian_model(self, x, a, b, c, d):
        """Gaussian (bell curve) model: Δvol=a·e^(-(Δv-b)²/(2c²)) + d"""
        return a * np.exp(-((x - b)**2) / (2 * c**2)) + d
    
    def arctan_model(self, x, a, b, c):
        """Arctangent model: Δvol=a·arctan(b·Δv) + c"""
        return a * np.arctan(b * x) + c
    
    def rational_model(self, x, a, b, c, d):
        """Rational function: Δvol=(a·Δv + b)/(c·|Δv| + d) · sign(Δv)"""
        return (a * x + b) / (c * np.abs(x) + d) * np.sign(x)
    
    def sinusoidal_damped_model(self, x, a, b, c, d):
        """Damped sinusoidal: Δvol=a·sin(b·Δv)·e^(-c·|Δv|) + d"""
        return a * np.sin(b * x) * np.exp(-c * np.abs(x)) + d
    
    def piecewise_linear_model(self, x, a, b, c, d):
        """Piecewise linear: Different slopes for positive and negative Δv"""
        return np.where(x >= 0, a * x + b, c * x + d)
    
    def log_quadratic_model(self, x, a, b, c):
        """Log-quadratic hybrid: Δvol=a·(Δv)² + b·log(|Δv| + 1)·sign(Δv) + c"""
        return a * x**2 + b * np.log(np.abs(x) + 1) * np.sign(x) + c
    
    def weibull_model(self, x, a, b, c, d):
        """Weibull-inspired: Δvol=a·(1 - e^(-((|Δv|/b)^c))) · sign(Δv) + d"""
        return a * (1 - np.exp(-((np.abs(x) / b) ** c))) * np.sign(x) + d
    
    def get_all_models(self):
        """Returns dictionary of all available models"""
        return {
            'Linear': (self.linear_model, [0, 0]),
            'Quadratic': (self.quadratic_model, [0, 0, 0]),
            'Cubic': (self.cubic_model, [0, 0, 0, 0]),
            'Square Root': (self.sqrt_model, [0, 0]),
            'Logarithmic': (self.logarithmic_model, [0, 0]),
            'Regime Mixture': (self.regime_model, [0, 0, 0]),
            'Exponential Decay': (self.exp_decay_model, [1e-6, 1e-8, 0]),
            'Inverse': (self.inverse_model, [1e-3, 1e6, 0]),
            'Power Law': (self.power_law_model, [1e-6, 0.5, 0]),
            'Hyperbolic Tangent': (self.tanh_model, [0.01, 1e-7, 0]),
            'Sigmoid': (self.sigmoid_model, [0.01, 1e-7, 0, 0]),
            'Gaussian': (self.gaussian_model, [0.01, 0, 1e6, 0]),
            'Arctangent': (self.arctan_model, [0.01, 1e-7, 0]),
            'Rational': (self.rational_model, [1e-9, 0, 1e-9, 1]),
            'Damped Sinusoidal': (self.sinusoidal_damped_model, [0.01, 1e-7, 1e-8, 0]),
            'Piecewise Linear': (self.piecewise_linear_model, [1e-9, 0, 1e-9, 0]),
            'Log-Quadratic': (self.log_quadratic_model, [0, 0, 0]),  # <-- Changed to 3 values
            'Weibull': (self.weibull_model, [0.01, 1e6, 1.5, 0])
        }
    
    def get_latex_equations(self):
        """Returns LaTeX equations for each model"""
        return {
            'Linear': r'$\Delta\mathrm{Vol}=a \cdot \Delta V + b$',
            'Quadratic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \Delta V + c$',
            'Cubic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^3 + b \cdot (\Delta V)^2 + c \cdot \Delta V + d$',
            'Square Root': r'$\Delta\mathrm{Vol}=a \cdot \sqrt{|\Delta V|} \cdot \operatorname{sign}(\Delta V) + b$',
            'Logarithmic': r'$\Delta\mathrm{Vol}=a \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + b$',
            'Regime Mixture': r'$\Delta\mathrm{Vol}=a \cdot \Delta V + b \cdot (\Delta V)^3 + c$',
            'Exponential Decay': r'$\Delta\mathrm{Vol}=a \cdot (1 - e^{-b \cdot |\Delta V|}) \cdot \operatorname{sign}(\Delta V) + c$',
            'Inverse': r'$\Delta\mathrm{Vol}=\frac{a}{|\Delta V| + b} \cdot \operatorname{sign}(\Delta V) + c$',
            'Power Law': r'$\Delta\mathrm{Vol}=a \cdot (|\Delta V| + 1)^b \cdot \operatorname{sign}(\Delta V) + c$',
            'Hyperbolic Tangent': r'$\Delta\mathrm{Vol}=a \cdot \tanh(b \cdot \Delta V) + c$',
            'Sigmoid': r'$\Delta\mathrm{Vol}=\frac{a}{1 + e^{-b(\Delta V - c)}} + d$',
            'Gaussian': r'$\Delta\mathrm{Vol}=a \cdot e^{-\frac{(\Delta V - b)^2}{2c^2}} + d$',
            'Arctangent': r'$\Delta\mathrm{Vol}=a \cdot \arctan(b \cdot \Delta V) + c$',
            'Rational': r'$\Delta\mathrm{Vol}=\frac{a \cdot \Delta V + b}{c \cdot |\Delta V| + d} \cdot \operatorname{sign}(\Delta V)$',
            'Damped Sinusoidal': r'$\Delta\mathrm{Vol}=a \cdot \sin(b \cdot \Delta V) \cdot e^{-c \cdot |\Delta V|} + d$',
            'Piecewise Linear': 'ΔVol=a·ΔV + b  (ΔV ≥ 0);  c·ΔV + d  (ΔV < 0)',
            'Log-Quadratic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + c$',
            'Weibull': r'$\Delta\mathrm{Vol}=a \cdot \left(1 - e^{-\left(\frac{|\Delta V|}{b}\right)^c}\right) \cdot \operatorname{sign}(\Delta V) + d$',
        }


class ModelFitter:
    """Fits regression models to data and calculates metrics"""
    
    def __init__(self):
        self.regression_models=RegressionModels()
        self.models=self.regression_models.get_all_models()
        self.results={}
    
    def fit_all_models(self, X, y, max_iterations=10000):
        print("\nFitting models...")
        
        for name, (model_func, p0) in self.models.items():
            try:
                params, _=curve_fit(model_func, X, y, p0=p0, maxfev=max_iterations)
                y_pred=model_func(X, *params)
                
                rmse=self._calculate_rmse(y, y_pred)
                r2=self._calculate_r2(y, y_pred)
                mae=self._calculate_mae(y, y_pred)
                
                self.results[name]={
                    'params': params,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'y_pred': y_pred,
                    'model_func': model_func
                }
                
            except Exception as e:
                self.results[name]=None
        
        return self.results

    def _calculate_rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_r2(self, y_true, y_pred):
        """R-squared (coefficient of determination)"""
        ss_res=np.sum((y_true - y_pred) ** 2)
        ss_tot=np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _calculate_mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def get_best_model(self, metric='rmse'):
        """Returns the best performing model based on specified metric"""
        valid_results={k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        if metric in ['rmse', 'mae']:
            best_model=min(valid_results.items(), key=lambda x: x[1][metric])
        elif metric == 'r2':
            best_model=max(valid_results.items(), key=lambda x: x[1][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model
    
    def generate_smooth_curve(self, X, model_name, num_points=300):
        """Generates smooth curve for plotting"""
        if model_name not in self.results or self.results[model_name] is None:
            return None, None
        
        result=self.results[model_name]
        X_smooth=np.linspace(X.min(), X.max(), num_points)
        y_smooth=result['model_func'](X_smooth, *result['params'])
        
        return X_smooth, y_smooth


if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    X_test=np.random.randn(100) * 1e6
    y_test=0.5e-9 * X_test + 0.001 + np.random.randn(100) * 0.01
    
    fitter=ModelFitter()
    results=fitter.fit_all_models(X_test, y_test)
    
    best_name, best_result=fitter.get_best_model('rmse')
    print(f"\n🏆 Best model: {best_name} (RMSE: {best_result['rmse']:.6f})")'''

"""
Module 2: Regression Models
Defines mathematical models for volatility-volume relationship
"""

import numpy as np
from scipy.optimize import curve_fit

class RegressionModels:
    """Collection of mathematical models for fitting volatility vs volume"""
    
    def linear_model(self, x, a, b):
        """Linear model: Δvol=a·Δv + b"""
        return a * x + b
    
    def quadratic_model(self, x, a, b, c):
        """Quadratic model: Δvol=a·(Δv)² + b·Δv + c"""
        return a * x**2 + b * x + c
    
    def sqrt_model(self, x, a, b):
        """Square root model: Δvol=a·√|Δv|·sign(Δv) + b"""
        return a * np.sqrt(np.abs(x)) * np.sign(x) + b
    
    def logarithmic_model(self, x, a, b):
        """Logarithmic model: Δvol=a·log(|Δv| + 1)·sign(Δv) + b"""
        return a * np.log(np.abs(x) + 1) * np.sign(x) + b
    
    def regime_model(self, x, a, b, c):
        """Regime mixture model: Δvol=a·Δv + b·(Δv)³ + c"""
        return a * x + b * x**3 + c
    
    def exp_decay_model(self, x, a, b, c):
        """Exponential decay model: Δvol=a·(1 - e^(-b·|Δv|))·sign(Δv) + c"""
        return a * (1 - np.exp(-b * np.abs(x))) * np.sign(x) + c
    
    def cubic_model(self, x, a, b, c, d):
        """Cubic model: Δvol=a·(Δv)³ + b·(Δv)² + c·Δv + d"""
        return a * x**3 + b * x**2 + c * x + d
    
    def inverse_model(self, x, a, b, c):
        """Inverse model: Δvol=a/(|Δv| + b) · sign(Δv) + c"""
        return a / (np.abs(x) + b) * np.sign(x) + c
    
    def power_law_model(self, x, a, b, c):
        """Power law model: Δvol=a·|Δv|^b · sign(Δv) + c"""
        return a * np.power(np.abs(x) + 1, b) * np.sign(x) + c
    
    def tanh_model(self, x, a, b, c):
        """Hyperbolic tangent model: Δvol=a·tanh(b·Δv) + c"""
        return a * np.tanh(b * x) + c
    
    def sigmoid_model(self, x, a, b, c, d):
        """Sigmoid model: Δvol=a/(1 + e^(-b·(Δv - c))) + d"""
        return a / (1 + np.exp(-b * (x - c))) + d
    
    def gaussian_model(self, x, a, b, c, d):
        """Gaussian (bell curve) model: Δvol=a·e^(-(Δv-b)²/(2c²)) + d"""
        return a * np.exp(-((x - b)**2) / (2 * c**2)) + d
    
    def arctan_model(self, x, a, b, c):
        """Arctangent model: Δvol=a·arctan(b·Δv) + c"""
        return a * np.arctan(b * x) + c
    
    def rational_model(self, x, a, b, c, d):
        """Rational function: Δvol=(a·Δv + b)/(c·|Δv| + d) · sign(Δv)"""
        return (a * x + b) / (c * np.abs(x) + d) * np.sign(x)
    
    def sinusoidal_damped_model(self, x, a, b, c, d):
        """Damped sinusoidal: Δvol=a·sin(b·Δv)·e^(-c·|Δv|) + d"""
        return a * np.sin(b * x) * np.exp(-c * np.abs(x)) + d
    
    def piecewise_linear_model(self, x, a, b, c, d):
        """Piecewise linear: Different slopes for positive and negative Δv"""
        return np.where(x >= 0, a * x + b, c * x + d)
    
    def log_quadratic_model(self, x, a, b, c, d):
        """Log-quadratic hybrid: Δvol=a·(Δv)² + b·log(|Δv| + 1)·sign(Δv) + c"""
        return a * x**2 + b * np.log(np.abs(x) + 1) * np.sign(x) + c
    
    def weibull_model(self, x, a, b, c, d):
        """Weibull-inspired: Δvol=a·(1 - e^(-((|Δv|/b)^c))) · sign(Δv) + d"""
        return a * (1 - np.exp(-((np.abs(x) / b) ** c))) * np.sign(x) + d
    
    def get_all_models(self):
        """Returns dictionary of all available models"""
        return {
            'Linear': (self.linear_model, [0, 0]),
            'Quadratic': (self.quadratic_model, [0, 0, 0]),
            'Cubic': (self.cubic_model, [0, 0, 0, 0]),
            'Square Root': (self.sqrt_model, [0, 0]),
            'Logarithmic': (self.logarithmic_model, [0, 0]),
            'Regime Mixture': (self.regime_model, [0, 0, 0]),
            'Exponential Decay': (self.exp_decay_model, [1e-6, 1e-8, 0]),
            'Inverse': (self.inverse_model, [1e-3, 1e6, 0]),
            'Power Law': (self.power_law_model, [1e-6, 0.5, 0]),
            'Hyperbolic Tangent': (self.tanh_model, [0.01, 1e-7, 0]),
            'Sigmoid': (self.sigmoid_model, [0.01, 1e-7, 0, 0]),
            'Gaussian': (self.gaussian_model, [0.01, 0, 1e6, 0]),
            'Arctangent': (self.arctan_model, [0.01, 1e-7, 0]),
            'Rational': (self.rational_model, [1e-9, 0, 1e-9, 1]),
            'Damped Sinusoidal': (self.sinusoidal_damped_model, [0.01, 1e-7, 1e-8, 0]),
            'Piecewise Linear': (self.piecewise_linear_model, [1e-9, 0, 1e-9, 0]),
            'Log-Quadratic': (self.log_quadratic_model, [0, 0, 0, 0]),
            'Weibull': (self.weibull_model, [0.01, 1e6, 1.5, 0])
        }
    
    @staticmethod
    def get_latex_equations():
        """Returns LaTeX equations for each model"""
        return {
            'Linear': r'$\Delta\mathrm{Vol}=a \cdot \Delta V + b$',
            'Quadratic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \Delta V + c$',
            'Cubic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^3 + b \cdot (\Delta V)^2 + c \cdot \Delta V + d$',
            'Square Root': r'$\Delta\mathrm{Vol}=a \cdot \sqrt{|\Delta V|} \cdot \operatorname{sign}(\Delta V) + b$',
            'Logarithmic': r'$\Delta\mathrm{Vol}=a \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + b$',
            'Regime Mixture': r'$\Delta\mathrm{Vol}=a \cdot \Delta V + b \cdot (\Delta V)^3 + c$',
            'Exponential Decay': r'$\Delta\mathrm{Vol}=a \cdot (1 - e^{-b \cdot |\Delta V|}) \cdot \operatorname{sign}(\Delta V) + c$',
            'Inverse': r'$\Delta\mathrm{Vol}=\frac{a}{|\Delta V| + b} \cdot \operatorname{sign}(\Delta V) + c$',
            'Power Law': r'$\Delta\mathrm{Vol}=a \cdot (|\Delta V| + 1)^b \cdot \operatorname{sign}(\Delta V) + c$',
            'Hyperbolic Tangent': r'$\Delta\mathrm{Vol}=a \cdot \tanh(b \cdot \Delta V) + c$',
            'Sigmoid': r'$\Delta\mathrm{Vol}=\frac{a}{1 + e^{-b(\Delta V - c)}} + d$',
            'Gaussian': r'$\Delta\mathrm{Vol}=a \cdot e^{-\frac{(\Delta V - b)^2}{2c^2}} + d$',
            'Arctangent': r'$\Delta\mathrm{Vol}=a \cdot \arctan(b \cdot \Delta V) + c$',
            'Rational': r'$\Delta\mathrm{Vol}=\frac{a \cdot \Delta V + b}{c \cdot |\Delta V| + d} \cdot \operatorname{sign}(\Delta V)$',
            'Damped Sinusoidal': r'$\Delta\mathrm{Vol}=a \cdot \sin(b \cdot \Delta V) \cdot e^{-c \cdot |\Delta V|} + d$',
            'Piecewise Linear': 'ΔVol=a·ΔV + b  (ΔV ≥ 0);  c·ΔV + d  (ΔV < 0)',
            'Log-Quadratic': r'$\Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + c$',
            'Weibull': r'$\Delta\mathrm{Vol}=a \cdot \left(1 - e^{-\left(\frac{|\Delta V|}{b}\right)^c}\right) \cdot \operatorname{sign}(\Delta V) + d$',
        }


class ModelFitter:
    """Fits regression models to data and calculates metrics"""
    
    def __init__(self):
        self.regression_models = RegressionModels()
        self.models = self.regression_models.get_all_models()
        self.results = {}
    
    def fit_all_models(self, X, y, max_iterations=10000):
        print("\nFitting models...")
        
        for name, (model_func, p0) in self.models.items():
            try:
                params, _ = curve_fit(model_func, X, y, p0=p0, maxfev=max_iterations)
                y_pred = model_func(X, *params)
                
                rmse = self._calculate_rmse(y, y_pred)
                r2 = self._calculate_r2(y, y_pred)
                mae = self._calculate_mae(y, y_pred)
                
                self.results[name] = {
                    'params': params,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'y_pred': y_pred,
                    'model_func': model_func
                }
                
            except Exception as e:
                self.results[name] = None
        
        return self.results

    def _calculate_rmse(self, y_true, y_pred):
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_r2(self, y_true, y_pred):
        """R-squared (coefficient of determination)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def _calculate_mae(self, y_true, y_pred):
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def get_best_model(self, metric='rmse'):
        """Returns the best performing model based on specified metric"""
        valid_results = {k: v for k, v in self.results.items() if v is not None}
        
        if not valid_results:
            return None, None
        
        if metric in ['rmse', 'mae']:
            best_model = min(valid_results.items(), key=lambda x: x[1][metric])
        elif metric == 'r2':
            best_model = max(valid_results.items(), key=lambda x: x[1][metric])
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return best_model
    
    def generate_smooth_curve(self, X, model_name, num_points=300):
        """Generates smooth curve for plotting"""
        if model_name not in self.results or self.results[model_name] is None:
            return None, None
        
        result = self.results[model_name]
        X_smooth = np.linspace(X.min(), X.max(), num_points)
        y_smooth = result['model_func'](X_smooth, *result['params'])
        
        return X_smooth, y_smooth


if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    X_test = np.random.randn(100) * 1e6
    y_test = 0.5e-9 * X_test + 0.001 + np.random.randn(100) * 0.01
    
    fitter = ModelFitter()
    results = fitter.fit_all_models(X_test, y_test)
    
    best_name, best_result = fitter.get_best_model('rmse')
    print(f"\nBest model: {best_name} (RMSE: {best_result['rmse']:.6f})")