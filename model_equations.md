# Model Equations

This file contains the LaTeX equations for all available models.

### Linear
\[ \Delta\mathrm{Vol}=a \cdot \Delta V + b \]

### Quadratic
\[ \Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \Delta V + c \]

### Cubic
\[ \Delta\mathrm{Vol}=a \cdot (\Delta V)^3 + b \cdot (\Delta V)^2 + c \cdot \Delta V + d \]

### Square Root
\[ \Delta\mathrm{Vol}=a \cdot \sqrt{|\Delta V|} \cdot \operatorname{sign}(\Delta V) + b \]

### Logarithmic
\[ \Delta\mathrm{Vol}=a \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + b \]

### Regime Mixture
\[ \Delta\mathrm{Vol}=a \cdot \Delta V + b \cdot (\Delta V)^3 + c \]

### Exponential Decay
\[ \Delta\mathrm{Vol}=a \cdot (1 - e^{-b \cdot |\Delta V|}) \cdot \operatorname{sign}(\Delta V) + c \]

### Inverse
\[ \Delta\mathrm{Vol}=\frac{a}{|\Delta V| + b} \cdot \operatorname{sign}(\Delta V) + c \]

### Power Law
\[ \Delta\mathrm{Vol}=a \cdot (|\Delta V| + 1)^b \cdot \operatorname{sign}(\Delta V) + c \]

### Hyperbolic Tangent
\[ \Delta\mathrm{Vol}=a \cdot \tanh(b \cdot \Delta V) + c \]

### Sigmoid
\[ \Delta\mathrm{Vol}=\frac{a}{1 + e^{-b(\Delta V - c)}} + d \]

### Gaussian
\[ \Delta\mathrm{Vol}=a \cdot e^{-\frac{(\Delta V - b)^2}{2c^2}} + d \]

### Arctangent
\[ \Delta\mathrm{Vol}=a \cdot \arctan(b \cdot \Delta V) + c \]

### Rational
\[ \Delta\mathrm{Vol}=\frac{a \cdot \Delta V + b}{c \cdot |\Delta V| + d} \cdot \operatorname{sign}(\Delta V) \]

### Damped Sinusoidal
\[ \Delta\mathrm{Vol}=a \cdot \sin(b \cdot \Delta V) \cdot e^{-c \cdot |\Delta V|} + d \]

### Piecewise Linear
ΔVol=a·ΔV + b  (ΔV ≥ 0);  c·ΔV + d  (ΔV < 0)

### Log-Quadratic
\[ \Delta\mathrm{Vol}=a \cdot (\Delta V)^2 + b \cdot \log(|\Delta V| + 1) \cdot \operatorname{sign}(\Delta V) + c \]

### Weibull
\[ \Delta\mathrm{Vol}=a \cdot \left(1 - e^{-\left(\frac{|\Delta V|}{b}\right)^c}\right) \cdot \operatorname{sign}(\Delta V) + d \]