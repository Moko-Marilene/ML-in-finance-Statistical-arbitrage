# Deep Learning Statistical Arbitrage –

Implementation of statistical arbitrage strategies based on the paper *"Deep Learning Statistical Arbitrage"* by Guijarro-Ordonez, Pelger & Zanotti (Stanford University).

## Objective

Build and compare statistical arbitrage strategies on a large universe of assets, constructing long/short portfolios neutral to common risk factors without manual pair selection.

## Methodology

### 1. Factor Models & Arbitrage Portfolios

Construction of tradable long/short portfolios from residuals of conditional factor models:
- Observable factors (Fama-French)
- Statistical factors (PCA)
- Conditional factors (IPCA)

The residual vector is projected as:  
`ε = (I - βᵀ w^F) R` → zero exposure to systematic risk.

### 2. Signal Extraction (3 approaches compared)

| Approach | Description |
|----------|-------------|
| **Parametric (OU + Threshold)** | Ornstein-Uhlenbeck process estimation + threshold rule based on standardized deviation |
| **Fourier + FFN** | FFT decomposition of cumulated residuals into frequency coefficients + Feed-Forward Network |
| **End-to-end Deep Learning** | CNN + Transformer architectures learning latent features directly |

The Fourier representation captures mean-reversion patterns across multiple frequencies, outperforming the parametric OU baseline.

### 3. Allocation & Optimization

- A non-parametric FFN converts the signal into long/short position weights
- Optimized on **Sharpe ratio maximization**
- Constraints: leverage constraint (`||w||₁ = 1`) + transaction costs

