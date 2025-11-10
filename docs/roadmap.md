# QuantLab Research Roadmap Review

This document consolidates the prospective modelling and analytics scope for QuantLab **and** captures the review notes from the latest assessment of that scope. The intent is to keep the aspirational backlog while surfacing near-term actions, sequencing considerations, and observable gaps so implementation workstreams can be prioritised and tracked.

## Executive Summary

- **Breadth:** The roadmap spans twelve domains ranging from classical supervised learning to optimisation, risk, and numerical methods. The coverage is comprehensive, but several themes overlap (e.g., Kalman filtering appears in multiple sections) and the lack of prioritisation obscures what should be tackled first.
- **Depth:** Most entries are high-level bullet points without references to current repository capabilities, datasets, or infrastructure, making it difficult to estimate effort or identify prerequisite work.
- **Actionability:** Immediate next steps are not specified. Introducing status, priority, and ownership metadata will make the document more operational.

### Quick Triage Matrix

| Area | Status | Priority | Notes |
| --- | --- | --- | --- |
| Foundational supervised learning (OLS, GLMs) | Not started | High | Forms baseline for multiple downstream models; needs data audit and evaluation framework. |
| Tree & ensemble methods | Not started | Medium | Useful for benchmark accuracy; depends on feature engineering utilities. |
| Time-series volatility (GARCH family) | Not started | High | Directly tied to trading risk; requires time-series data cleaning pipelines. |
| Market microstructure models | Not started | Medium | Requires order book simulation infrastructure and event timestamp alignment. |
| Optimisation & control | Not started | Medium | Blocked until portfolio datasets and constraints are catalogued. |
| Risk analytics (VaR / CVaR) | Not started | High | Critical for compliance; needs integration with reporting layer. |

### Immediate Recommendations

1. **Instrument the backlog:** Add issue tracker entries that map the bullet lists below into actionable tickets with clear deliverables, data sources, and validation strategies.
2. **Define prerequisites:** Capture shared infrastructure requirements (feature store, backtesting harness, evaluation metrics) to avoid redundant implementation.
3. **Stage delivery:** Propose phased releases (Foundational ML → Time-series core → Risk analytics) so stakeholders can see value incrementally.
4. **Document current state:** Add a short subsection per topic indicating existing assets or knowledge gaps, even if the status is "not started".

The remainder of this document retains the full backlog for completeness.

## 1. Machine Learning & AI
### 1.1 Supervised Learning (Tabular / Numeric)
- Ordinary least squares (OLS) regression
- Ridge regression
- LASSO / Elastic Net
- Generalised linear models (GLMs):
  - Logistic regression
  - Probit regression
  - Poisson / negative binomial regression
- Support Vector Machines (SVMs) for classification & regression
- k-Nearest Neighbours (kNN)
- Naive Bayes classifiers

### 1.2 Tree & Ensemble Methods
- Decision trees
- Random forests
- Gradient boosting machines:
  - XGBoost-style models
  - LightGBM / CatBoost–type ideas
- Bagging / bootstrap aggregating
- Stacking / model ensembling (meta-models combining many base learners)

### 1.3 Neural Networks / Deep Learning
- Feed-forward fully connected networks (MLPs)
- 1D convolutional neural networks (CNNs) for time-series / signals
- Recurrent neural networks (RNNs)
  - LSTM
  - GRU
- Attention mechanisms / Transformer-style architectures (for sequences & text)
- Autoencoders (for dimensionality reduction / anomaly detection)

### 1.4 Probabilistic / Bayesian ML
- Bayesian linear and logistic regression
- Gaussian process regression / classification
- Bayesian hierarchical models (multi-level models)
- Bayesian model averaging

### 1.5 Unsupervised Learning
- Clustering:
  - k-means
  - Gaussian mixture models (GMM)
  - Hierarchical clustering
  - Spectral clustering
- Dimensionality reduction:
  - Principal Component Analysis (PCA)
  - Kernel PCA
  - Factor analysis
  - Independent Component Analysis (ICA)
  - t-SNE / UMAP (for exploration/visualisation)

### 1.6 Anomaly / Outlier Detection
- Robust statistics (robust z-scores, median/MAD, etc.)
- One-class SVM
- Isolation forest
- Robust PCA / low-rank + sparse decomposition

### 1.7 Text & Alternative Data (NLP Tools)
- Bag-of-words, n-grams
- TF–IDF
- Topic models (e.g. LDA)
- Sentiment models (logistic, LSTM, transformer-based)
- Word / sentence embeddings

### 1.8 Reinforcement Learning & Bandits
- Multi-armed bandits (ε-greedy, UCB, Thompson sampling)
- Tabular Q-learning
- Deep Q-learning (DQN-style ideas)
- Policy gradient methods (REINFORCE)
- Actor–critic / advantage actor–critic (A2C/A3C)
- Approximate dynamic programming (fitted value iteration, etc.)

## 2. Time-Series & Econometrics
### 2.1 Linear Time-Series Models
- AR, MA, ARMA, ARIMA
- SARIMA (seasonal ARIMA)
- ARFIMA (long memory)

### 2.2 Volatility Models
- GARCH
- EGARCH
- GJR-GARCH (asymmetric)
- Multivariate GARCH (e.g. DCC-GARCH, BEKK)

### 2.3 Multivariate Time-Series
- VAR (Vector Autoregression)
- VARMA
- VECM (Vector Error-Correction Models) for cointegrated series

### 2.4 State-Space & Regime Models
- State-space models (linear & non-linear)
- Kalman filter / smoother
- Extended / Unscented Kalman filters
- Hidden Markov Models (HMMs)
- Markov regime-switching models

## 3. General Statistics & Inference
### 3.1 Classical Regression & Inference
- Linear regression with full inferential machinery
- GLMs (logistic, probit, Poisson, negative binomial, etc.)
- Hypothesis testing (t-tests, F-tests, likelihood ratio tests)
- Nonparametric tests (rank-based, etc.)
- Multiple testing control:
  - Bonferroni
  - False Discovery Rate (Benjamini–Hochberg, etc.)

### 3.2 Nonparametric & Flexible Models
- Kernel density estimation
- Kernel regression / Nadaraya–Watson
- Local regression (LOESS / LOWESS)
- Splines (B-splines, smoothing splines)
- Quantile regression

### 3.3 Bayesian Statistics
- Priors/posteriors, conjugate priors
- Hierarchical / multi-level models
- Markov Chain Monte Carlo (MCMC):
  - Metropolis–Hastings
  - Gibbs sampling
  - Hamiltonian Monte Carlo (HMC) / NUTS
- Variational inference

## 4. Stochastic Processes & Mathematical Finance
### 4.1 Continuous-Time Processes
- Brownian motion / Wiener process
- Lévy processes (with jumps)
- Ornstein–Uhlenbeck (OU) processes (for mean reversion)
- CIR (Cox–Ingersoll–Ross) process (e.g. rates, variance)
- Jump-diffusion models (Merton, Kou, etc.)
- Stochastic volatility models (e.g. Heston)

### 4.2 Stochastic Calculus
- Itô calculus (Itô's lemma, stochastic integrals)
- Martingales and stopping times
- First-passage / hitting-time analysis

### 4.3 Dependence & Extremes
- Copulas:
  - Gaussian copula
  - t-copula
  - Archimedean copulas (Clayton, Gumbel, etc.)
- Extreme Value Theory (EVT):
  - Block maxima (GEV distribution)
  - Peaks-over-threshold (POT, Generalised Pareto)

## 5. Cointegration, Statistical Arbitrage & Spreads
- Unit-root tests (ADF, Phillips–Perron, etc.)
- Cointegration tests:
  - Engle–Granger
  - Johansen
- Error-correction models (ECM / VECM)
- Spread modelling with:
  - AR(1)
  - OU processes
  - VECM between multiple legs
- Half-life estimation for mean reversion

## 6. Market Microstructure & Event Processes
### 6.1 Point Processes
- Poisson processes
- Self-exciting Hawkes processes (univariate & multivariate)

### 6.2 Order Book & Trade Models
- Queueing models for limit order book depths
- Survival / duration models:
  - Hazard models (Cox proportional hazards, parametric survival models)
- Fill probability models (logistic/probit for "will this limit order fill in X seconds?")
- Price impact models:
  - Almgren–Chriss–style optimal execution
  - Empirical power laws for temporary/permanent impact

## 7. Optimisation, Control & Numerics
### 7.1 Portfolio & Risk Optimisation
- Mean–variance optimisation (Markowitz)
- Quadratic programming (QP)
- Linear programming (LP) for constraints / simpler cases
- Nonlinear programming (general constrained optimisation)
- Risk-parity constructions
- CVaR (Conditional VaR) optimisation
- Kelly criterion / fractional Kelly for leverage decisions
- Robust optimisation:
  - Worst-case / uncertainty sets
  - Distributionally robust optimisation

### 7.2 Numerical Optimisation Algorithms
- Gradient descent, stochastic gradient descent (SGD)
- Momentum, Nesterov's acceleration
- Adam / RMSProp / Adagrad (for ML training)
- Newton / quasi-Newton methods (BFGS, L-BFGS)
- Coordinate descent
- Projected gradient methods (for constrained problems)
- Mixed-integer programming (MIP) for discrete decisions

### 7.3 Dynamic Programming & Control
- Bellman equations
- Backward induction for finite-horizon problems
- Approximate dynamic programming (ADP)
- Reinforcement learning methods (Q-learning, policy gradient, etc.) for execution/scheduling

## 8. Linear Algebra, Matrices & Large-Scale Statistics
- Eigenvalue / eigenvector decomposition
- Singular Value Decomposition (SVD)
- PCA (again, but from linear algebra perspective)
- Randomised SVD / low-rank approximations
- Covariance / correlation matrix estimation
- Shrinkage estimators (e.g. Ledoit–Wolf)
- Factor-model-based covariance estimation
- Precision matrix estimation:
  - Graphical Lasso (sparse inverse covariance)
- Iterative solvers:
  - Conjugate Gradient
  - GMRES, BiCGSTAB, etc.

## 9. Signal Processing & Spectral Methods
- Discrete Fourier Transform (DFT), FFT
- Spectral density estimation (periodogram, smoothed periodograms)
- Wavelet transforms (discrete wavelet transform, wavelet denoising)
- Digital filters:
  - Moving averages (simple, exponential, double/triple EMA)
  - FIR/IIR filters
- Kalman filtering (signal-processing perspective)
- Particle filters (for non-linear/non-Gaussian state-space models)

## 10. Risk Management & Performance Analytics
- Value-at-Risk (VaR):
  - Parametric (variance–covariance)
  - Historical
  - Monte Carlo
- Expected Shortfall / CVaR
- Stress testing & scenario analysis
- Factor-based risk models (Barra-style frameworks, custom factor sets)
- Performance attribution:
  - Brinson-type attribution
  - Factor vs idiosyncratic P&L breakdown
- Drawdown and tail-risk metrics (max drawdown, tail ratios, etc.)

## 11. Information Theory & Model Selection
- Entropy, cross-entropy
- Mutual information (feature selection / dependency)
- Kullback–Leibler (KL) divergence
- Jensen–Shannon divergence
- Minimum Description Length (MDL) principles
- Information criteria:
  - AIC, BIC, HQIC, etc.

## 12. Miscellaneous Mathematical Tools
- Convex analysis (subgradients, duality, KKT conditions)
- Combinatorics and graph theory (network structures, clustering, propagation of shocks)
- Numerical integration & Monte Carlo:
  - Standard Monte Carlo
  - Importance sampling
  - Quasi–Monte Carlo (low-discrepancy sequences)

