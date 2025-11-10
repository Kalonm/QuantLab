# Bayesian Inference Examples

This folder contains lightweight walkthroughs demonstrating how to use the
Bayesian tooling introduced in QuantLab.

## Linear Regression

```python
import numpy as np
from src.models.bayesian import BayesianLinearRegression

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 2))
true_w = np.array([1.5, -2.0])
y = X @ true_w + rng.normal(scale=0.3, size=100)

prior_mean = np.zeros(2)
prior_cov = np.eye(2)
model = BayesianLinearRegression(prior_mean, prior_cov, noise_variance=0.3**2)
model.fit_conjugate(X, y)
print(model.posterior_mean)
```

## Logistic Regression

```python
import numpy as np
from src.models.bayesian import BayesianLogisticRegression

rng = np.random.default_rng(1)
X = rng.normal(size=(200, 3))
true_w = np.array([0.7, -1.2, 0.5])
logits = X @ true_w
probs = 1 / (1 + np.exp(-logits))
y = rng.binomial(1, probs)

prior_mean = np.zeros(3)
prior_cov = np.eye(3)
model = BayesianLogisticRegression(prior_mean, prior_cov)
model.fit_variational(X, y)
print(model.posterior_mean)
```

## Posterior Sampling

```python
from src.inference import metropolis_hastings

logp = model._log_posterior(X, y)
result = metropolis_hastings(logp, model.posterior_mean, n_samples=2000)
print(result.acceptance_rate)
```
