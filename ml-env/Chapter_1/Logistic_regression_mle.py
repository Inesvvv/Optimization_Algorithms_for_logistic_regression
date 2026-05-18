"""
Created 07/02/2026
Implementation of MLE 
"""

import numpy as np
from scipy.optimize import minimize

# defining logistic function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# defining negative log likelihood 
def neg_log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)

    # numerical stability
    eps = 1e-9
    p = np.clip(p, eps, 1 - eps)

    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

# defining the gradient (makes optimization faster and cleaner)
def grad_neg_log_likelihood(beta, X, y):
    z = X @ beta
    p = sigmoid(z)
    return X.T @ (p - y)

#defining toy dataset 
np.random.seed(0)

n = 200
d = 2

X = np.random.randn(n, d)
true_beta = np.array([1.5, -2.0])

p = sigmoid(X @ true_beta)
y = np.random.binomial(1, p)

#running optimizer
beta0 = np.zeros(d)

res = minimize(
    neg_log_likelihood,
    beta0,
    args=(X, y),
    jac=grad_neg_log_likelihood,
    method="BFGS"
)

beta_hat = res.x
print("True beta:", true_beta)
print("Estimated beta:", beta_hat)