"""
Created 08/02/2026
This file compares OLS and LASSO 
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# =============================================
# 1. Basic comparison (5 features, 2 irrelevant)
# =============================================
np.random.seed(0)

n = 200  # nb of samples
d = 5    # nb of features

X = np.random.randn(n, d)
true_beta = np.array([1.5, -2.0, 0.0, 3.0, 0.0])  # some coefficients are zero
noise = np.random.randn(n) * 0.5

y = X @ true_beta + noise 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# OLS (Ordinary Least Squares)
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)

# LASSO (L1 Regularization)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

print("=" * 55)
print("1. BASIC COMPARISON (5 features, 2 irrelevant)")
print("=" * 55)
print(f"{'True beta:':<20} {true_beta}")
print(f"{'OLS coefficients:':<20} {np.round(ols_model.coef_, 4)}")
print(f"{'LASSO coefficients:':<20} {np.round(lasso_model.coef_, 4)}")

# Prediction error on test set
ols_mse = mean_squared_error(y_test, ols_model.predict(X_test))
lasso_mse = mean_squared_error(y_test, lasso_model.predict(X_test))
print(f"\nTest MSE  -  OLS: {ols_mse:.4f}  |  LASSO: {lasso_mse:.4f}")

# =============================================
# 2. High-dimensional setting (20 features, 15 irrelevant)
# =============================================
np.random.seed(1)

n = 200
d = 20  # much more features now

X_hd = np.random.randn(n, d)
true_beta_hd = np.zeros(d)
true_beta_hd[:5] = [1.5, -2.0, 0.5, 3.0, -1.0]  # only 5 out of 20 are relevant
noise_hd = np.random.randn(n) * 0.5

y_hd = X_hd @ true_beta_hd + noise_hd

X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(
    X_hd, y_hd, test_size=0.3, random_state=42
)

ols_hd = LinearRegression().fit(X_train_hd, y_train_hd)
lasso_hd = Lasso(alpha=0.1).fit(X_train_hd, y_train_hd)

print("\n" + "=" * 50)
print("2. HIGH-DIMENSIONAL (20 features, only 5 relevant)")
print("=" * 50)
print(f"{'True beta:':<20} {true_beta_hd}")
print(f"{'OLS coefficients:':<20} {np.round(ols_hd.coef_, 4)}")
print(f"{'LASSO coefficients:':<20} {np.round(lasso_hd.coef_, 4)}")

ols_hd_mse = mean_squared_error(y_test_hd, ols_hd.predict(X_test_hd))
lasso_hd_mse = mean_squared_error(y_test_hd, lasso_hd.predict(X_test_hd))
print(f"\nTest MSE  -  OLS: {ols_hd_mse:.4f}  |  LASSO: {lasso_hd_mse:.4f}")

ols_nonzero = np.sum(np.abs(ols_hd.coef_) > 1e-4)
lasso_nonzero = np.sum(np.abs(lasso_hd.coef_) > 1e-4)
print(f"Non-zero coefs  -  OLS: {ols_nonzero}/{d}  |  LASSO: {lasso_nonzero}/{d}")

# =============================================
# 3. Effect of varying LASSO alpha
# =============================================
print("\n" + "=" * 50)
print("3. EFFECT OF ALPHA ON LASSO (high-dim dataset)")
print("=" * 50)
print(f"{'Alpha':<10} {'Non-zero coefs':<18} {'Test MSE':<12}")
print("-" * 50)

for alpha in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    lasso_temp = Lasso(alpha=alpha).fit(X_train_hd, y_train_hd)
    mse = mean_squared_error(y_test_hd, lasso_temp.predict(X_test_hd))
    nonzero = np.sum(np.abs(lasso_temp.coef_) > 1e-4)
    print(f"{alpha:<10} {nonzero:<18} {mse:<12.4f}")

# CONCLUSIONS to remember:
# LASSO shrinks some coefficients to exactly 0 (feature selection)
# Higher alpha = more regularization = more zeros = simpler model
# Too high alpha = underfitting, too low alpha = behaves like OLS