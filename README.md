# Random Feature Representation Boosting

This repository contains code from the paper ["Random Feature Representation Boosting"](https://arxiv.org/abs/2501.18283), which intorduces RFRBoost, a novel method for constructing deep residual random feature neural networks using boosting theory.

## Usage Example: Toy Regression Problem

Here's a simple example demonstrating how to use both the `GradientRFRBoostRegressor` and `GreedyRFRBoostRegressor` on a toy regression problem:

```python
import numpy as np
import torch
from models.random_feature_representation_boosting import GradientRFRBoostRegressor, GreedyRFRBoostRegressor
from sklearn.model_selection import train_test_split

# Generate toy data
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
n_samples = 2000
X = torch.randn(n_samples, 2)
y = X[:, 0] + X[:, 1]**2 + 0.1 * torch.randn(n_samples)
y = y.unsqueeze(1)  # Reshape y to (n_samples, 1) for regression

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed
)

# 1. GradientRFRBoostRegressor
gradient_rfrboost = GradientRFRBoostRegressor(
    in_dim=2,
    out_dim=1,
    hidden_dim=64,
    n_layers=3,
    boost_lr=0.5,
    feature_type="SWIM",
    upscale_type="identity",
    l2_reg=0.001,
    l2_ghat=0.001
)

# Train and evaluate
gradient_rfrboost.fit(X_train, y_train)
y_pred_train_gradient = gradient_rfrboost(X_train)
y_pred_test_gradient = gradient_rfrboost(X_test)

train_rmse_gradient = torch.sqrt(torch.mean((y_pred_train_gradient - y_train)**2))
test_rmse_gradient = torch.sqrt(torch.mean((y_pred_test_gradient - y_test)**2))

print("GradientRFRBoostRegressor:")
print(f"  Train RMSE: {train_rmse_gradient:.4f}")
print(f"  Test RMSE:  {test_rmse_gradient:.4f}")

# 2. GreedyRFRBoostRegressor
greedy_rfrboost = GreedyRFRBoostRegressor(
    in_dim=2,
    out_dim=1,
    hidden_dim=64,
    n_layers=3,
    boost_lr=0.5,
    feature_type="SWIM",
    upscale_type="identity",
    sandwich_solver="dense",
    l2_reg=0.001,
    l2_ghat=0.001
)

# Train and evaluate
greedy_rfrboost.fit(X_train, y_train)
y_pred_train_greedy = greedy_rfrboost(X_train)
y_pred_test_greedy = greedy_rfrboost(X_test)

train_rmse_greedy = torch.sqrt(torch.mean((y_pred_train_greedy - y_train)**2))
test_rmse_greedy = torch.sqrt(torch.mean((y_pred_test_greedy - y_test)**2))

print("\nGreedyRFRBoostRegressor:")
print(f"  Train RMSE: {train_rmse_greedy:.4f}")
print(f"  Test RMSE:  {test_rmse_greedy:.4f}")

# Baseline: Naive prediction using mean
naive_train_rmse = torch.sqrt(torch.mean((y_train - y_train.mean())**2))
naive_test_rmse = torch.sqrt(torch.mean((y_test - y_train.mean())**2))

print("\nNaive (mean) baseline:")
print(f"  Train RMSE: {naive_train_rmse:.4f}")
print(f"  Test RMSE:  {naive_test_rmse:.4f}")
```

Output:

```console
GradientRFRBoostRegressor:
  Train RMSE: 0.1736
  Test RMSE:  0.2599

GreedyRFRBoostRegressor:
  Train RMSE: 0.1813
  Test RMSE:  0.2921

Naive (mean) baseline:
  Train RMSE: 1.7043
  Test RMSE:  1.7306
```

## Reproducing Experiments

The following code snippet shows how to reproduce the experiments from the paper. In this particular example we show how to run the experiments on the first three datasets for `RFRBoost` and the baseline `Logistic Regression`.

```python
# Example for running experiments
!python classification_param_specs.py \
    --models RFRBoost_ID_batchnormFalse LogisticRegression \
    --dataset_indices 0 1 2 \
    --save_dir /home/nikita/Code/random-feature-boosting/save/OpenMLClassification/ \
    --n_optuna_trials 100 \
    --k_folds 5 \
    --cv_seed 42
```
