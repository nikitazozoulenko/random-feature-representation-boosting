from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable

import torch
from torch import Tensor


def fit_ridge_ALOOCV(
        X: Tensor,
        y: Tensor,
        alphas: List[float] = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 1],
        fit_intercept: bool = True
    ) -> Tuple[Tensor, float]:
    """
    ALOOCV for Ridge regression optimized for case D < N 
    using eigendecomposition.

    Args:
        X (Tensor): Shape (N, D).
        y (Tensor): Shape (N, p).
        alphas (List[float]): Alphas to test.
        fit_intercept (bool): Whether to fit an intercept.
    
    Returns:
        Tuple[Tensor, Tensor, float]: Coefficients beta (D, p), intercept (1, p), and best alpha
    """
    if y.ndim == 1:
        y = y.unsqueeze(1)
    if X.ndim == 1:
        X = X.unsqueeze(1)

    if fit_intercept:
        X_mean = X.mean(dim=0, keepdim=True)
        y_mean = y.mean(dim=0, keepdim=True)
        X = X - X.mean(dim=0)
        y = y - y.mean(dim=0)

    N = X.size(0)
    
    # Convert alphas to tensor on the same device as X
    alphas = torch.tensor(alphas, device=X.device, dtype=X.dtype) * N  # Shape (n_alphas,)
    #eigvecs, eigvals, _ = torch.linalg.svd(X.T @ X, full_matrices=False)  # eigvals: (D,), eigvecs: (D, D) NOTE linalg.eigh throws errors for cuda due to cuBLAS solver. date=2024
    eigvals, eigvecs = torch.linalg.eigh(X.T @ X)  # eigvals: (D,), eigvecs: (D, D)
    
    # Project y onto the eigenspace
    XTy = X.T @ y  # Shape (D, p)
    eigvecs_T_XTy = (eigvecs.T @ XTy)[None, :, :]  # Shape (1, D, p)

    # Compute denominators for all alphas
    denom = eigvals[None, :, None] + alphas[:, None, None] # Shape (n_alphas, D, 1)
    beta_ridge = (eigvecs @ (eigvecs_T_XTy / denom)).permute(1,0,2)  # Shape (D, n_alphas, p)
    
    # Compute predictions and residuals
    y_pred = torch.tensordot(X, beta_ridge, dims=1)  # Shape (N, n_alphas, p)
    residuals = y[:, None, :] - y_pred  # Shape (N, n_alphas, p)

    # Compute hat matrix diagonals (same for all output dimensions)
    U = X @ eigvecs  # Shape (N, D)
    H_diag = (U**2 @ (1 / denom.squeeze(-1)).T)  # Shape (N, n_alphas)

    # Compute LOOCV errors across all dimensions
    errors = ((residuals / (1 - H_diag.unsqueeze(-1))) ** 2).mean(dim=(0, 2))  # Shape (n_alphas,)

    # Find the best alpha
    best_idx = errors.argmin()
    best_alpha = alphas[best_idx].item() / N
    beta_optimal = beta_ridge[:, best_idx, :]  # Shape (D, p)

    if fit_intercept:
        intercept = y_mean - X_mean @ beta_optimal
    else:
        intercept = torch.zeros_like(y_mean, device=X.device, dtype=X.dtype)
    return beta_optimal, intercept, best_alpha