from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def sandwiched_LS_scalar(
        R: Tensor, 
        W: Tensor, 
        X: Tensor,
        l2_reg: float = 0.01,
        ):
    """Solves the sandwiched least squares problem for scalar Delta:
    Delta = argmin_{Delta} sum_i ||R_i - W' Delta X_i||^2 + lambda_reg ||Delta||^2

    Args:
        R (Tensor): Shape (N, d)
        W (Tensor): Shape (D, d)
        X (Tensor): Shape (N, D)
        l2_reg (float): L2 regularization parameter.
    
    Returns:
        Delta (Tensor): Shape (,)
    """
    N = X.shape[0]
    XW = X @ W
    numerator = torch.sum(R * XW / N)
    denominator = torch.sum(XW * XW / N)
    Delta = numerator / (denominator + l2_reg)
    return Delta


def sandwiched_LS_diag(
        R: Tensor, 
        W: Tensor, 
        X: Tensor,
        l2_reg: float = 0.01,
        ):
    """Solves the sandwiched least squares problem for diagonal matrices Delta:
    Delta = argmin_{Delta} sum_i ||R_i - W' Delta X_i||^2 + lambda_reg ||Delta||^2

    Args:
        R (Tensor): Shape (N, d)
        W (Tensor): Shape (D, d)
        X (Tensor): Shape (N, D)
        l2_reg (float): L2 regularization parameter.
    
    Returns:
        Delta (Tensor): Shape (D,)
    """
    N, D = X.shape
    b = torch.mean( (R @ W.T) * X, axis=0)
    A = (W @ W.T) * (X.T @ X) / N
    eye = torch.eye(D, dtype=A.dtype, device=A.device)
    Delta = torch.linalg.solve(A + l2_reg * eye, b)
    return Delta


def sandwiched_LS_dense(
        R: Tensor, 
        W: Tensor, 
        X: Tensor,
        l2_reg: float = 0.01,
        ):
    """Solves the sandwiched least squares problem for dense matrices Delta:
    Delta = argmin_{Delta} sum_i ||R_i - W' Delta' X_i||^2 + lambda_reg ||Delta||_F^2
          = argmin_{Delta}       ||X Delta W - R||_F^2 + lambda_reg ||Delta||^2

    Args:
        R (Tensor): Shape (N, d)
        W (Tensor): Shape (D, d)
        X (Tensor): Shape (N, p)
        l2_reg (float): L2 regularization parameter.
    
    Returns:
        Delta (Tensor): Shape (p, D)
    """
    N = X.size(0)
    # NOTE torch.linalg.eigh sometimes throws errors for cuda due to cuBLAS solver. date=2024
    # U, SW, _ = torch.linalg.svd(W @ W.T, full_matrices=False) # shape (D, d), (d,)
    # V, SX, _ = torch.linalg.svd(X.T @ X, full_matrices=False) # shape (p, p), (p,)
    SW, U = torch.linalg.eigh(W @ W.T)  # shape (d,), (D, d)
    SX, V = torch.linalg.eigh(X.T @ X)  # shape (p,), (p, p)
    #Delta = (U.T @ W @ R.T @ (X/N) @ V) / (l2_reg + SW[:, None]*SX[None, :])
    Delta = torch.linalg.multi_dot( (U.T, W, R.T, X, V) ) / (N*l2_reg + SW[:, None]*SX[None, :])
    Delta = U @ Delta @ V.T
    return Delta.T