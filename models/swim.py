from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from models.base import FittableModule


class SWIMLayer(FittableModule):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 activation: nn.Module = nn.Tanh(),
                 epsilon: float = 0.01,
                 sampling_method: Literal['uniform', 'gradient'] = 'gradient',
                 c: float = 0.5
                 ):
        """Dense MLP layer with pair sampled weights (uniform or gradient-weighted).

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            activation (nn.Module): Activation function.
            epsilon (float): Small constant to avoid division by zero.
            sampling_method (str): Pair sampling method. Uniform or gradient-weighted.
            c (float): Scaling factor, ie the value of (A x_i + b)_i in the SWIM paper.
        """
        super(SWIMLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.epsilon = epsilon
        self.sampling_method = sampling_method
        self.c = c

        self.dense = nn.Linear(in_dim, out_dim)


    def fit(self, X: Tensor, y: Tensor):
        """Given forward-propagated training data X at the previous 
        hidden layer, and supervised target labels y, fit the weights
        iteratively by letting rows of the weight matrix be given by
        pairs of samples from X. See paper for more details.

        Args:
            X (Tensor): Forward-propagated activations of training data, shape (N, D).
            y (Tensor): Training labels, shape (N, p).
        """
        self.to(X.device)
        if self.out_dim == 0:
            return self
        
        with torch.no_grad():
            N, D = X.shape
            dtype = X.dtype
            device = X.device

            #obtain pair indices
            n = 5*N
            idx1 = torch.arange(0, n, dtype=torch.int32, device=device) % N
            delta = torch.randint(1, N, size=(n,), dtype=torch.int32, device=device)
            idx2 = (idx1 + delta) % N
            dx = X[idx2] - X[idx1]
            dists = torch.linalg.norm(dx, axis=1, keepdims=True).clamp(min=self.epsilon)
            
            if self.sampling_method=="gradient":
                #calculate 'gradients'
                dy = y[idx2] - y[idx1]
                y_norm = torch.linalg.norm(dy, axis=1, keepdims=True) #NOTE 2023 paper uses ord=inf instead of ord=2
                grad = (y_norm / dists).reshape(-1) 
                p = grad/grad.sum()
                torch.set_printoptions(precision=20)
            elif self.sampling_method=="uniform":
                p = torch.ones(n, dtype=dtype, device=device) / n
            else:
                raise ValueError(f"sampling_method must be 'uniform' or 'gradient'. Given: {self.sampling_method}")

            #sample pairs
            selected_idx = torch.multinomial(
                p,
                self.out_dim,
                replacement=True,
                )
            idx1 = idx1[selected_idx]
            dx = dx[selected_idx]
            dists = dists[selected_idx]

            #define weights and biases
            weights = abs(2*self.c) * dx / (dists**2)
            biases = -torch.sum(weights * X[idx1], axis=1) - self.c
            #biases = -torch.einsum('ij,ij->i', weights, X[idx1]) - self.c
            self.dense.weight.data = weights
            self.dense.bias.data = biases

        return self
    

    def forward(self, X):
        return self.activation(self.dense(X))