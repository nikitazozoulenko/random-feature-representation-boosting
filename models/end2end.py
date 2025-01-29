from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor

from models.base import FittableModule


######################################
##### End2End MLPResNet       #####
######################################


class End2EndMLPResNet(FittableModule):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 bottleneck_dim: int,
                 out_dim: int,
                 n_blocks: int,
                 activation: nn.Module = nn.ReLU(),
                 loss: Literal["mse", "cce", "bce"] = "mse",
                 lr: float = 1e-3,
                 end_lr_factor: float = 1e-2,
                 n_epochs: int = 10,
                 weight_decay: float = 1e-5,
                 batch_size: int = 64,
                 upsample: bool = True,
                 ):
        """End-to-end trainer for residual networks using Adam optimizer 
        with a CosineAnnealingLR scheduler with end_lr = lr * end_lr_factor.
        
        Args:
            in_dim (int): Input dimension.
            hidden_dim (int): Dimension of the hidden layers.
            bottleneck_dim (int): Dimension of the bottleneck layer.
            out_dim (int): Output dimension.
            n_blocks (int): Number of residual blocks.
            activation (nn.Module): Activation function.
            loss (nn.Module): Loss function.
            lr (float): Learning rate for Adam optimizer.
            end_lr_factor (float): Factor for the end learning rate in the scheduler.
            n_epochs (int): Number of training epochs.
            weight_decay (float): Weight decay for Adam optimizer.
            batch_size (int): Batch size for training.
            upsample (bool): Whether to use an upsample layer in the beginning.
        """
        super(End2EndMLPResNet, self).__init__()
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        #upsample
        if upsample:
            self.upsample = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                activation
                )
        else:
            self.upsample = nn.Identity()
            if hidden_dim != in_dim:
                raise ValueError("If upsample is False, hidden_dim must be equal to in_dim.")

        #residual blocks
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                activation,
                nn.Linear(bottleneck_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            ) for _ in range(n_blocks)
        ])
        self.output_layer = nn.Linear(hidden_dim, out_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs, eta_min = lr * end_lr_factor
            )
        
        # Loss (needs to be strings due to Optuna)
        if loss == "mse":
            self.loss = nn.functional.mse_loss
        elif loss == "cce":
            self.loss = nn.functional.cross_entropy
        elif loss == "bce":
            self.loss = nn.functional.binary_cross_entropy_with_logits
        else:
            raise ValueError(f"Unknown value of loss argument. Given: {loss}")


    def fit(self, X: Tensor, y: Tensor):
        """Trains network end to end with Adam optimizer and a tabular data loader"""
        device = X.device
        self.to(device)

        # DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
        )

        # training loop
        for epoch in tqdm(range(self.n_epochs)):
            for batch_X, batch_y in loader:
                if batch_X.size(0) < self.batch_size:
                    continue  # Skip smaller batches since i sometimes have dataset size of 257 leading to training errors with batch norm
                self.optimizer.zero_grad()
                outputs = self(batch_X)
                loss = self.loss(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        return self


    def forward(self, X: Tensor) -> Tensor:
        X = self.upsample(X)
        for block in self.residual_blocks:
            X = X + block(X)
        X = self.output_layer(X)
        return X