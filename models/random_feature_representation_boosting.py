from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable, Type
import abc

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch import Tensor
import torchmin

from models.sandwiched_least_squares import sandwiched_LS_dense, sandwiched_LS_diag, sandwiched_LS_scalar
from models.swim import SWIMLayer
from models.base import FittableModule, RidgeModule, RidgeCVModule, FittableSequential, Identity, LogisticRegression




############################################################################
################# Base classes for Random Feature Boosting #################
############################################################################

class ScaleLayer(nn.Module):
    def __init__(self, scale):
        super(ScaleLayer, self).__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def create_layer(
        in_dim: int, 
        out_dim: int, 
        layer_type: Literal["iid", "SWIM", "identity"],
        iid_scale: float = 1.0,
        SWIM_scale: float = 0.5,
        activation: nn.Module = nn.Tanh(),
        ):
    """Takes in the input and output dimensions and returns 
    a layer of the specified type."""
    if layer_type == "iid":
        layer = FittableSequential(
            nn.Linear(in_dim, out_dim), 
            ScaleLayer(iid_scale), 
            activation,
        )
    elif layer_type == "SWIM":
        layer = SWIMLayer(in_dim, out_dim, activation=activation, c=SWIM_scale)
    elif layer_type == "identity":
        layer = Identity()
    else:
        raise ValueError(f"Unknown upscale type {upscale_type}")
    return layer



class Upscale(FittableModule):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int = 128,
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 iid_scale: float = 1.0,
                 SWIM_scale: float = 0.5,
                 activation: nn.Module = nn.Tanh(),
                 ):
        self.upscale_type = upscale_type
        super(Upscale, self).__init__()
        self.upscale = create_layer(in_dim, hidden_dim, upscale_type, iid_scale, SWIM_scale, activation)
    
    def fit(self, X: Tensor, y: Tensor):
        self.upscale.fit(X, y)
        return self

    def forward(self, X: Tensor):
        return self.upscale(X)



class RandomFeatureLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



class GhatBoostingLayer(nn.Module):
    @abc.abstractmethod
    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, top_level_module: FittableModule) -> Tensor:
        """Takes in the random features, resnet representations Xt, target y, 
        and the top level module and fits the boosting layer (functional gradient), 
        and returns the gradient estimates"""

    @abc.abstractmethod
    def forward(self, F: Tensor) -> Tensor:
        """Takes in the random features and returns the gradient estimates"""



class BaseGRFRBoost(FittableModule):
    def __init__(
            self, 
            upscale: Upscale,
            top_level_modules: List[FittableModule],
            random_feature_layers: List[RandomFeatureLayer],
            ghat_boosting_layers: List[GhatBoostingLayer],
            boost_lr: float = 1.0,
            use_batchnorm: bool = True,
            freeze_top_at_t: Optional[int] = None,
            return_features: bool = False,  #logits or features
            ):
        """
        Base class for (Greedy/Gradient) Random Feature Representation Boosting.
        NOTE that we currently store all intermediary classifiers/regressors,
        for simplicity. We only use the topmost one for prediction.
        """
        super(BaseGRFRBoost, self).__init__()
        self.boost_lr = boost_lr
        self.freeze_top_at_t = freeze_top_at_t
        self.return_features = return_features

        self.upscale = upscale # simple upscale layer, same for everyone
        self.top_level_modules = nn.ModuleList(top_level_modules) # either ridge, or multiclass logistic, or binary logistic
        self.random_feature_layers = nn.ModuleList(random_feature_layers) # random features, same for everyone
        self.ghat_boosting_layers = nn.ModuleList(ghat_boosting_layers) # functional gradient boosting layers
        if not use_batchnorm:
            self.batchnorms = nn.ModuleList([Identity() for _ in range(len(ghat_boosting_layers))])
        else:
            self.batchnorms = nn.ModuleList([nn.BatchNorm1d(ghat_boosting_layers[-1].hidden_dim,
                                                            momentum=1, affine=False,
                                                            track_running_stats=False) 
                                             for _ in range(len(ghat_boosting_layers))])
            

    def fit(self, X: Tensor, y: Tensor):
        """Fits the Random Feature Representation Boosting model.
        NOTE that in the classification case, y has to be onehot for the
        multiclass case, and (N, 1) for binary classification. For regression
        y has to be (N, d)

        Args:
            X (Tensor): Input data, shape (N, in_dim)
            y (Tensor): Targets, shape (N, d) for regression,
            or onehot (N, C) for multiclass classification, 
            or (N, 1) for binary classification.
        """
        with torch.no_grad():
            X0 = X

            # upscale
            X = self.upscale.fit_transform(X0, y)          

            # Create top level regressor or classifier W_0
            self.top_level_modules[0].fit(X, y)

            for t in range(self.n_layers):
                # Step 1: Create random feature layer
                F = self.random_feature_layers[t].fit_transform(X, X0, y)
                # Step 2: Greedily or Gradient boost to minimize R(W_t, Phi_t + Delta F)
                Ghat = self.ghat_boosting_layers[t].fit_transform(F, X, y, self.top_level_modules[t])
                X = X + self.boost_lr * Ghat
                X = self.batchnorms[t](X)
                # Step 3: Learn top level classifier W_t
                if self.freeze_top_at_t is None or t < self.freeze_top_at_t:
                    self.top_level_modules[t+1].fit(X, y)
                else:
                    self.top_level_modules[t+1] = self.top_level_modules[t]

        return self


    def forward(self, X: Tensor) -> Tensor:
        """Forward pass for random feature representation boosting.
        
        Args:
            X (Tensor): Input data shape (N, in_dim)"""
        with torch.no_grad():
            #upscale
            X0 = X
            X = self.upscale(X0)
            for randfeat_layer, ghat_layer, batchnorm in zip(self.random_feature_layers, 
                                                             self.ghat_boosting_layers,
                                                             self.batchnorms):
                F = randfeat_layer(X, X0)
                Ghat = ghat_layer(F)
                X = X + self.boost_lr * Ghat
                X = batchnorm(X)
            # Top level regressor
            if self.return_features:
                return X
            else:
                return self.top_level_modules[-1](X)
        



############################################################################
#################    Random feature layer       #################
############################################################################

class RandomFeatureLayer(nn.Module, abc.ABC):
    @abc.abstractmethod
    def fit_transform(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Takes in both Xt and X0 and y and fits the random 
        feature layer and returns the random features"""

    @abc.abstractmethod
    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        """Takes in both Xt and X0 and returns the random features"""



class RandFeatLayer(RandomFeatureLayer):
    def __init__(self, 
                 in_dim: int,
                 hidden_dim: int,
                 randfeat_xt_dim: int,
                 randfeat_x0_dim: int,
                 feature_type : Literal["iid", "SWIM"],
                 iid_scale: float = 1.0,
                 SWIM_scale: float = 0.5,
                 activation: nn.Module = nn.Tanh(),
                 add_features: bool = False, #add features or concat features
                 ):
        self.hidden_dim = hidden_dim
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.add_features = add_features
        super(RandFeatLayer, self).__init__()
        
        if randfeat_xt_dim > 0:
            self.Ft = create_layer(hidden_dim, randfeat_xt_dim, feature_type, iid_scale, SWIM_scale, activation)
        if randfeat_x0_dim > 0:
            self.F0 = create_layer(in_dim, randfeat_x0_dim, feature_type, iid_scale, SWIM_scale, activation)


    def fit(self, Xt: Tensor, X0: Tensor, y: Tensor) -> Tensor:
        """Note that SWIM requires y to be onehot or binary"""
        if self.randfeat_xt_dim > 0:
            self.Ft.fit(Xt, y)
        if self.randfeat_x0_dim > 0:
            self.F0.fit(X0, y)
        return self


    def fit_transform(self, Xt, X0, y):
        self.fit(Xt, X0, y)
        return self.forward(Xt, X0)
    

    def forward(self, Xt: Tensor, X0: Tensor) -> Tensor:
        features = []
        if self.randfeat_xt_dim > 0:
            features.append(self.Ft(Xt))
        if self.randfeat_x0_dim > 0:
            features.append(self.F0(X0))

        if self.add_features:
            return sum(features)
        else:
            return torch.cat(features, dim=1)




############################################################################
#################    Ghat layer, Gradient Boosting Regression       ########
############################################################################


class GhatGradientLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 hidden_dim: int = 128,
                 l2_ghat: float = 0.01,
                 ):
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        super(GhatGradientLayerMSE, self).__init__()
        self.ridge = RidgeModule(l2_ghat)


    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        """Fits the functional gradient given features, resnet neurons, and targets,
        and returns the gradient predictions

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Targets, shape (N, d)
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # compute negative gradient, L_2(mu_N) normalized
        N = y.size(0)
        r = y - auxiliary_reg(Xt)
        G = r @ auxiliary_reg.W.T
        G = G / torch.norm(G) * N**0.5

        # fit to negative gradient (finding functional direction)
        Ghat = self.ridge.fit_transform(F, G)

        # line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = sandwiched_LS_scalar(r, auxiliary_reg.W, Ghat, 1e-5)
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)
    


class GhatGreedyLayerMSE(GhatBoostingLayer):
    def __init__(self,
                 hidden_dim: int = 128,
                 l2_ghat: float = 0.01,
                 sandwich_solver: Literal["dense", "diag", "scalar"] = "dense",
                 ):
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        self.sandwich_solver = sandwich_solver
        super(GhatGreedyLayerMSE, self).__init__()

        if sandwich_solver == "dense":
            self.sandwiched_LS = sandwiched_LS_dense
        elif sandwich_solver == "diag":
            self.sandwiched_LS = sandwiched_LS_diag
        elif sandwich_solver == "scalar":
            self.sandwiched_LS = sandwiched_LS_scalar
        else:
            raise ValueError(f"Unknown sandwich solver {sandwich_solver}")


    def fit(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        """Greedily solves the regularized sandwiched least squares problem
        argmin_Delta R(W_t, Phi_t + Delta F) for MSE loss.

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Targets, shape (N, d)
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # greedily minimize R(W_t, Phi_t + Delta F)
        r = y - auxiliary_reg(Xt)
        self.Delta = self.sandwiched_LS(r, auxiliary_reg.W, F, self.l2_ghat)

    
    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_reg: RidgeModule) -> Tensor:
        self.fit(F, Xt, y, auxiliary_reg)
        return self(F)
    

    def forward(self, F: Tensor) -> Tensor:
        if self.sandwich_solver == "scalar":
            return F * self.Delta
        elif self.sandwich_solver == "diag":
            return F * self.Delta[None, :]
        elif self.sandwich_solver == "dense":
            return F @ self.Delta


############################################################################
################# Random Feature Representation Boosting for Regression ###################
############################################################################





class GreedyRFRBoostRegressor(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 512,
                 randfeat_x0_dim: int = 512,
                 l2_reg: float = 0.0001,
                 l2_ghat: float = 0.0001,
                 boost_lr: float = 1.0,
                 sandwich_solver: Literal["dense", "diag", "scalar"] = "dense",
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 use_batchnorm: bool = True,
                 iid_scale: float = 1.0,
                 SWIM_scale: float = 0.5,
                 activation: Literal["tanh", "relu"] = "tanh",
                 return_features: bool = False,  #logits or features
                 add_features: bool = False,  #add features or concat features
                 ):
        """
        Tabular Greedy Random Feaute Representation Boosting.

        If 'sandwich_solver' is 'diag' or 'scalar', the arguments
        'randfeat_xt_dim', 'randfeat_x0_dim', and 'add_features' are ignored 
        and the feature space is set to 'hidden_dim' with 'add_features==True'

        If 'upscale' is 'identity', the 'hidden_dim' argument is ignored.
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_reg = l2_reg
        self.l2_ghat = l2_ghat
        self.boost_lr = boost_lr
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        #activation (needs to be string due to my json code)
        if activation.lower() == "tanh":
            activation = nn.Tanh()
        elif activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim
        upscale = Upscale(in_dim, hidden_dim, upscale_type, iid_scale, SWIM_scale,
                          activation if feature_type=="SWIM" else nn.Tanh())

        # top level regressors
        top_level_regs = [RidgeModule(l2_reg) for _ in range(n_layers+1)]

        # random feature layers
        if sandwich_solver != "dense":
            randfeat_xt_dim = hidden_dim
            randfeat_x0_dim = hidden_dim
            add_features = True
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type,
                          iid_scale, SWIM_scale, activation, add_features)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGreedyLayerMSE(hidden_dim, l2_ghat, sandwich_solver)
            for _ in range(n_layers)
        ]

        super(GreedyRFRBoostRegressor, self).__init__(
            upscale, top_level_regs, random_feature_layers, ghat_boosting_layers, boost_lr, use_batchnorm, return_features=return_features
        )



class GradientRFRBoostRegressor(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 512,
                 randfeat_x0_dim: int = 512,
                 l2_reg: float = 0.0001,
                 l2_ghat: float = 0.0001,
                 boost_lr: float = 1.0,
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 use_batchnorm: bool = True,
                 iid_scale: float = 1.0,
                 SWIM_scale: float = 0.5,
                 activation: Literal["tanh", "relu"] = "tanh",
                 return_features: bool = False,  #logits or features
                 add_features: bool = False,  #add features or concat features
                 ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_reg = l2_reg
        self.l2_ghat = l2_ghat
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        #activation (needs to be string due to my json code)
        if activation.lower() == "tanh":
            activation = nn.Tanh()
        elif activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim
        upscale = Upscale(in_dim, hidden_dim, upscale_type, iid_scale, SWIM_scale,
                          activation if feature_type=="SWIM" else nn.Tanh())

        # top level regressors
        top_level_regs = [RidgeModule(l2_reg) for _ in range(n_layers+1)]

        # random feature layers
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type,
                          iid_scale, SWIM_scale, activation, add_features)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGradientLayerMSE(hidden_dim, l2_ghat)
            for _ in range(n_layers)
        ]

        super(GradientRFRBoostRegressor, self).__init__(
            upscale, top_level_regs, random_feature_layers, ghat_boosting_layers, boost_lr, use_batchnorm, return_features=return_features
        )


############################################
############# End Regression ###############
############################################



#######################################################
############ START CLASSIFICATION #####################
#######################################################



def line_search_cross_entropy(n_classes, cls, X, y, G_hat):
    """Solves the line search risk minimizatin problem
    R(W, X + a * g) for mutliclass cross entropy loss"""
    # No onehot encoding
    if n_classes>2:
        y_labels = torch.argmax(y, dim=1)
    else:
        y_labels = y

    #loss function
    if n_classes > 2:
        loss_fn = nn.functional.cross_entropy #this is with logits
    else:
        loss_fn = nn.functional.binary_cross_entropy_with_logits

    with torch.enable_grad():
        alpha = torch.tensor([0.0], requires_grad=True, device=X.device, dtype=X.dtype)

        def closure(a):
            new_X = X + a * G_hat
            logits = cls(new_X)
            loss = loss_fn(logits, y_labels) + 0.00001 * a**2
            return loss

        result = torchmin.minimize(closure, alpha, method='newton-exact')
    return result.x.detach().item()



class GhatGradientLayerCrossEntropy(GhatBoostingLayer):
    def __init__(self,
                 n_classes: int,
                 hidden_dim: int,
                 l2_ghat: float,
                 do_linesearch: bool,
                 ghat_ridge_solver: Literal["lbfgs", "solve", "ridgecv"] = "solve",
                 ):
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.l2_ghat = l2_ghat
        self.do_linesearch = do_linesearch
        super(GhatGradientLayerCrossEntropy, self).__init__()
        if ghat_ridge_solver == "lbfgs":
            raise NotImplementedError("L-BFGS ghat solver not implemented yet")
        elif ghat_ridge_solver == "solve":
            self.ridge = RidgeModule(l2_ghat)
        elif ghat_ridge_solver == "ridgecv":
            self.ridge = RidgeCVModule()
        else:
            raise ValueError(f"Unknown ghat ridge solver {ghat_ridge_solver}")


    def fit_transform(self, F: Tensor, Xt: Tensor, y: Tensor, auxiliary_cls: LogisticRegression) -> Tensor:
        """Fits the functional gradient given features, resnet neurons, and targets,
        and returns the gradient predictions

        Args:
            F (Tensor): Features, shape (N, p)
            Xt (Tensor): ResNet neurons, shape (N, D)
            y (Tensor): Labels, onehot shape (N, C) or (N, 1) for binary classification
            auxiliary_reg (RidgeModule): Auxiliary top level regressor.
        """
        # compute negative gradient, L_2(mu_N) normalized
        N = y.size(0)
        if self.n_classes==2:
            probs = nn.functional.sigmoid(auxiliary_cls(Xt))
        else:
            probs = nn.functional.softmax(auxiliary_cls(Xt), dim=1)


        # fit to negative gradient (finding functional direction)
        G = (y - probs) @ auxiliary_cls.linear.weight
        G = G / torch.norm(G) * N**0.5
        Ghat = self.ridge.fit_transform(F, G)

        #line search closed form risk minimization of R(W_t, Phi_{t+1})
        self.linesearch = 1.0 if not self.do_linesearch else line_search_cross_entropy(
            self.n_classes, auxiliary_cls, Xt, y, Ghat
            )
        return Ghat * self.linesearch
    

    def forward(self, F: Tensor) -> Tensor:
        return self.linesearch * self.ridge(F)
    

class GradientRFRBoostClassifier(BaseGRFRBoost):
    def __init__(self,
                 in_dim: int,
                 n_classes: int,
                 hidden_dim: int = 128,
                 n_layers: int = 5,
                 randfeat_xt_dim: int = 512,
                 randfeat_x0_dim: int = 512,
                 l2_cls: float = 0.0001,
                 l2_ghat: float = 0.0001,
                 boost_lr: float = 1.0,
                 feature_type : Literal["iid", "SWIM"] = "SWIM",
                 upscale_type: Literal["iid", "SWIM", "identity"] = "iid",
                 ghat_ridge_solver: Literal["lbfgs", "solve", "ridgecv"] = "solve",
                 lbfgs_lr: float = 1.0,
                 lbfgs_max_iter: int = 300,
                 use_batchnorm: bool = True,
                 iid_scale: float = 1.0,
                 SWIM_scale: float = 0.5,
                 activation: Literal["tanh", "relu"] = "tanh",
                 do_linesearch: bool = True,
                 freeze_top_at_t: Optional[int] = None,
                 return_features: bool = False,  #logits or features
                 add_features: bool = False,  #add features or concat features
                 ):
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.randfeat_xt_dim = randfeat_xt_dim
        self.randfeat_x0_dim = randfeat_x0_dim
        self.l2_cls = l2_cls
        self.l2_ghat = l2_ghat
        self.feature_type = feature_type
        self.upscale_type = upscale_type

        #activation (needs to be string due to my json code)
        if activation.lower() == "tanh":
            activation = nn.Tanh()
        elif activation.lower() == "relu":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation {activation}")

        # if no upscale, set hidden_dim to in_dim
        if upscale_type == "identity":
            self.hidden_dim = in_dim
            hidden_dim = in_dim
        upscale = Upscale(in_dim, hidden_dim, upscale_type, iid_scale, SWIM_scale, 
                          activation = activation if upscale_type=="SWIM" else nn.Tanh())

        # auxiliary classifiers
        top_level_classifiers = [LogisticRegression(n_classes, l2_cls, lbfgs_lr, lbfgs_max_iter) 
                                 for _ in range(n_layers+1)] 

        # random feature layers
        random_feature_layers = [
            RandFeatLayer(in_dim, hidden_dim, randfeat_xt_dim, randfeat_x0_dim, feature_type,
                          iid_scale, SWIM_scale, activation, add_features)
            for _ in range(n_layers)
        ]

        # ghat boosting layers
        ghat_boosting_layers = [
            GhatGradientLayerCrossEntropy(n_classes, hidden_dim, l2_ghat, do_linesearch, ghat_ridge_solver)
            for t in range(n_layers)
        ]

        super(GradientRFRBoostClassifier, self).__init__(
            upscale, top_level_classifiers, random_feature_layers, ghat_boosting_layers, boost_lr, use_batchnorm, freeze_top_at_t, return_features
        )