from typing import Dict, List, Any, Optional, Literal, Union
import time

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV, KFold, ShuffleSplit
from sklearn.metrics import roc_auc_score
import numpy as np
import torch
from torch import Tensor



class SKLearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 reg_or_cls: Literal["regression", "classification"] = "classification",
                 seed=0,
                 modelClass=None, 
                 **model_params,
                 ):
        self.reg_or_cls = reg_or_cls
        self.modelClass = modelClass
        self.model_params = model_params
        self.seed = seed
        self.model = None

    def fit(self, X:Tensor, y:Tensor):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        self.model = self.modelClass(**self.model_params)
        self.model.fit(X, y)

        #classes, either label for binary or one-hot for multiclass
        if self.reg_or_cls == "classification":
            if len(y.size()) == 1 or y.size(1) == 1:
                self.classes_ = np.unique(y.detach().cpu().numpy())
            else:
                self.classes_ = np.unique(y.argmax(axis=1).detach().cpu().numpy())
        return self

    def predict(self, X:Tensor):
        #binary classification
        if self.reg_or_cls == "classification":
            if len(self.classes_) == 2:
                proba_1 = torch.sigmoid(self.model(X))
                return (proba_1 > 0.5).detach().cpu().numpy()
            else:
                #multiclass
                return torch.argmax(self.model(X), dim=1).detach().cpu().numpy()
        else: #regression
            return self.model(X).detach().cpu().numpy()
    

    def predict_proba(self, X:Tensor):
        #binary classification
        if len(self.classes_) == 2:
            proba_1 = torch.nn.functional.sigmoid(self.model(X))
            return torch.cat((1 - proba_1, proba_1), dim=1).detach().cpu().numpy()
        else:
            #multiclass
            logits = self.model(X)
            proba = torch.nn.functional.softmax(logits, dim=1)
            return proba.detach().cpu().numpy()
    

    def decision_function(self, X:Tensor):
        logits = self.model(X)
        return logits.detach().cpu().numpy()


    def set_params(self, **params):
        self.modelClass = params.pop('modelClass', self.modelClass)
        self.seed = params.pop('seed', self.seed)
        self.reg_or_cls = params.pop('reg_or_cls', self.reg_or_cls)
        self.model_params.update(params)
        return self


    def get_params(self, deep=True):
        params = {'modelClass': self.modelClass,
                'seed': self.seed,
                'reg_or_cls': self.reg_or_cls}
        params.update(self.model_params)
        return params
        

    def score(self, X:Tensor, y:Tensor):
        self.model.eval()
        if self.reg_or_cls == "classification":
            logits = self.model(X)
            if y.size(1) == 1:
                y_true = y.detach().cpu().numpy()
                y_score = logits.detach().cpu().numpy()
                auc = roc_auc_score(y_true, y_score)
                score = auc
            else:
                pred = torch.argmax(logits, dim=1)
                y = torch.argmax(y, dim=1)
                acc = (pred == y).float().mean()
                score = acc.detach().cpu().item()
        else:
            rmse = -torch.sqrt(torch.mean((y - self.model(X)) ** 2)).detach().cpu().item()
            score = rmse
        return score







class SkLearnGridsearchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 modelClass=None,
                 reg_or_cls: Literal["regression", "classification"] = "classification",
                 param_grid: Dict[str, List[Any]] = {},
                 validation_strategy: Literal["holdout", "kfold"] = "holdout",
                 kfolds: int = 5,
                 holdout_percentage: float = 0.2,
                 seed: Optional[int] = 42,
                 verbose: int = 2
                 ):
        self.modelClass = modelClass
        self.reg_or_cls = reg_or_cls
        self.param_grid = param_grid
        self.validation_strategy = validation_strategy
        self.kfolds = kfolds
        self.holdout_percentage = holdout_percentage
        self.seed = seed
        self.verbose = verbose
        self.best_model = None
        self.best_params_ = None


    def fit(self, X:Tensor, y:Tensor):
        """
        Performs either k-fold CV or holdout validation for hyperparameter tuning
        based on self.param_grid, and then fits the best model on the whole train set.
        """

        t0 = time.time()
        # Setup cross-validation strategy
        if self.validation_strategy == "kfold":
            cv = KFold(n_splits=self.kfolds, shuffle=True, random_state=self.seed)
        else:  # holdout
            cv = ShuffleSplit(n_splits=1, test_size=self.holdout_percentage, random_state=self.seed)
        
        # Perform grid search
        estimator = SKLearnWrapper(modelClass=self.modelClass, reg_or_cls=self.reg_or_cls, seed=self.seed)
        grid_search = GridSearchCV(
            estimator=estimator,
            param_grid=self.param_grid,
            cv=cv,
            verbose=self.verbose,
            n_jobs=1,
        )
        
        grid_search.fit(X, y)
        t1 = time.time()
        self.fit_time = t1 - t0
        
        # Save best model and params
        self.best_model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_
        
        if self.verbose > 0:
            print(f"Best parameters: {self.best_params_}")
            
        return self


    def predict(self, X:Tensor):
        """
        Use the best model to make predictions
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")
            
        return self.best_model.predict(X)
    

    def predict_proba(self, X):
        """
        Return probability estimates for samples
        """
        if self.best_model is None:
            raise ValueError("Model has not been trained yet. Call fit() first.")

        return self.best_model.predict_proba(X)
    

    def score(self, X, y):
        return self.best_model.score(X, y)
    


#### usage example ####
if __name__ == "__main__":
    from models.random_feature_representation_boosting import GradientRFRBoostClassifier

    # synthetic data
    D = 10
    N = 100
    n_classes = 3
    X_train = torch.randn(N, D)
    y_train = torch.randint(0, n_classes, (N,))
    X_test = torch.randn(N, D)
    y_test = torch.randint(0, n_classes, (N,))

    # Define hyperparameter search space
    param_grid = {
            'l2_cls': np.logspace(-4, 0, 5),
            'l2_ghat': np.logspace(-4, 0, 5),
            'in_dim': [2],
            'n_classes': [3],
            'hidden_dim': [2],
            'n_layers': [1, 2, 3],
            'randfeat_xt_dim': [512],
            'randfeat_x0_dim': [512],
            'feature_type': ["SWIM"],
            'upscale_type': ["iid"],
            'use_batchnorm': [True],
            'lbfgs_max_iter': [300],
            'lbfgs_lr': [1.0],
            'SWIM_scale': [0.5],
            'activation': ["swim"],
        }

    #fit with grid search
    gs_model = SkLearnGridsearchWrapper(
        modelClass=GradientRFRBoostClassifier,
        param_grid=param_grid,
        reg_or_cls="classification",
        validation_strategy='holdout',
        holdout_percentage=0.2,
        seed=42,
        verbose=3,
    )
    gs_model.fit(X_train, y_train)
    train_acc = gs_model.score(X_train, y_train)
    test_acc = gs_model.score(X_test, y_test)
    print("train_acc", train_acc)
    print("test_acc", test_acc)