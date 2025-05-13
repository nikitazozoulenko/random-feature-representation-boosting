from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import json
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
import pandas as pd
import openml
import optuna
from sklearn.model_selection import KFold


# Set OpenML cache directory to a location with write permissions
cache_path = os.path.abspath('save/openml_cache')
openml.config.set_root_cache_directory(cache_path)


#########################  |
##### Dataset Code  #####  |
#########################  V

openML_reg_ids = np.array([ # n_features < 200 after one-hot
    41021, 44956, 44957, 44958, 44959, 44960, 44962, 44963,
    44964, 44965, 44966, 44967, 44969, 44970, 44971, 44972,
    44973, 44974, 44975, 44976, 44977, 44978, 44979, 44980,
    44981, 44983, 44984, 44987, 44989, 44990       , 44993,
    44994, 45012, 45402,
    ])

openML_cls_ids = np.array([ # n_features < 200 after one-hot
    3,     6,     11,           14,    15,    16,    18,    22,    23,
    28,    29,    31,    32,    37,    38,    44,           50,    54,
    151,   182,   188,          307,   458,   469,          1049,  1050,
    1053,  1063,  1067,  1068,  1461,  1462,  1464,         1475,       
    1480,         1486,  1487,  1489,  1494,  1497,         1510,  1590,
           4534,  4538,  6332,  23381, 23517, 40499, 40668,        40701,
                  40966, 40975,               40982, 40983, 40984, 40994,
           41027,
    ])


def np_load_openml_dataset(
        dataset_id: int, 
        regression_or_classification: str = "regression",
        max_samples: int = 5000,
        seed: int = 42,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Downloads the openML dataset and normalizes it.
    For regression, it also normalizes the targets.
    For classification, it makes one-hot encodings.
    
    Returns X (shape N,D), and y (shape N,d) for regression or 
    one-hot (N,C) for classification.
    """
    # Fetch dataset from OpenML by its ID
    dataset_id = int(dataset_id)
    dataset = openml.datasets.get_dataset(dataset_id)
    df, _, categorical_indicator, attribute_names = dataset.get_data()
    
    # for classification, dont convert label to onehot
    if regression_or_classification == "classification":
        target_index = attribute_names.index(dataset.default_target_attribute)
        categorical_indicator[target_index] = False
    
    # Dataset 44962 has categorical month and day
    if dataset_id == 44962:
        categorical_indicator[2:4] = [True, True]
        df['month'] = df['month'].astype('category')
        df['day'] = df['day'].astype('category')

    # Replace missing values with the median for numerical columns
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype == 'object':
            # For categorical columns, check if mode exists
            mode_value = df[col].mode()
            fill_value = mode_value[0] if not mode_value.empty else 0
            df[col] = df[col].fillna(fill_value).astype('category')
        else:
            # For numerical columns, fill with median or 0 if all NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value if pd.notnull(median_value) else 0)

    
    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=df.columns[categorical_indicator])
    
    # Separate target variable
    y = np.array(df.pop(dataset.default_target_attribute))
    X = np.array(df).astype(np.float32)

    # Set seed and shuffle data
    np.random.seed(seed)
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]
    
    # Take the first 'max_samples' rows
    X = X[:max_samples]
    y = y[:max_samples]

    # Normalize
    X = X - X.mean(axis=0, keepdims=True)
    X = X / (X.std(axis=0, keepdims=True) + 1e-5)
    X = np.clip(X, -3, 3)
    if regression_or_classification == "regression":
        if y.ndim == 1:
            y = y[:, None]
        y = y - y.mean(axis=0, keepdims=True)
        y = y / (y.std(axis=0, keepdims=True) + 1e-5)
        y = np.clip(y, -3, 3)
        y = y.astype(np.float32)
    else:
        if len(np.unique(y)) > 2:
            # Convert to pandas categorical first to handle string labels
            y_series = pd.Series(y.ravel())
            y = pd.get_dummies(y_series).values.astype(np.float32)
        else:
            # For binary classification, convert labels to 0/1
            unique_labels = np.unique(y)
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            y = np.array([label_map[val] for val in y], dtype=np.float32)[..., None]

    return X, y


def pytorch_load_openml_dataset(
        dataset_id: int, 
        regression_or_classification: Literal["classification", "regression"],
        device: str = "cpu",
        max_samples: int = 5000,
        seed: int = 42,
        ) -> Tuple[Tensor, Tensor]:
    """
    See 'np_load_openml_dataset' for preprocessing details.
    Converts arrays to PyTorch tensors and moves them to the device.
    """
    X, y = np_load_openml_dataset(dataset_id, regression_or_classification, max_samples, seed)
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    return X, y


###################################################################  |
#####  Boilerplate code for tabular PyTorch model evaluation  #####  |
#####  with Optuna hyperparameter tuning inner kfoldcv        #####  |
###################################################################  V


class EarlyStoppingCallback:
    def __init__(self, patience: int = 20, min_delta: float = 1e-8):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('inf')
        self.no_improvement_count = 0
        
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        current_value = trial.value
        
        if current_value is None:
            return
            
        if current_value < (self.best_value - self.min_delta):
            self.best_value = current_value
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        if self.no_improvement_count >= self.patience:
            study.stop()



def get_pytorch_optuna_cv_objective(
        trial,
        ModelClass: Callable,
        get_optuna_params: Callable,
        X_train: Tensor, 
        y_train: Tensor, 
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        ):
    """The objective to be minimized in Optuna's 'study.optimize(objective, n_trials)' function."""
    
    try:
        params = get_optuna_params(trial)

        inner_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
        scores = []
        for inner_train_idx, inner_valid_idx in inner_cv.split(X_train):
            X_inner_train, X_inner_valid = X_train[inner_train_idx], X_train[inner_valid_idx]
            y_inner_train, y_inner_valid = y_train[inner_train_idx], y_train[inner_valid_idx]

            np.random.seed(cv_seed)
            torch.manual_seed(cv_seed)
            torch.cuda.manual_seed(cv_seed)
            model = ModelClass(**params)
            model.fit(X_inner_train, y_inner_train)

            preds = model(X_inner_valid)
            if regression_or_classification == "classification":
                if y_inner_valid.shape[1] > 2:  # Multiclass classification
                    preds = torch.argmax(preds, dim=1)
                    gt = torch.argmax(y_inner_valid, dim=1)
                    acc = (preds == gt).float().mean()
                    scores.append(-acc.item())  # score is being minimized in Optuna
                else:  # Binary classification
                    preds = torch.sigmoid(preds).round()
                    acc = (preds == y_inner_valid).float().mean()
                    scores.append(-acc.item())  # score is being minimized in Optuna
            else:
                rmse = torch.sqrt(nn.functional.mse_loss(y_inner_valid, preds))
                scores.append(rmse.item())

        return np.mean(scores)
    except (RuntimeError, ValueError, torch._C._LinAlgError) as e:
        print(f"Error encountered during training: {e}. Returning score 2.0 to optuna")
        return 2.0 #rmse random guessing is 1.0, and random guessing accuracy is -1/C



def evaluate_pytorch_model_single_fold(
        ModelClass : Callable,
        get_optuna_params : Callable,
        X_train: Tensor,
        X_test: Tensor,
        y_train: Tensor,
        y_test: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int, # e.g. 25 with n_trials=100
        ):
    """
    Evaluates a PyTorch model on a specified Train and Test set.
    Hyperparameters are tuned using Optuna with an inner k-fold CV loop.

    Returns the train and test scores, the time to fit the model, 
    inference time, and the best hyperparameters.
    """
    #hyperparameter tuning with Optuna
    sampler = optuna.samplers.TPESampler(seed=cv_seed)  # Make the sampler behave in a deterministic way.
    study = optuna.create_study(direction="minimize", sampler=sampler)
    objective = lambda trial: get_pytorch_optuna_cv_objective(
        trial, ModelClass, get_optuna_params, X_train, y_train, 
        k_folds, cv_seed, regression_or_classification
        )
    study.optimize(
        objective, 
        n_trials=n_optuna_trials,
        callbacks=[EarlyStoppingCallback(early_stopping_patience)],
        )

    #fit model with optimal hyperparams
    np.random.seed(cv_seed)
    torch.manual_seed(cv_seed)
    torch.cuda.manual_seed(cv_seed)
    t0 = time.perf_counter()
    model = ModelClass(**study.best_params).to(device)
    model.fit(X_train, y_train)

    #predict
    t1 = time.perf_counter()
    preds_train = model(X_train)
    preds_test = model(X_test)
    t2 = time.perf_counter()

    #evaluate
    if regression_or_classification == "classification":
        if y_train.shape[1] > 2:  # Multiclass classification
            preds_train = torch.argmax(preds_train, dim=1)
            gt_train = torch.argmax(y_train, dim=1)
            acc_train = (preds_train == gt_train).float().mean()
            score_train = -acc_train

            preds_test = torch.argmax(preds_test, dim=1) 
            gt_test = torch.argmax(y_test, dim=1)
            acc_test = (preds_test == gt_test).float().mean()
            score_test = -acc_test
        else:  # Binary classification
            preds_train = torch.sigmoid(preds_train).round()
            acc_train = (preds_train == y_train).float().mean()
            score_train = -acc_train

            preds_test = torch.sigmoid(preds_test).round()
            acc_test = (preds_test == y_test).float().mean() 
            score_test = -acc_test
    else:
        preds_train = model(X_train)
        score_train = torch.sqrt(nn.functional.mse_loss(y_train, preds_train))

        preds_test = model(X_test)
        score_test = torch.sqrt(nn.functional.mse_loss(y_test, preds_test))
    
    return (score_train.item(), score_test.item(), t1-t0, t2-t1, study.best_params.copy())

    

def evaluate_pytorch_model_kfoldcv(
        ModelClass : Callable,
        get_optuna_params : Callable,
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    """
    Evaluates a PyTorch model using k-fold cross-validation,
    with an inner Optuna hyperparameter tuning loop for each fold.
    The model is then trained on the whole fold train set and evaluated
    on the fold test set.

    Inner and outer kFoldCV use the same number of folds.

    Regression: RMSE is used as the evaluation metric.
    Classification: (negative) Accuracy is used as the evaluation metric.
    """
    outer_cv = KFold(n_splits=k_folds, shuffle=True, random_state=cv_seed)
    outer_train_scores = []
    outer_test_scores = []
    chosen_params = []
    fit_times = []
    inference_times = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        score_train, score_test, t_fit, t_inference, best_params = evaluate_pytorch_model_single_fold(
            ModelClass, get_optuna_params,
            X_train, X_test, y_train, y_test, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
            )

        #save
        outer_train_scores.append(score_train)
        outer_test_scores.append(score_test)
        fit_times.append(t_fit)
        inference_times.append(t_inference)
        chosen_params.append(best_params)
    
    return (outer_train_scores,
            outer_test_scores,
            fit_times,
            inference_times,
            chosen_params,
            )


def save_experiments_json(
        experiments: Dict[str, Dict[str, Dict[str, Any]]],
        save_path: str,
        ):
    os.makedirs(os.path.dirname(str(save_path)), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(experiments, f, indent=4)


def evaluate_dataset_with_model(
        X: Tensor,
        y: Tensor,
        name_dataset: str,
        evaluate_model_func: Callable,
        name_model: str,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        save_dir: str,
        early_stopping_patience: int,
        ):
    """Evaluates a model on a given tabular dataset (X, y).
    Returns a json of the kfoldCV results.

    Args:
        X (Tensor): Shape (N, D) of the input features.
        y (Tensor): Shape (N, C) for classification (NOTE one-hot), or (N, d) 
                    for regression (NOTE y.dim==2 even if d==1).
        name_dataset (str): Name of the dataset.
        evaluate_model_func (Callable): Function that evaluates the model, 
                                        see e.g. 'evaluate_LogisticRegression'.
        name_model (str): Name of the model.
        k_folds (int): Number of folds in the outer and inner CV.
        cv_seed (int): Seed for all the randomness.
        regression_or_classification (str): Either 'classification' or 'regression'
        n_optuna_trials (int): Number of Optuna trials for hyperparameter tuning.
        device (str): PyTorch device.
        save_dir (Optional[str]): If not None, path to the save directory.
        early_stopping_patience (int): Patience for early stopping optimization in Optuna.
    """
    np.random.seed(cv_seed)
    torch.manual_seed(cv_seed)
    torch.cuda.manual_seed(cv_seed)

    # Fetch and process each dataset
    results = evaluate_model_func(
        X, y, k_folds, cv_seed, regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    
    # store results in nested dict
    experiments = {}
    experiments[str(name_dataset)] = {}
    experiments[str(name_dataset)][name_model] = {
        "score_train": results[0],
        "score_test": results[1],
        "t_fit": results[2],
        "t_inference": results[3],
        "hyperparams": results[4],
    }

    # Save results if specified
    if save_dir is not None:
        path = os.path.join(
            save_dir, 
            f"{regression_or_classification}_{str(name_dataset)}_{name_model}.json"
            )
        save_experiments_json(experiments, path)

    return experiments



def run_all_openML_with_model(
        dataset_ids: List[int],
        evaluate_model_func: Callable,
        name_model: str,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: Literal["classification", "regression"],
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        save_dir: str,
        early_stopping_patience: int,
        max_samples: int = 5000,
        ):
    """Evaluates a model on a list of OpenML datasets.

    Args:
        dataset_ids (List[int]): List of OpenML dataset IDs.
        evaluate_model_func (Callable): Function that evaluates the model, 
                                        see e.g. 'evaluate_LogisticRegression'.
        name_model (str): Name of the model.
        k_folds (int): Number of folds in the outer and inner CV.
        cv_seed (int): Seed for all the randomness.
        regression_or_classification (str): Either 'classification' or 'regression'
        n_optuna_trials (int): Number of Optuna trials for hyperparameter tuning.
        device (str): PyTorch device.
        save_dir (str): Path to the save directory.
        early_stopping_patience (int): Patience for early stopping optimization in Optuna.
        max_samples (int): Maximum number of samples to use.
    """
    # Fetch and process each dataset
    experiments = {}
    for i, dataset_id in enumerate(dataset_ids):
        dataset_id = str(dataset_id)
        X, y = pytorch_load_openml_dataset(dataset_id, regression_or_classification, device, max_samples)
        print("X shape", X.shape)
        
        json = evaluate_dataset_with_model(
            X, y, dataset_id, evaluate_model_func, name_model, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, save_dir, early_stopping_patience
            )
        experiments[dataset_id] = json[dataset_id]
        print(f" {i+1}/{len(dataset_ids)} Processed dataset {dataset_id}")
    
    return experiments