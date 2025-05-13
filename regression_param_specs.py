from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor

from models.base import RidgeModule
from models.xgboost_wrapper import XGBoostRegressorWrapper
from models.end2end import End2EndMLPResNet
from models.random_feature_representation_boosting import GradientRFRBoostRegressor, GreedyRFRBoostRegressor
from optuna_kfoldCV import evaluate_pytorch_model_kfoldcv

##############################################################  |
##### Create "evalute_MODELHERE" function for each model #####  |
##############################################################  V


def get_GradientRFRBoost_eval_fun(
        feature_type: Literal["iid", "SWIM"] = "SWIM",
        upscale_type: Literal["iid", "SWIM", "identity"] = "SWIM",
        activation: str = "tanh",
        use_batchnorm: bool = True,
        add_features: bool = False,
        ):
    """Returns a function that evaluates the GradientRFRBoost model
    with the specified number of layers"""
    def evaluate_GRFRBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        ModelClass = GradientRFRBoostRegressor
        get_optuna_params = lambda trial : {
            # Fixed values
            "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]),
            "upscale_type": trial.suggest_categorical("upscale_type", [upscale_type]),
            "randfeat_xt_dim": trial.suggest_categorical("randfeat_xt_dim", [512]),
            "randfeat_x0_dim": trial.suggest_categorical("randfeat_x0_dim", [512]),
            "activation": trial.suggest_categorical("activation", [activation]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [use_batchnorm]),
            "add_features": trial.suggest_categorical("add_features", [add_features]),
            # Hyperparameters
            "n_layers": trial.suggest_int("n_layers", 1, 10, log=True),
            "l2_reg": trial.suggest_float("l2_reg", 1e-5, 10, log=True),
            "l2_ghat": trial.suggest_float("l2_ghat", 1e-5, 10, log=True),
            "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0, log=True),
            "hidden_dim": (
                trial.suggest_int("hidden_dim", 16, 512, log=True) if upscale_type != "identity"
                else trial.suggest_categorical("hidden_dim", [X.size(1)])
            ),
            "SWIM_scale" if feature_type == "SWIM" else "iid_scale" : (
                (trial.suggest_float("SWIM_scale", 0.25, 2.0) if feature_type == "SWIM"
                else trial.suggest_float("iid_scale", 0.1, 10, log=True))
            ),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_GRFRBoost



def get_GreedyRFRBoost_eval_fun(
        feature_type: Literal["iid", "SWIM"] = "SWIM",
        upscale_type: Literal["iid", "SWIM", "identity"] = "SWIM",
        sandwich_solver: Literal["dense", "diag", "scalar"] = "dense",
        activation: str = "tanh",
        use_batchnorm: bool = True,
        add_features: bool = False,
        ):
    """Returns a function that evaluates the GreedyRFRBoost model
    with the specified number of layers"""
    def evaluate_GreedyRFRBoost(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        
        ModelClass = GreedyRFRBoostRegressor
        get_optuna_params = lambda trial : {
            # Fixed values
            "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),
            "feature_type": trial.suggest_categorical("feature_type", [feature_type]),
            "upscale_type": trial.suggest_categorical("upscale_type", [upscale_type]),
            "randfeat_xt_dim": trial.suggest_categorical("randfeat_xt_dim", [512]),
            "randfeat_x0_dim": trial.suggest_categorical("randfeat_x0_dim", [512]),
            "sandwich_solver": trial.suggest_categorical("sandwich_solver", [sandwich_solver]),
            "activation": trial.suggest_categorical("activation", [activation]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [use_batchnorm]),
            "add_features": trial.suggest_categorical("add_features", [add_features]),
            # Hyperparameters
            "n_layers": trial.suggest_int("n_layers", 1, 10, log=True),
            "l2_reg": trial.suggest_float("l2_reg", 1e-5, 10, log=True),
            "l2_ghat": trial.suggest_float("l2_ghat", 1e-5, 10, log=True),
            "boost_lr": trial.suggest_float("boost_lr", 0.1, 1.0, log=True),
            "hidden_dim": (
                trial.suggest_int("hidden_dim", 16, 512, log=True) if upscale_type != "identity"
                else trial.suggest_categorical("hidden_dim", [X.size(1)])
            ),
            "SWIM_scale" if feature_type == "SWIM" else "iid_scale" : (
                (trial.suggest_float("SWIM_scale", 0.25, 2.0) if feature_type == "SWIM"
                else trial.suggest_float("iid_scale", 0.1, 10, log=True))
            ),
        }

        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_GreedyRFRBoost




def get_RandomFeatureNetwork_eval_fun(
        feature_type: Literal["iid", "SWIM"] = "SWIM",
        activation: str = "tanh",
        ):
    """Returns a function that evaluates the RandomFeatureNetwork model
    (1 hidden layer random neural network)"""
    def evaluate_RandomFeatureNetwork(
            X: Tensor,
            y: Tensor,
            k_folds: int,
            cv_seed: int,
            regression_or_classification: str,
            n_optuna_trials: int,
            device: Literal["cpu", "cuda"],
            early_stopping_patience: int,
            ):
        ModelClass = GradientRFRBoostRegressor
        get_optuna_params = lambda trial : {
            # Fixed values
            "n_layers": trial.suggest_categorical("n_layers", [0]),
            "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),
            "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),
            "upscale_type": trial.suggest_categorical("upscale_type", [feature_type]),
            "activation": trial.suggest_categorical("activation", [activation]),
            "use_batchnorm": trial.suggest_categorical("use_batchnorm", [False]),
            # Hyperparameters
            "l2_reg": trial.suggest_float("l2_reg", 1e-5, 10, log=True),
            "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
            "SWIM_scale" if feature_type == "SWIM" else "iid_scale" : (
                (trial.suggest_float("SWIM_scale", 0.25, 2.0) if feature_type == "SWIM"
                else trial.suggest_float("iid_scale", 0.1, 10, log=True))
            ),
        }


        return evaluate_pytorch_model_kfoldcv(
            ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
            regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )
    return evaluate_RandomFeatureNetwork



def evaluate_Ridge(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    ModelClass = RidgeModule
    get_optuna_params = lambda trial : {
        "l2_reg": trial.suggest_float("l2_reg", 1e-5, 10, log=True),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
    )





def evaluate_XGBoostRegressor(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    ModelClass = XGBoostRegressorWrapper
    get_optuna_params = lambda trial : {
        #fixed
        "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),   # Fixed value
        #hyperparms
        "alpha": trial.suggest_float("alpha", 0.00001, 0.01, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
    )



def evaluate_End2End(
        X: Tensor,
        y: Tensor,
        k_folds: int,
        cv_seed: int,
        regression_or_classification: str,
        n_optuna_trials: int,
        device: Literal["cpu", "cuda"],
        early_stopping_patience: int,
        ):
    
    ModelClass = End2EndMLPResNet
    get_optuna_params = lambda trial : {
        # Fixed values
        "in_dim": trial.suggest_categorical("in_dim", [X.size(1)]),
        "out_dim": trial.suggest_categorical("out_dim", [y.size(1)]),
        "loss": trial.suggest_categorical("loss", ["mse"]),
        "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", [512]),
        # Hyperparameters
        "n_blocks": trial.suggest_int("n_blocks", 1, 10),
        "hidden_dim": trial.suggest_int("hidden_dim", 16, 512, log=True),
        "lr": trial.suggest_float("lr", 1e-6, 1e-1, log=True),
        "end_lr_factor": trial.suggest_float("end_lr_factor", 0.01, 1.0, log=True),
        "n_epochs": trial.suggest_int("n_epochs", 10, 50, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 0.001, log=True),
        "batch_size": trial.suggest_int("batch_size", 128, min(512, int(X.size(0) * (k_folds-1)/k_folds)), step=128),
    }

    return evaluate_pytorch_model_kfoldcv(
        ModelClass, get_optuna_params, X, y, k_folds, cv_seed, 
        regression_or_classification, n_optuna_trials, device, early_stopping_patience
        )



######################################################  |
#####  command line argument to run experiments  #####  |
######################################################  V

# start by using all datasets. I can change this later
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
    parser.add_argument(
        "--models", 
        nargs='+', 
        type=str, 
        default=["GradientRFBoost", "GreedyRFBoost"], 
        help="List of model names to run."
    )
    parser.add_argument(
        "--dataset_indices", 
        nargs='+', 
        type=int, 
        default=[i for i in range(len(openML_reg_ids))], 
        help="List of datasets to run."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/nikita/Code/random-feature-boosting/save/OpenMLRegression/",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--n_optuna_trials",
        type=int,
        default=100,
        help="Number of optuna trials in the inner CV hyperparameter loop."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device to run the experiments on."
    )
    parser.add_argument(
        "--k_folds",
        type=int,
        default=5,
        help="Number of inner and outer CV folds."
    )
    parser.add_argument(
        "--cv_seed",
        type=int,
        default=42,
        help="Seed for all randomness."
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=50,
        help="Number of trials before early stopping in Optuna early stopping callback."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of rows per dataset."
    )
    return parser.parse_args()


from optuna_kfoldCV import run_all_openML_with_model, openML_reg_ids

if __name__ == "__main__":
    args = parse_args()

    # Run experiments
    for model_name in args.models:
        # baseline models
        if "End2End" in model_name:
            eval_fun = evaluate_End2End
        elif "Ridge" in model_name:
            eval_fun = evaluate_Ridge
        elif "XGBoost" in model_name:
            eval_fun = evaluate_XGBoostRegressor
        #RFNN
        elif model_name=="RFNN":
            eval_fun = get_RandomFeatureNetwork_eval_fun("SWIM", "tanh")
        elif model_name=="RFNN_iid":
            eval_fun = get_RandomFeatureNetwork_eval_fun("iid", "tanh")
        #RFRBoost
        elif "RFRBoost" in model_name:
            upscale_type = "iid" if "upscaleiid" in model_name else "SWIM"
            feat_type = "iid" if "featiid" in model_name else "SWIM"
            if "ID" in model_name:
                upscale_type = "identity"
            activation = "relu" if "relu" in model_name else "tanh"
            use_batchnorm = False if "batchnormFalse" in model_name else True
            add_features = True if "addfeatures" in model_name else False
            if "Greedy" in model_name:
                sandwich = "dense" if "Dense" in model_name else "diag" if "Diag" in model_name else "scalar"
                eval_fun = get_GreedyRFRBoost_eval_fun(feat_type, upscale_type, sandwich, activation, use_batchnorm, add_features)
            else:
                eval_fun = get_GradientRFRBoost_eval_fun(feat_type, upscale_type, activation, use_batchnorm, add_features)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # run the experiments
        run_all_openML_with_model(
            dataset_ids = openML_reg_ids[args.dataset_indices],
            evaluate_model_func = eval_fun,
            name_model = model_name,
            k_folds = args.k_folds,
            cv_seed = args.cv_seed,
            regression_or_classification="regression",
            n_optuna_trials = args.n_optuna_trials,
            device = args.device,
            save_dir = args.save_dir,
            early_stopping_patience = args.early_stopping_patience,
            max_samples=args.max_samples,
        )