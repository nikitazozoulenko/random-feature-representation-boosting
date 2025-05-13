from typing import Tuple, List, Union, Any, Optional, Dict, Literal, Callable
import time
import os
import sys
import json
import requests
import zipfile
import io
from pathlib import Path
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, tensor
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



from optuna_kfoldCV import run_all_openML_with_model
from classification_param_specs import evaluate_LogisticRegression, evaluate_XGBoostClassifier
from models.gridsearch_wrapper import SkLearnGridsearchWrapper
from optuna_kfoldCV import save_experiments_json
from models.end2end import End2EndMLPResNet
from models.xgboost_wrapper import XGBoostClassifierWrapper, XGBoostRegressorWrapper


#########################
### data loading code ###
#########################


def download_and_save_ypmsd(filepath):
    """Downloads and saves the YearPredictionMSD dataset from the UCI repository."""
    url = "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        z = zipfile.ZipFile(io.BytesIO(response.content))
        file_name = "YearPredictionMSD.txt"
        with z.open(file_name) as f:
            df = pd.read_csv(f, header=None) # read directly from the zipfile object.

        df.to_csv(filepath, index=False)
        print(f"YearPredictionMSD downloaded and saved to {filepath}")
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error downloading: {e}")
        return None
    except zipfile.BadZipFile as e:
        print(f"Error with zip file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None



def load_or_download_yp_msd(filepath="year_prediction_msd.csv"):
    """Loads YearPredictionMSD from disk if it exists, otherwise downloads and saves it from UCI.

    Args:
        filepath (str): The path to save/load the CSV file.

    Returns:
        pandas.DataFrame: The YearPredictionMSD dataset as a DataFrame.
    """
    if os.path.exists(filepath):
        print(f"Loading YearPredictionMSD from {filepath}...")
        try:
            df = pd.read_csv(filepath)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Attempting to download again.")
            return download_and_save_ypmsd(filepath)

    else:
        print("YearPredictionMSD not found. Downloading from UCI...")
        return download_and_save_ypmsd(filepath)



def get_ypmsd(
        savepath = "Folder/YPMSD.csv",
        device = "cuda",
    ):
    ypmsd_df = load_or_download_yp_msd(filepath = savepath)

    X = ypmsd_df.iloc[:, 1:].values
    y = ypmsd_df.iloc[:, 0].values
    X_train, X_test = X[:463715], X[463715:]
    y_train, y_test = y[:463715], y[463715:]

    # normalize
    epsilon = 1e-6
    X_means = np.mean(X_train, axis=0)
    X_stds = np.std(X_train, axis=0) + epsilon
    y_means = np.mean(y_train)
    y_stds = np.std(y_train) + epsilon

    X_train = (X_train - X_means) / X_stds
    y_train = (y_train - y_means) / y_stds
    X_test = (X_test - X_means) / X_stds
    y_test = (y_test - y_means) / y_stds

    return (torch.from_numpy(X_train).float().to(device), 
            torch.from_numpy(y_train[..., None]).float().to(device), 
            torch.from_numpy(X_test).float().to(device), 
            torch.from_numpy(y_test[..., None]).float().to(device),
            )


def get_covtype(device="cuda"):
    # download covtype dataset
    covtype = datasets.fetch_covtype()
    X, y = covtype.data, covtype.target

    #onehot encode target
    y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))

    #split train test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # First 10 are cont, other are onehot. normalize the cont
    epsilon = 1e-6
    means = np.mean(X_train[:, :10], axis=0)
    stds = np.std(X_train[:, :10], axis=0) + epsilon

    X_train[:, :10] = (X_train[:, :10] - means) / stds
    X_test[:, :10] = (X_test[:, :10] - means) / stds

    return (torch.from_numpy(X_train).float().to(device), 
            torch.from_numpy(y_train).float().to(device), 
            torch.from_numpy(X_test).float().to(device), 
            torch.from_numpy(y_test).float().to(device),
            )



############################################
# The main function to run the experiments #
############################################



def evaluate_big_experiments_for_model(
    param_grid: Dict[str, List[Any]],
    modelClass: Any,
    model_name: str,
    X_train,
    X_test,
    y_train,
    y_test,
    save_dir: Path,
    dataset_name: str,
    reg_or_cls: Literal["classification", "regression"],
    train_sizes: List[int] = [2000, 4000, 8000, 16000, 32000, 64000],
    model_rerun_seeds: List[int] = [0, 1, 2, 3, 4, 5],
    holdout_percentage = 0.2,
):
    """
    Evaluate a model on a range of dataset sizes, using a grid search wrapper.
    """
    for train_size in train_sizes:
        scores = []
        fit_times = []
        inference_times = []
        best_params = []
        for seed in model_rerun_seeds:
            print(f"Running {model_name} on {dataset_name} with {train_size} examples and seed {seed}...")
            X_train_new, y_train_new, X_test_new, y_test_new = (
                X_train, y_train, X_test, y_test
            )

            #subsample within train
            X_train_subsample, _, y_train_subsample, _ = train_test_split(
                X_train_new, y_train_new, 
                train_size=min(train_size, len(X_train_new)-1),
                random_state=seed,
            )

            #fit with grid search
            gs_model = SkLearnGridsearchWrapper(
                modelClass=modelClass,
                param_grid={**param_grid},
                reg_or_cls=reg_or_cls,
                validation_strategy='holdout',
                holdout_percentage=holdout_percentage,
                seed= seed,
                verbose=3,
            )
            gs_model.fit(X_train_subsample, y_train_subsample)
            t0 = time.time()
            score = gs_model.score(X_test_new, y_test_new)
            t1 = time.time()
            scores.append(score)
            fit_times.append(gs_model.fit_time)
            inference_times.append(t1-t0)
            best_params.append(gs_model.best_params_)
            print("score", score)

        # save
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "train_size": train_size,
            "score": scores,
            "fit_time": fit_times,
            "inference_time": inference_times,
            "best_params": best_params,
        }
        print(results)
        save_experiments_json(results, save_dir / dataset_name / f"{model_name}_{train_size}.json")



#############################################################################
# Use this to queue up some jobs on the cluster.                            #
# The majority of the experiments are run directly in the Jupyter notebook. #
#############################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments with different models and datasets.")
    parser.add_argument(
        "--model", 
        type=str, 
        default="XGBoost",
        help="Model to run. Options: 'XGBoost', 'E2E MLP ResNet'."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="YPMSD",
        help="Name of the dataset to use. Options: 'YPMSD', 'CoverType'."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/rds/general/user/nz423/home/random-feature-boosting/save/LargeExperiments",
        help="Directory where the results json will be saved to file."
    )
    parser.add_argument(
        "--YPMSD_dir",
        type=str,
        default="/rds/general/user/nz423/home/random-feature-boosting/save/LargeExperiments",
        help="Directory where the YPMSD dataset will be saved."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device to run the experiments on."
    )    
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    # boring old switch case will do... i am aware this is bad practice, but it is a small script and not very important
    if args.dataset_name == "YPMSD" and "XGBoost" in args.model:
        X_train, y_train, X_test, y_test = get_ypmsd(Path(args.YPMSD_dir)/"YPMSD.csv", device=args.device)
        evaluate_big_experiments_for_model(
            param_grid = {
                "objective": ["reg:squarederror"],
                "learning_rate": [0.01, 0.033, 0.1],
                "n_estimators": [250, 500, 1000],
                "max_depth": [1, 3, 6, 10],
                "lambda": [1],
                "device": [args.device],
            },
            modelClass = XGBoostRegressorWrapper,
            model_name=args.model,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            dataset_name=args.dataset_name,
            reg_or_cls="regression",
            train_sizes = [2000, 2000, 4000, 8000, 16000, 32000, 64000,
                        128000, 256000, 463715],
            model_rerun_seeds = [0, 1, 2, 3, 4],
            save_dir=Path(args.save_dir),
        )
    
    elif args.dataset_name == "YPMSD" and "E2E MLP ResNet" in args.model:
        X_train, y_train, X_test, y_test = get_ypmsd(Path(args.YPMSD_dir)/"YPMSD.csv", device=args.device)
        evaluate_big_experiments_for_model(
            param_grid = {
                'loss': ["mse"],
                "in_dim": [X_train.shape[1]],
                "out_dim": [y_train.shape[1]],
                'bottleneck_dim': [512],
                "n_blocks": [3],
                'lr': [0.1, 0.01, 0.001],
                'n_epochs': [10, 20, 30],
                'end_lr_factor': [0.01],
                'weight_decay': [1e-5],
                'batch_size': [256],
                'upsample': [False],
                "hidden_dim": [X_train.shape[1]],
            },
            modelClass = End2EndMLPResNet,
            model_name=args.model,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            dataset_name=args.dataset_name,
            reg_or_cls="regression",
            train_sizes = [2000, 2000, 4000, 8000, 16000, 32000, 64000,
                        128000, 256000, 463715],
            model_rerun_seeds = [0, 1, 2, 3, 4],
            save_dir=Path(args.save_dir),
        )

    elif args.dataset_name == "CoverType" and "XGBoost" in args.model:
        X_train, y_train, X_test, y_test = get_covtype(args.device)
        evaluate_big_experiments_for_model(
            param_grid = {
                "objective": ["multi:softmax"],
                "learning_rate": [0.01, 0.033, 0.1],
                "n_estimators": [250, 500, 1000],
                "max_depth": [1, 3, 6, 10],
                "lambda": [1],
                "device": [args.device],
            },
            modelClass = XGBoostClassifierWrapper,
            model_name=args.model,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            dataset_name=args.dataset_name,
            reg_or_cls="classification",
            train_sizes = [2000, 2000, 4000, 8000, 16000, 32000, 64000, 
                        128000, 256000, 464809],
            model_rerun_seeds = [0, 1, 2, 3, 4],
            save_dir=Path(args.save_dir),
        )

    elif args.dataset_name == "CoverType" and "E2E MLP ResNet" in args.model:
        X_train, y_train, X_test, y_test = get_covtype(args.device)
        evaluate_big_experiments_for_model(
            param_grid = {
                'loss': ["cce"],
                "in_dim": [X_train.shape[1]],
                "out_dim": [y_train.shape[1]],
                'bottleneck_dim': [512],
                "n_blocks": [3],
                'lr': [0.1, 0.01, 0.001],
                'n_epochs': [10, 20, 30],
                'end_lr_factor': [0.01],
                'weight_decay': [1e-5],
                'batch_size': [256],
                'upsample': [False],
                "hidden_dim": [X_train.shape[1]],
            },
            modelClass = End2EndMLPResNet,
            model_name=args.model,
            X_train = X_train,
            X_test = X_test,
            y_train = y_train,
            y_test = y_test,
            dataset_name=args.dataset_name,
            reg_or_cls="classification",
            train_sizes = [2000, 2000, 4000, 8000, 16000, 32000, 64000, 
                        128000, 256000, 464809],
            model_rerun_seeds = [0, 1, 2, 3, 4],
            save_dir=Path(args.save_dir),
    )