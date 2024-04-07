import os
from pathlib import Path
import pickle
from typing import Union, Tuple
import argparse


# library for feature engineering and EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# library for sampling
from imblearn.combine import SMOTEENN

# library for ML
import sklearn
import sklearn.linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="data.csv",
        help="train data path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="model.save",
        help="output model path",
    )

    parser.add_argument(
        "--model-dir",
        default="/models",
        help="output model directory",
    )

    parser.add_argument(
        "--data-dir",
        default="/data",
        help="data directory",
    )

    parser.add_argument(
        "--seed",
        default=1111,
        type=int,
        help="random seed",
    )
    
    parser.add_argument(
        "--target-column",
        default="Adaptivity Level",
        type=str,
        help="random seed",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="random seed",
    )


    return parser.parse_args()


def data_loading(fpath: Path) -> pd.DataFrame:

    if not fpath.exists():
        raise FileExistsError(fpath)

    return pd.read_csv(fpath)


def data_preprocessing(
    df: pd.DataFrame, target_column: str, test_size: float = 0.3, seed: int = 1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    Y = df[target_column]
    X = df.drop([target_column], axis=1)

    label_encoder = LabelEncoder()
    Y_enc = label_encoder.fit_transform(Y)  #
    X_enc = pd.get_dummies(X)  # one-hot encoding

    X_train, X_test, Y_train, Y_test = train_test_split(
        X_enc, Y_enc, test_size=test_size, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test


def model_training(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_val: pd.DataFrame,
    seed: int = 1,
    n_jobs:int=2
):

    model = LogisticRegression(
        C=1,
        class_weight="balanced",
        random_state=seed,
        multi_class="ovr",
        n_jobs=n_jobs,
        solver="lbfgs",
    ).fit(X_train, Y_train)

    print(f"Training Result: {model.score(X_train, Y_train)*100:.2f} %")
    print(f"Validation Result: {model.score(X_val, Y_val)*100:.2f} %")

    return model

def verify_model_performance(model:LogisticRegression,X:pd.DataFrame,Y:pd.DataFrame):
    X_pred = model.predict(X)
    print(f"Metric:\n{classification_report(Y,X_pred)}")
    
    
def save_model(model:LogisticRegression,fpath:Path, overwrite=True):
    
    flag = True if not fpath.exists() else False
    
    if overwrite:
        flag = True
    
    if flag :
        with open(fpath,'wb') as f:
            pickle.dump(model, f)
        print(f"success save model to '{fpath}'")
    else:
        raise FileExistsError(fpath)

def main():
    args = get_args()
    data_fpath = Path(args.data_dir) / args.input
    model_fpath = Path(args.model_dir) / args.output

    # data loading
    df = data_loading(data_fpath)

    # data preprocessing
    X_train, X_val, Y_train, Y_val = data_preprocessing(df,args.target_column)

    # model training
    model = model_training(X_train, X_val,Y_train,Y_val)

    # model performance verification
    verify_model_performance(model, X_val,Y_val)

    # save model
    save_model(model,model_fpath,overwrite=args.overwrite)
    

if __name__ == "__main__":
    main()
