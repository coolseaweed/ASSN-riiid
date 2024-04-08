from pathlib import Path
import pickle
from typing import Tuple
import argparse
import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        default="/data/data.csv",
        help="train data path",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="/models",
        help="output model directory",
    )
    parser.add_argument(
        "--seed",
        default=1111,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite models",
    )

    return parser.parse_args()


def data_loading(fpath: Path) -> pd.DataFrame:

    if not fpath.exists():
        raise FileExistsError(fpath)

    return pd.read_csv(fpath)


def data_preprocessing(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:

    def text_norm(text):
        return text.lower().replace(" ", "_").replace("-", "_")

    df.columns = list(map(text_norm, df.columns))

    target_column = "adaptivity_level"
    Y = df[target_column]
    X = df.drop([target_column], axis=1)

    return X, Y


def model_training(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_val: pd.DataFrame,
    seed: int = 1,
    n_jobs: int = 2,
) -> LogisticRegression:

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


def verify_model_performance(model: LogisticRegression, X: pd.DataFrame, Y: pd.DataFrame):
    X_pred = model.predict(X)
    print(f"Metric:\n{classification_report(Y,X_pred)}")


def save_model(model, fpath: Path, overwrite=True):

    flag = True if not fpath.exists() else False

    if overwrite:
        flag = True

    if flag:
        with open(fpath, "wb") as f:
            pickle.dump(model, f)
        print(f"success to save model: '{fpath}'")
    else:
        raise FileExistsError(fpath)


def main():
    args = get_args()
    data_fpath = Path(args.input)
    model_fpath = Path(args.output) / "model.pkl"
    one_hot_encoder_fpath = Path(args.output) / "one_hot_encoder.pkl"
    label_encoder_fpath = Path(args.output) / "label_encoder.pkl"

    # data loading
    df = data_loading(data_fpath)

    # data preprocessing
    X, Y = data_preprocessing(df)

    ## label encoding (Y)
    label_encoder = LabelEncoder()
    Y_enc = label_encoder.fit_transform(Y)

    ## one-hot encoding (X)
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    ohe = ColumnTransformer(
        transformers=[("ohe", one_hot_encoder, X.columns)],
        remainder="passthrough",
    )
    ohe.set_output(transform="pandas")
    ohe.fit(X)
    X_enc = ohe.transform(X)
    X_enc.head()

    X_train, X_val, Y_train, Y_val = train_test_split(X_enc, Y_enc, test_size=0.3, random_state=args.seed)

    # model training
    model = model_training(X_train, X_val, Y_train, Y_val)

    # model performance verification
    verify_model_performance(model, X_val, Y_val)

    # save models
    save_model(model, model_fpath, overwrite=args.overwrite)
    save_model(label_encoder, label_encoder_fpath, overwrite=args.overwrite)
    save_model(ohe, one_hot_encoder_fpath, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
