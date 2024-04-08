import pickle
from pathlib import Path
import pandas as pd
from typing import List
from backend.core.config import config
from backend.schema.student_perf import StudentInfo


def load(model_path: Path):
    """load model"""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileExistsError(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
        return model


def predict(data: StudentInfo):
    """inference single request"""
    x = pd.DataFrame([data.model_dump()])
    x_enc = one_hot_enc.transform(x)
    x_pred = model.predict(x_enc)
    y_pred = label_enc.inverse_transform(x_pred)

    return y_pred[0]


def batch(data: List[StudentInfo]):
    """inference batch request"""
    x = pd.DataFrame([s.model_dump() for s in data])
    x_enc = one_hot_enc.transform(x)
    x_pred = model.predict(x_enc)
    y_pred = label_enc.inverse_transform(x_pred)

    return y_pred


model = load(config.model_path)
one_hot_enc = load(config.one_hot_encoder_path)
label_enc = load(config.label_encoder_path)
