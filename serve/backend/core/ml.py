import pickle
from pathlib import Path
import pandas as pd
import sys

from typing import List

sys.path.append("../../")

from backend.core.config import config
from backend.schema.student_perf import StudentInfo


def load(model_path: Path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileExistsError(model_path)

    with open(model_path, "rb") as f:
        model = pickle.load(f)
        return model


def predict(data: StudentInfo):
    x = pd.DataFrame([data.model_dump()])
    x_enc = one_hot_enc.transform(x)
    x_pred = model.predict(x_enc)
    y_pred = label_enc.inverse_transform(x_pred)

    return y_pred[0]


def batch(data: List[StudentInfo]):
    x = pd.DataFrame([s.model_dump() for s in data])
    x_enc = one_hot_enc.transform(x)
    x_pred = model.predict(x_enc)
    y_pred = label_enc.inverse_transform(x_pred)

    return y_pred


# model = load(config.model_path)
model = load("/home/tom/PRJ/Riiid/question3/models/model.pkl")
one_hot_enc = load("/home/tom/PRJ/Riiid/question3/models/one_hot_encoder.pkl")
label_enc = load("/home/tom/PRJ/Riiid/question3/models/label_encoder.pkl")


if __name__ == "__main__":

    sample = StudentInfo(
        gender="Boy",
        age="1-5",
        education_level="College",
        institution_type="Government",
        it_student="No",
        location="No",
        load_shedding="High",
        financial_condition="Mid",
        internet_type="Mobile Data",
        network_type="2G",
        class_duration="1-3",
        self_lms="No",
        device="Tab",
    )

    test_dict = sample.model_dump()

    print(test_dict)

    t = pd.DataFrame([test_dict])
    print(t)

    t_enc = one_hot_enc.transform(t)

    x_predict = model.predict(t_enc)
    print(x_predict)
    y_predict = label_enc.inverse_transform(x_predict)
    print(y_predict)
    # from fastapi.encoders import jsonable_encoder

    # t = pd.DataFrame(jsonable_encoder(test_dict))
    # print(t)
    # print(test_dict)
    # df = pd.DataFrame([sample.dict()])
    # print(df)
    # # preprocessing()

    pass
