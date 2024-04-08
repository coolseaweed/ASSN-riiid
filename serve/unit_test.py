import pytest
from httpx import AsyncClient
from typing import Optional, Dict, Any, List
from loguru import logger
from backend.app import wrapped_app
import pandas as pd
from pathlib import Path
from backend.core.config import config
import asyncio
import random


version = "v1"
service_name = "student_perf"
base_url = "http://test"
data_path = Path(config.data_path)


def data_prep(data_path: Path = config.data_path) -> List[Dict]:
    if not data_path.exists():
        raise FileExistsError(data_path)

    df = pd.read_csv(data_path)

    def text_norm(text):
        return text.lower().replace(" ", "_").replace("-", "_")

    df.columns = list(map(text_norm, df.columns))
    target_column = "adaptivity_level"

    df = df.drop([target_column], axis=1)
    return df.to_dict(orient="records")


def generate_invalid_data(data_list):
    res = []
    for data in data_list:
        random_key = random.choice(list(data.keys()))
        data[random_key] = "invalid"
        res.append(data)
    return res


@pytest.fixture
def anyio_backend():
    return "asyncio"


data_list = data_prep(data_path)


@pytest.mark.anyio
async def test_predict():

    async with AsyncClient(app=wrapped_app, base_url=base_url) as client:

        futures = [client.post(f"{version}/{service_name}/predict", json=data) for data in data_list]

        results = await asyncio.gather(*futures)

        for response in results:
            assert response.status_code == 200, f"{response.json()},{response.status_code}"


@pytest.mark.anyio
async def test_batch():

    async with AsyncClient(app=wrapped_app, base_url=base_url) as client:
        response = await client.post(f"{version}/{service_name}/batch", json=data_list)

        assert response.status_code == 200

        assert len(response.json()) == len(data_list)


@pytest.mark.anyio
async def test_data():
    invalid_data_list = generate_invalid_data(data_list)

    async with AsyncClient(app=wrapped_app, base_url=base_url) as client:

        futures = [client.post(f"{version}/{service_name}/predict", json=data) for data in invalid_data_list]

        results = await asyncio.gather(*futures)

        for response in results:
            assert response.status_code == 422, f"{response.json()},{response.status_code}"
