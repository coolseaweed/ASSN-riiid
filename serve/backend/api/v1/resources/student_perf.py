from fastapi import APIRouter, status
from typing import List
from backend.schema.student_perf import StudentInfo, PredictResponse
from loguru import logger
from backend.core import ml

router = APIRouter()


@router.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK, tags=["student performance"])
async def predict(request: StudentInfo):
    # async def predict():
    """predict student adaptive level based on informations"""

    logger.info(f"request: {request}")

    pred = ml.predict(request)

    logger.info(f"prediction: {pred}")
    response = PredictResponse(adaptivity_level=pred)

    return response


@router.post(
    "/batch", response_model=List[PredictResponse], status_code=status.HTTP_200_OK, tags=["student performance"]
)
async def batch(request: List[StudentInfo]):
    """predict student adaptive level based on informations"""

    logger.info(f"request: {request}")

    pred = ml.batch(request)

    response = [PredictResponse(adaptivity_level=p) for p in pred]
    logger.info(f"prediction: {pred}")

    return response
