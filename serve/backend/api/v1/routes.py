from fastapi import APIRouter
from loguru import logger

from backend.api.v1 import resources
from backend.core.config import config

version = "v1"
api_router = APIRouter()


def _add_router(router, prefix):
    """add a sub-router with an optional route prefix"""
    api_router.include_router(router, prefix=prefix, dependencies=[])


# add all sub-routers, for each task

logger.debug("starting routes for API")
_add_router(resources.health.router, prefix=f"/{version}")
_add_router(resources.student_perf.router, prefix=f"/{version}/student_perf")
