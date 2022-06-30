import time

from fastapi import APIRouter, Depends

from .app_services import get_model_service
from .model import CompletionParams
from .services import ModelService

api_router = APIRouter()


@api_router.post("/completion")
async def generate_completion(
    params: CompletionParams, model_service: ModelService = Depends(get_model_service)
):
    result = model_service.generate_completion(params)
    return dict(choices=[result], created=int(time.time()))
