from .app import model, tokenizer
from .services import ModelService 
from fastapi import Depends


async def get_model_service():
    return ModelService(model, tokenizer)
