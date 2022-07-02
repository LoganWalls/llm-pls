from fastapi import APIRouter, FastAPI

from . import config
from .model import load_model

model, tokenizer = load_model(config.model_name)

from .views import api_router

app = FastAPI()
app.include_router(api_router)
