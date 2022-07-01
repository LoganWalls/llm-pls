from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from .model import load_model

model, tokenizer = load_model("gpt2")

from .views import api_router

app = FastAPI()
app.include_router(api_router)
