import orjson
from fastapi import APIRouter, FastAPI, JSONResponse

from .model import load_model

model, tokenizer = load_model("gpt2")

from .views import api_router


class ORJSONResponse(JSONResponse):
    media_type = "application/json"

    def render(self, content) -> bytes:
        return orjson.dumps(content)


app = FastAPI(default_response_class=ORJSONResponse)
app.include_router(api_router)
