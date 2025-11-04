from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx, os

HF_ENDPOINT = os.environ.get("HF_ENDPOINT")  # ex: ton endpoint Spaces/Endpoint
app = FastAPI()

class PredictIn(BaseModel):
    image_url: str
    threshold: float = 0.9

@app.post("/predict")
async def predict(payload: PredictIn):
    if not HF_ENDPOINT:
        raise HTTPException(500, "HF_ENDPOINT not set")
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(HF_ENDPOINT, json=payload.dict())
        r.raise_for_status()
        return r.json()
