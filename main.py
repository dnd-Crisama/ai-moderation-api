from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from model_loader import load_model
from inference import predict
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()          # download + load on startup
    yield

app = FastAPI(title='Toxic Comment Moderation API', lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv('ALLOWED_ORIGIN', '*')],
    allow_methods=['POST', 'GET'],
    allow_headers=['*'],
)

class PredictRequest(BaseModel):
    text: str

@app.get('/health')
def health():
    return { 'status': 'ok' }

@app.post('/predict')
def predict_endpoint(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail='text is required')
    result = predict(req.text)
    return result

# Response shape: { score: float, action: 'ALLOW'|'FLAG'|'DELETE', label: str }
