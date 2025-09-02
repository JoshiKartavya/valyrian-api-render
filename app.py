# app.py

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import torch
from fastapi.middleware.cors import CORSMiddleware

# Initialize the FastAPI app
app = FastAPI(title="High Valyrian Translator API")

# Configure CORS to allow all origins (simplest for now)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Your Hugging Face model ID
MODEL_ID = "royalbhudev/t5-small-english-to-valyrian"
device = 0 if torch.cuda.is_available() else -1

print("Loading Valyrian translator pipeline...")
translator = pipeline("translation", model=MODEL_ID, device=device)
print("✅ Pipeline loaded successfully.")

# Pydantic models for request and response validation
class TranslationRequest(BaseModel):
    text: str

class TranslationResponse(BaseModel):
    translation_text: str

# Health check endpoint
@app.get("/", tags=["Health Check"])
def read_root():
    return {"status": "API is running."}

# Translation endpoint
@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
def translate_text(request: TranslationRequest):
    prefixed_text = "translate English to Valyrian: " + request.text
    result = translator(prefixed_text)
    translated_text = result[0]['translation_text']
    return {"translation_text": translated_text}