from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import torch.nn.functional as F
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

class TextData(BaseModel):
    text: str

@app.post("/classify/")
async def classify_text(data: TextData):
    inputs = tokenizer(data.text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1)
        confidence = torch.max(probabilities).item()
        predicted_class = torch.argmax(logits).item()
        
    return {
        "sentiment": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities.tolist()
    }