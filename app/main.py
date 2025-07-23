from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# CORS for frontend (Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model & tokenizer
model = load_model("model/abacus_lstm_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.post("/predict_fake_resume/")
async def predict(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    vector = vectorizer.transform([text]).toarray()
    padded = np.pad(vector, ((0, 0), (0, 250 - vector.shape[1])), mode='constant')

    pred = model.predict(padded)[0][0]
    label = "Fake Resume" if pred > 0.5 else "Genuine Resume"
    return {"prediction": label, "confidence": float(pred)}
