# app/routes/predict.py

from fastapi import APIRouter, UploadFile, File
import pickle
import numpy as np
from tensorflow.keras.models import load_model

router = APIRouter()

# Load model and vectorizer only once
model = load_model("model/abacus_lstm_model.h5")

with open("model/tokenizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@router.post("/predict_fake_resume/")
async def predict_fake_resume(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Vectorize and pad
        vectorized = vectorizer.transform([text]).toarray()
        padded = np.pad(vectorized, ((0,0),(0,250-vectorized.shape[1])), mode='constant')

        # Prediction
        pred = model.predict(padded)[0][0]
        result = "Fake Resume" if pred > 0.5 else "Genuine Resume"

        return {"prediction": result, "confidence": float(pred)}

    except Exception as e:
        return {"error": str(e)}
