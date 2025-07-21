from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from io import StringIO
import pandas as pd

app = FastAPI()

# Enable CORS (for Streamlit frontend to talk to backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ABACUS-LSTM model and tokenizer
model = load_model("model/abacus_lstm_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.post("/predict_fake_resume/")
async def predict_fake_resume(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Vectorize and preprocess
        vectorized = vectorizer.transform([text]).toarray()
        padded = np.pad(vectorized, ((0,0),(0,250-vectorized.shape[1])), mode='constant')

        # Predict
        pred = model.predict(padded)[0][0]
        result = "Fake Resume" if pred > 0.5 else "Genuine Resume"

        return {"prediction": result, "confidence": float(pred)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
