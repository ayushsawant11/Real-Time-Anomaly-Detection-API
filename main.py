from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load environment variables and configure Gemini
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
llm_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the API
app = FastAPI(title="Fraud Detection API with XAI", version="1.0")

# Load the trained ML model
try:
    ml_model = joblib.load('isolation_forest.pkl')
    print("✅ ML Model loaded successfully!")
except Exception as e:
    print("❌ Failed to load model.")

class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str
    location_distance_miles: float
    time_since_last_tx_minutes: float
    user_age_days: int

@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # 1. Extract features
    input_data = pd.DataFrame([{
        'amount': transaction.amount,
        'location_distance_miles': transaction.location_distance_miles,
        'time_since_last_tx_minutes': transaction.time_since_last_tx_minutes,
        'user_age_days': transaction.user_age_days
    }])
    
    # 2. Make the Prediction
    prediction = ml_model.predict(input_data)[0]
    is_anomaly = True if prediction == -1 else False
    
    explanation = "Transaction looks safe. No further review needed."

    # 3. Explainable AI Layer (Only triggers if fraud is detected)
    if is_anomaly:
        prompt = f"""
        You are an expert fraud analyst. A machine learning model just flagged this transaction as highly anomalous.
        Write a 2-sentence explanation for a human security team explaining WHY this looks suspicious based on the data.
        
        Data:
        Amount: ${transaction.amount}
        Distance from home: {transaction.location_distance_miles} miles
        Time since last transaction: {transaction.time_since_last_tx_minutes} minutes
        User account age: {transaction.user_age_days} days
        
        Keep it professional, concise, and direct.
        """
        
        try:
            response = llm_model.generate_content(prompt)
            explanation = response.text.strip()
        except Exception as e:
            explanation = "Fraud flagged, but AI explanation service is currently unreachable."

    return {
        "transaction_id": transaction.transaction_id,
        "is_anomaly_detected": is_anomaly,
        "ai_explanation": explanation
    }