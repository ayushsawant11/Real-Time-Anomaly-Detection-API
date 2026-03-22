from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the API
app = FastAPI(
    title="Fraud Detection API", 
    description="Real-time anomaly detection for financial transactions",
    version="1.0"
)

# 1. Define what a "Transaction" looks like (Data Validation)
class Transaction(BaseModel):
    transaction_id: str
    amount: float
    merchant_category: str
    location_distance_miles: float
    time_since_last_tx_minutes: float
    user_age_days: int

# Endpoint 1: The Root
@app.get("/")
def read_root():
    return {"status": "API is live", "message": "Welcome to the Anomaly Detection System backend."}

# Endpoint 2: The ML Prediction Endpoint (Using POST to receive data)
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Right now, we are just proving the API can receive the data.
    # Tomorrow, we will put the Scikit-Learn Anomaly Detection model right here!
    
    return {
        "message": "Transaction data received successfully!",
        "received_data": transaction,
        "is_anomaly_detected": False, # This is just a dummy response for now
        "risk_score": 0.01
    }