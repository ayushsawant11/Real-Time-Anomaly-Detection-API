# Real-Time Fraud & Anomaly Detection API

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)]()

An end-to-end MLOps pipeline that processes streaming transaction data to detect anomalies in milliseconds. This system uses an unsupervised machine learning model (Isolation Forest) wrapped in a high-speed FastAPI backend to bridge the gap between core data science and production-ready software engineering.

## 🚀 The Architecture

1. **Model Training Pipeline:** A dedicated script (`train_model.py`) generates synthetic transaction data, trains a Scikit-Learn Isolation Forest model to detect multivariate outliers, and serializes the model using `joblib`.
2. **Inference Layer:** A decoupled FastAPI backend (`main.py`) loads the pre-trained `.pkl` model into memory during server startup to eliminate cold-start latency.
3. **Real-Time Evaluation:** The API receives continuous JSON payloads simulating live credit card transactions, extracts the relevant features, and evaluates them for fraud with sub-200ms latency.

## ⚙️ Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI & Uvicorn | High-performance asynchronous API routing |
| **Machine Learning** | Scikit-Learn | Unsupervised anomaly detection (Isolation Forest) |
| **Data Processing** | Pandas & NumPy | Feature engineering and data structuring |
| **Model Serialization**| Joblib | Saving and loading the trained ML model |

## 🛠️ Local Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/anomaly-detection-api.git](https://github.com/yourusername/anomaly-detection-api.git)
cd anomaly-detection-api
