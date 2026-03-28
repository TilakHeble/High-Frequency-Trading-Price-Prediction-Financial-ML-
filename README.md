# 📈 High-Frequency Trading LOB Forecasting (Dual-Output ML/DL)

## 🚀 Overview
This project implements a **dual-output machine learning pipeline** for **Limit Order Book (LOB) forecasting** in a High-Frequency Trading (HFT) setting.

The system simultaneously predicts:
- 📊 **Future mid-price (Regression)**
- 📉 **Price direction (Classification)**

This multi-task approach improves trading signal generation by combining **magnitude + direction**, making it more practical for real-world trading decisions.

---

## 📂 Dataset
- Amazon (AMZN) high-frequency LOB data  
- 10 levels of bid/ask prices and volumes  
- ~150,000 time-ordered observations used  

Key extracted features:
- Mid-price  
- Bid-ask spread  
- Price & volume across levels  

---

## 🧠 Feature Engineering

### 🔹 Microstructure Features
- Ask-Bid spread per level  
- Volume imbalance  
- Price differences  

### 🔹 Temporal Features
- Lagged mid-prices (t-1, t-2)  
- Rolling mean (5, 10)  
- Rolling volatility (std)  

These features allow models to capture both:
- Instantaneous order book state  
- Short-term market dynamics  

---

## 🎯 Problem Formulation

### 1. Regression Task
Predict next-step mid-price (log-transformed)

### 2. Classification Task
Predict direction:
- 1 → Price increase  
- 0 → No increase  

### ⭐ Advanced Design
Multi-task learning (shared representation for both outputs)

---

## ⚙️ Model Architectures

The following models were implemented and compared:

- MLP (Shallow, Deep, Very Deep)
- LSTM (Bidirectional)
- CNN (1D Convolution)
- CNN-LSTM Hybrid
- Ensemble (LSTM + CNN-LSTM)

Each model has:
- Regression head → Linear + MSE  
- Classification head → Sigmoid + Binary Cross-Entropy  

---

## 🏋️ Training Strategy

- TimeSeriesSplit (5-fold CV) to prevent leakage  
- Sliding window (look_back = 10) for sequential models  
- Class weighting for imbalance  
- Early stopping (patience = 2)  

### Loss Function
```python
loss = {
    "regression": "mse",
    "classification": "binary_crossentropy"
}
