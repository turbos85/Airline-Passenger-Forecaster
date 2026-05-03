# ✈️ Airline Passenger Forecaster

A time-series forecasting application using a Simple RNN to predict monthly airline passenger traffic.

## 📁 Project Structure

```text
├── app.py                   # The main Streamlit application for inference
├── requirements.txt         # List of Python dependencies
└── production_model/        # Folder containing serialized artifacts
    ├── rnn_weights.pth      # Trained PyTorch RNN model weights
    └── data_scaler.pkl      # MinMaxScaler instance for data normalization

## 🚀 Features
- **Recurrent Neural Network:** Built with PyTorch to capture sequential patterns.
- **MLOps Pipeline:** Includes data normalization, sliding window transformation, and experiment tracking.
- **Deployment:** Interactive UI built with Streamlit.

## 🛠️ Installation & Setup
1. Clone the repo: `git clone https://github.com/your-username/repo-name.git`
2. Create venv: `python -m venv venv`
3. Activate venv: `source venv/bin/activate` (or `.\venv\Scripts\activate` on Windows)
4. Install: `pip install -r requirements.txt`

## 📊 Dataset
Uses the [Air Passengers dataset from Kaggle](https://www.kaggle.com/datasets/rakannimer/air-passengers).

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://turbos85-airline-passenger-forecaster-app-jhx3oe.streamlit.app/)