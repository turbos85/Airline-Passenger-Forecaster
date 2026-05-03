import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Model Definition (Must match the training architecture) ---
class PassengerRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

# --- 2. Load Artifacts ---
@st.cache_resource
def load_assets():
    # Load Scaler
    scaler = joblib.load("production_model/data_scaler.pkl")
    
    # Load Model
    model = PassengerRNN(hidden_size=64, num_layers=2)
    model.load_state_dict(torch.load("production_model/rnn_weights.pth", map_location=torch.device('cpu')))
    model.eval()
    return model, scaler

model, scaler = load_assets()

# --- 3. Streamlit UI ---
st.title("✈️ Airline Passenger Forecaster")
st.write("Input the last 12 months of passenger data to predict the next month.")

# User Input (Simulation of the last 12 months)
st.subheader("Monthly Input Data")
input_data = st.text_input("Enter 12 values separated by commas", 
                          "417, 391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432")

if st.button("Predict Next Month"):
    try:
        # Convert input to array
        raw_values = np.array([float(x.strip()) for x in input_data.split(',')])
        
        if len(raw_values) != 12:
            st.error("Please enter exactly 12 months of data.")
        else:
            # Preprocess
            scaled_input = scaler.transform(raw_values.reshape(-1, 1))
            tensor_input = torch.FloatTensor(scaled_input).unsqueeze(0) # Add batch dimension
            
            # Inference
            with torch.no_grad():
                prediction_scaled = model(tensor_input)
                prediction_final = scaler.inverse_transform(prediction_scaled.numpy())
            
            # Display Results
            st.success(f"Predicted Passengers for next month: **{int(prediction_final[0][0])}**")
            
            # Visualizing the trend
            full_series = np.append(raw_values, prediction_final)
            fig, ax = plt.subplots()
            ax.plot(range(1, 13), raw_values, label="Historical", marker='o')
            ax.plot([12, 13], [raw_values[-1], prediction_final[0][0]], label="Forecast", color='red', linestyle='--', marker='s')
            ax.set_ylabel("Passengers")
            ax.set_xlabel("Months")
            ax.legend()
            st.pyplot(fig)

    except ValueError:
        st.error("Invalid input. Ensure you only use numbers and commas.")