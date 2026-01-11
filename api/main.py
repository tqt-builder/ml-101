from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import sys
import os


sys.path.append(os.path.join(os.getcwd(), 'src'))
from linear_regression import LinearRegression
from neural_network import SimpleNeuralNetwork

# Initialize FastAPI

app = FastAPI()

# Load model, scaler and columns at startup
with open('models/scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)

with open('models/columns.pkl', 'rb') as f:
  model_columns = pickle.load(f)

with open('models/scaler_y.pkl', 'rb') as f:
  scaler_y = pickle.load(f)

model = LinearRegression()
model.load()

nn_model = SimpleNeuralNetwork(n_features=len(model_columns))
nn_model.load()

class InsuranceInput(BaseModel):
  age: int
  sex: str
  bmi: float
  children: int
  smoker: str
  region: str
  model_type: str = "linear"

@app.post("/predict")
def predict(data: InsuranceInput):
  df = pd.DataFrame([data.model_dump()])
  df = pd.get_dummies(df)
  for col in model_columns:
    if col not in df.columns:
      df[col] = 0
  df = df[model_columns]

  num_cols = ['age', 'bmi', 'children']
  df[num_cols] = scaler.transform(df[num_cols])

  if data.model_type == "nn":
    prediction_scaled = nn_model.predict(df)
    prediction = float(scaler_y.inverse_transform(prediction_scaled)[0][0])
  else:
    prediction = float(model.predict(df)[0][0])

  return {"prediction": prediction}
