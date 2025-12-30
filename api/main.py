from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import sys
import os


sys.path.append(os.path.join(os.getcwd(), 'src'))
from linear_regression import LinearRegression

app = FastAPI()

# Load model, scaler and columns at startup
model = LinearRegression()
model.load()

with open('models/scaler.pkl', 'rb') as f:
  scaler = pickle.load(f)

with open('models/columns.pkl', 'rb') as f:
  model_columns = pickle.load(f)

class InsuranceInput(BaseModel):
  age: int
  sex: str
  bmi: float
  children: int
  smoker: str
  region: str

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

  prediction = model.predict(df)

  return {"prediction": prediction[0]}
