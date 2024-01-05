from fastapi import FastAPI, Form
from pydantic import BaseModel
import pandas as pd
from joblib import load
import joblib
import function_1_data_pipeline as function_1_data_pipeline
import function_2_data_processing as function_2_data_processing
import function_3_modeling as function_3_modeling
# import src.function_1_data_pipeline as function_1_data_pipeline
# import src.function_2_data_processing as function_2_data_processing
# import src.function_3_modeling as function_3_modeling
import util as util
import requests
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pickle
from tqdm import tqdm
import os
import copy
import yaml
from datetime import datetime
import uvicorn
import sys

#API
app = FastAPI() 
config_data = util.load_config()
# Memuat TfidfVectorizer dari file
tfidf_vectorizer = function_2_data_processing.load_tfidf_vectorizer(config_data["tfidf_vectorizer"])
# Load model and make prediction])
model = joblib.load(config_data["model_final"])

class api_data(BaseModel):
    title: str

@app.get("/")
def home():
    return "Hello, FastAPI up!"    

@app.post("/predict/")
def predict(data: api_data):
    # Convert data api to dataframe
    config_data = util.load_config()
    #Input data
    df = pd.DataFrame(data.dict(), index=[0])
    
    # Membersihkan teks dari karakter khusus dan mengonversi teks menjadi huruf kecil
    df = function_2_data_processing.preprocess_text(df, "title")
    
    # Transformasi data tfidf
    df = tfidf_vectorizer.transform(df['title'])
    
    # Make prediction
    predicted_class = model.predict(df)

    # Pastikan predicted_class adalah array NumPy dan ambil elemen pertama
    predicted_class = predicted_class[0]

    # Mapping kelas ke konteks yang diberikan
    class_mapping = {
        0: "Electronics",
        1: "Grocery & Gourmet Food",
        2: "Home & Kitchen",
        3: "Industrial & Scientific",
        4: "Office Products",
        5: "Tools & Home Improvement"
    }

    # Menentukan hasil prediksi sesuai dengan kelas yang tepat
    if predicted_class in class_mapping:
        return {class_mapping[predicted_class]}
    else:
        return "Unknown Category"

    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)

#    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
# "uvicorn src.api:app --reload"    
#Contoh bisa digunakan
"""
{
  "Geography": "Germany",
  "Gender": "Female",
  "CreditScore": 500,
  "Age": 60,
  "Tenure": 3,
  "Balance": 34562,
  "NumOfProducts": 2,
  "IsActiveMember": 0,
  "HasCrCard": 0,
  "EstimatedSalary": 11267
  
      # Menentukan hasil prediksi sesuai dengan kelas yang tepat
    if predicted_class in class_mapping:
        return f"Predicted Category: {class_mapping[predicted_class]}"
    else:
        return "Unknown Category"
}
"""