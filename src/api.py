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

import pandas as pd
import re

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords untuk bahasa Indonesia (lakukan sekali saja)
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Daftar stopwords dalam bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

def clean_tokenize_and_stem(df, text_column):
    # Mengubah teks menjadi huruf kecil
    df[text_column] = df[text_column].str.lower()  
    # Menhapus tanda baca, emotikon, dan karakter khusus yang tidak dibutuhkan
    df[text_column] = df[text_column].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x))
    # Menhapus angka pada teks
    df[text_column] = df[text_column].apply(lambda x: re.sub(r"\d+", "", x))
    # Menghapus stopwords dari teks
    df[text_column] = df[text_column].apply(lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))
    # Tokenisasi teks
    df['tokenize_content'] = df[text_column].apply(word_tokenize)
    # Stemming teks
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    df[text_column] = df[text_column].apply(stemmer.stem)
    
    return df

#API
app = FastAPI() 
config_data = util.load_config()
# Memuat TfidfVectorizer dari file
tfidf_vectorizer = function_2_data_processing.load_tfidf_vectorizer(config_data["tfidf_vectorizer"])
# Load model and make prediction])
model = joblib.load(config_data["model_final"])

class api_data(BaseModel):
    content: str

@app.get("/")
def home():
    return "Hello, FastAPI up!"    

@app.post("/predict/")
def predict(data: api_data):
    # Convert data api to dataframe
    config_data = util.load_config()
    #Input data
    df = pd.DataFrame(data.dict(), index=[0])
    
    # cleansing, tokenize and stemm
    df = clean_tokenize_and_stem(df, 'content')
    
    # Transformasi data tfidf
    df = tfidf_vectorizer.transform(df['content'])
    
    # Make prediction
    predicted_class = model.predict(df)

    # Pastikan predicted_class adalah array NumPy dan ambil elemen pertama
    predicted_class = predicted_class[0]

    # Mapping kelas ke konteks yang diberikan
    class_mapping = {
        0: "Positif",
        1: "Negatif"
    }

    # Menentukan hasil prediksi sesuai dengan kelas yang tepat
    if predicted_class in class_mapping:
        return {class_mapping[predicted_class]}
    else:
        return "Unknown Category"

    
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080, reload=True)