from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import copy
import joblib
import yaml
from datetime import datetime
import pandas as pd
import numpy as np
import json
import nltk
nltk.download('stopwords')
import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
import re
import pickle
import datetime
import util as util
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from gensim import corpora, models
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
# Download data NLTK
nltk.download('wordnet')
import matplotlib.pyplot as plt
from nltk.tag import CRFTagger
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Daftar stopwords dalam bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))
# Load the stopwords from the NLTK library
config_dir = "config/config.yaml"
config_data = util.load_config()

############################################
# Fungsi untuk menghapus stopwords dari teks
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def clean_text(df, text_field, new_text_field_name):
    # Mengubah teks menjadi huruf kecil
    df[new_text_field_name] = df[text_field].str.lower()  
    # Menhapus tanda baca, emotikon, dan karakter khusus yang tidak dibutuhkan
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", x))
    # Menhapus angka pada teks
    df[new_text_field_name] = df[new_text_field_name].apply(lambda x: re.sub(r"\d+", "", x))
    return df

# Fungsi untuk menghapus stopwords dari teks
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

# Fungsi untuk tokenisasi teks
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

#-----------------STEMMING -----------------
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# proses stemming
def stemmed_wrapper(term):
    return stemmer.stem(term)

def get_stemmed_term(document):
    return [term_dict[term] for term in document]

# Inisialisasi dan fit TfidfVectorizer
def fit_tfidf_vectorizer(X_train):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_vectorizer.fit(X_train['stemmed_content'])
    
    # Simpan TfidfVectorizer
    with open(config_data["tfidf_vectorizer"], 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    
    return tfidf_vectorizer

# Load TfidfVectorizer dari file
def load_tfidf_vectorizer(file_path):
    with open(file_path, 'rb') as f:
        tfidf_vectorizer = pickle.load(f)
    return tfidf_vectorizer


if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()
    #Import data yang sudah dibuat sebelumnya dari proses data preparation
    data_preproc = util.pickle_load(config_data["data_preparation_result_path"][1])
    # Memanggil fungsi clean_text
    data_preproc = clean_text(data_preproc, 'content', 'cleaned_content')
    # Terapkan fungsi pada kolom yang bersih
    data_preproc['cleaned_content_stopwords'] = data_preproc['cleaned_content'].apply(remove_stopwords)
    # Terapkan fungsi pada kolom yang sudah dihapus stopwords
    data_preproc['tokenize_content'] = data_preproc['cleaned_content_stopwords'].apply(tokenize_text)
    #memisahkan file eksekusinya setelah pembacaaan term selesai
    # Proses stemming
    term_dict = {}
    hitung = 0

    # Buat kamus kata hasil stemming
    for document in data_preproc['tokenize_content']:
        for term in document:
            if term not in term_dict:
                term_dict[term] = stemmed_wrapper(term)
                
    data_preproc['stemmed_content'] = data_preproc['tokenize_content'].apply(lambda x:' '.join(get_stemmed_term(x)))
    # Ganti nilai 'Negatif' dengan 1 dan 'Positif' dengan 0 pada kolom 'sentiment'
    data_preproc['sentiment'] = data_preproc['sentiment'].replace({'Negatif': 1, 'Positif': 0})
    # Splitting X dan y
    X = data_preproc['stemmed_content']
    y = data_preproc['sentiment']
    #Split Data 70% training 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = 0.3,
                                                        random_state = 123,
                                                        stratify=y)
    # Split data train menjadi train dan validation set
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, 
                                                        test_size=0.4, 
                                                        random_state=42,
                                                        stratify = y_test)
    # Reset index
    X_train = X_train.reset_index()
    X_test  = X_test.reset_index()
    X_valid = X_valid.reset_index()
    
    # Fit tfidf menggunakan data train
    X_train_fit = fit_tfidf_vectorizer(X_train)
    # Memuat TfidfVectorizer dari file
    tfidf_vectorizer = load_tfidf_vectorizer(config_data["tfidf_vectorizer"])
    # Transformasi data pelatihan
    X_train_tfidf = tfidf_vectorizer.transform(X_train['stemmed_content'])
    # Transformasi data uji
    X_test_tfidf = tfidf_vectorizer.transform(X_test['stemmed_content'])
    # Transformasi data validasi
    X_valid_tfidf = tfidf_vectorizer.transform(X_valid['stemmed_content'])
        
    # Save Data
    util.pickle_dump(X_train_tfidf, config_data["train_tfidf_set_path"][0])
    util.pickle_dump(y_train, config_data["train_tfidf_set_path"][1])
        
    util.pickle_dump(X_valid_tfidf, config_data["valid_tfidf_set_path"][0])
    util.pickle_dump(y_valid, config_data["valid_tfidf_set_path"][1])

    util.pickle_dump(X_test_tfidf, config_data["test_tfidf_set_path"][0])
    util.pickle_dump(y_test, config_data["test_tfidf_set_path"][1])