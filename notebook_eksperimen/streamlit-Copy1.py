import streamlit as st
import requests
import pandas as pd

# Judul aplikasi Streamlit
st.title("E-Commerce Product Classification")

# Input teks untuk prediksi
user_input = st.text_input("Masukkan judul produk:")

# Tombol untuk memulai prediksi berdasarkan input teks
if st.button("Predict Text"):
    # Membuat permintaan ke API FastAPI
    api_url = "http://api:8080/predict/"  # Sesuaikan dengan URL API FastAPI Anda
    #khusus kalau mau pakai 2 terminal di jupyterlab atau powershell ganti laman diatas dengan laman dibawah ini
    # api_url = "http://localhost:8080/predict/"  # Sesuaikan dengan URL API FastAPI Anda
    payload = {"title": user_input}
    response = requests.post(api_url, json=payload)

    # Memeriksa status respons
    if response.status_code == 200:
        result = response.text
        st.success(f"Hasil Prediksi: {result}")
    else:
        st.error("Terjadi kesalahan dalam memproses permintaan.")

# Divider
st.markdown("----")

# Judul untuk bagian unggah file CSV
st.header("Prediksi dari File CSV")

# Unggah file CSV
uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

# Tombol untuk memulai prediksi dari file CSV
if st.button("Predict CSV") and uploaded_file is not None:
    # Define `api_url` di sini juga
    api_url = "http://api:8080/predict/"  # Sesuaikan dengan URL API FastAPI Anda
    #khusus kalau mau pakai 2 terminal di jupyterlab atau powershell ganti laman diatas dengan laman dibawah ini
    # api_url = "http://localhost:8080/predict/"  # Sesuaikan dengan URL API FastAPI Anda

    # Membaca file CSV menjadi DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Memeriksa apakah DataFrame memiliki kolom 'title'
    if 'title' not in df.columns:
        st.error("File CSV harus memiliki kolom 'title'.")
    else:
        # Mengirim setiap judul produk ke API untuk prediksi
        predicted_classes = []
        for title in df['title']:
            payload = {"title": title}
            response = requests.post(api_url, json=payload)
            if response.status_code == 200:
                predicted_class = response.text
                predicted_classes.append(predicted_class)
            else:
                predicted_classes.append("Error")
        
        # Menambahkan kolom 'predicted_class' ke DataFrame
        df['predicted_class'] = predicted_classes
        
        # Menampilkan DataFrame hasil prediksi
        st.write(df)
