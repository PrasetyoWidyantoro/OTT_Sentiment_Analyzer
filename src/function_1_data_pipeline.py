from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import os
import joblib
import yaml
from datetime import datetime
import util as util
import numpy as np
# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")
# import pickle and json file for columns and model file
import pickle
import json
import copy
import re


def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat([pd.read_csv(raw_dataset_dir + i), raw_dataset])
    
    # Return raw dataset
    return raw_dataset


######################################################################################################################    

if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()
    
    # Read all raw Dataset
    data = read_raw_data(config_data)
    
    # Mengubah tipe data kolom "at" menjadi datetime dengan format yang diinginkan
    data['at'] = pd.to_datetime(data['at'], format='%Y-%m-%d %H:%M:%S')
    # Ekstraksi tanggal, jam, dan menit
    data['event_date'] = data['at'].dt.date
    data['event_hour'] = data['at'].dt.hour
    data['event_minute'] = data['at'].dt.minute
    # Menambahkan event_date
    data['event_date'] = pd.to_datetime(data['event_date'])
    # sortir
    data = data.sort_values(by='event_date')
    #khusus untuk data preproc
    data_for_preproc = data[['content','sentiment']]
    
    # Save Data
    util.pickle_dump(data, config_data["data_preparation_result_path"][0])
    util.pickle_dump(data_for_preproc, config_data["data_preparation_result_path"][1])
    
    print("Data Pipeline passed successfully.")
