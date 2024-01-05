#import all realated libraries
#import libraries for data analysis
import numpy as np
import pandas as pd

# import library for visualization
import matplotlib.pyplot as plt

# import pickle and json file for columns and model file
import pickle
import json
import joblib
import yaml
import scipy.stats as scs

# import warnings for ignore the warnings
import warnings 
warnings.filterwarnings("ignore")

# library for model selection and models
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import xgboost as xgb

# evaluation metrics for classification model
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import GridSearchCV
import json
from datetime import datetime
from sklearn.metrics import classification_report
import uuid

from tqdm import tqdm
import pandas as pd
import os
import copy
import yaml
import joblib
import util as util
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

###################################################

def time_stamp() -> datetime:
    # Return current date and time
    return datetime.now()

def load_data_clean(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    #Read data X_train dan y_sm hasil dari data preparation
    X_train_clean = joblib.load(config_data["train_tfidf_set_path"][0])
    y_train = joblib.load(config_data["train_tfidf_set_path"][1])

    #Read data X_valid dan y_valid hasil dari data preparation
    X_valid_clean = joblib.load(config_data["valid_tfidf_set_path"][0])
    y_valid = joblib.load(config_data["valid_tfidf_set_path"][1])

    #Read data X_test dan y_test hasil dari data preparation
    X_test_clean = joblib.load(config_data["test_tfidf_set_path"][0])
    y_test = joblib.load(config_data["test_tfidf_set_path"][1])

    # Return 3 set of data
    return X_train_clean, y_train, X_valid_clean, y_valid, X_test_clean, y_test

def binary_classification_svc_tuned(x_train, y_train, x_valid, y_valid, x_test, y_test):
    # Set hyperparameters for Support Vector Classifier (SVC)
    svc_params = {'C': 0.5, 'gamma': 'scale', 'kernel': 'linear','probability':True}
    
    # Instantiate the SVC classifier
    svc_classifier = SVC(**svc_params, random_state=123)
    
    svc_classifier.fit(x_train, y_train)
    
    # Evaluate on the validation set
    valid_pred = svc_classifier.predict(x_valid)
    report = classification_report(y_valid, valid_pred, output_dict=True)
    valid_recall = report['weighted avg']['recall']
    valid_precision = report['weighted avg']['precision']
    print('Validation recall:', valid_recall)
    print('Validation precision:', valid_precision)
    
    # Evaluate on the test set
    test_pred = svc_classifier.predict(x_test)
    report = classification_report(y_test, test_pred, output_dict=True)
    test_recall = report['weighted avg']['recall']
    test_precision = report['weighted avg']['precision']
    print('Test recall:', test_recall)
    print('Test precision:', test_precision)
    
    return svc_classifier

def save_model_log(model, model_name, X_test, y_test):
    # generate unique id
    model_uid = uuid.uuid4().hex
    
    # get current time and date
    now = datetime.now()
    training_time = now.strftime("%H:%M:%S")
    training_date = now.strftime("%Y-%m-%d")
    
    # generate classification report
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # create dictionary for log
    log = {"model_name": model_name,
           "model_uid": model_uid,
           "training_time": training_time,
           "training_date": training_date,
           "classification_report": report}
    
    # menyimpan log sebagai file JSON
    with open('training_log/training_log.json', 'w') as f:
        json.dump(log, f)
        
if __name__ == "__main__":
    # Load configuration file
    config_data = util.load_config()
    
    # Load dataset
    X_train_clean, y_train, X_valid_clean, y_valid, X_test_clean, y_test = load_data_clean(config_data)
    
    # Modeling
    svc_best = binary_classification_svc_tuned(x_train=X_train_clean, y_train=y_train,
                                               x_valid=X_valid_clean, y_valid=y_valid,
                                               x_test=X_test_clean, y_test=y_test)
    
    # Save Log Model
    save_model_log(model = svc_best, model_name = "SVC Best", X_test = X_test_clean, y_test=y_test)
    
    # Save Model
    svc_best_cv = config_data["model_final"]
    with open(svc_best_cv, 'wb') as file:
        pickle.dump(svc_best, file)