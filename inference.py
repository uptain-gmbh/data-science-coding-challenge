import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import sys
import json
import re
from utils import extract_email_domain, check_email_format, extract_digits

# Load the model and LabelEncoder
model_filename = 'best_lightgbm_classification_model.pkl'
model = joblib.load(model_filename)
le = joblib.load('label_encoder.pkl')

def preprocess_data(df):
    df['username'], df['domain'] = zip(*df['email'].apply(extract_email_domain))
    df['valid_email'] = df['domain'].apply(check_email_format)
    df['digits'], df['len_digits'] = zip(*df['username'].apply(extract_digits))
    df['digits'] = df['digits'].astype('int')   
    df['len_username'] = df['username'].apply(len)
    df['has_underscore'] = df['username'].apply(lambda x: 1 if '_' in x else 0)
    df['has_dot'] = df['username'].apply(lambda x: 1 if '.' in x else 0)
    df['has_hyphen'] = df['username'].apply(lambda x: 1 if '-' in x else 0)
    df['has_special_char'] = df['username'].apply(lambda x: 1 if re.search(r'\W', x) else 0)
    df['len_capital'] = df['username'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
    df['domain'] = df['domain'].astype('category')
    return df


def predict(df):
    X = df.drop(columns=['email', 'username'], errors='ignore')
    probs = model.predict_proba(X)
    predictions = model.predict(X)
    results = []
    for email, pred, prob in zip(df['email'], predictions, probs):
        age_class = le.inverse_transform([pred])[0]
        score = max(prob)
        result = {"email": email, "age": age_class, "score": float(score)}
        results.append(result)
    return results

def single_prediction(email):
    data = {'email': [email]}
    df = pd.DataFrame(data)
    df = preprocess_data(df)
    return predict(df)[0]

def batch_prediction(data_path):
    df = pd.read_csv(data_path)
    df = preprocess_data(df)
    return predict(df)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 inference.py <data.csv> or python3 inference.py <email>")
        sys.exit(1)
    
    input_value = sys.argv[1]

    if input_value.endswith('.csv'):
        # Batch prediction
        results = batch_prediction(input_value)
        for result in results:
            print(json.dumps(result))
    else:
        # Single prediction
        result = single_prediction(input_value)
        print(json.dumps(result))