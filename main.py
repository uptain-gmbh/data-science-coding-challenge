import pandas as pd
import numpy as np
import joblib
from utils import create_features, encode_domain, is_valid_email, create_domain_features

def get_prediction(email):
    # prediction json
    prediction = {
    }
    # get features using email as a dictionary
    features_dict = create_features(email)
    # encode the email domain as a suitable feature 
    domain_name = features_dict['domain']
    features_dict['domain'] = encode_domain(domain_name)
    features_dict = create_domain_features(features_dict,features_dict['domain'])

    # create a pandas dataframe
    X = pd.DataFrame(features_dict, index=[0])
 
    # standard scaling of the numerical features
    scaler = joblib.load('standard_scaler.joblib')
    X[['digits','username_length']] = scaler.transform(X[['digits','username_length']])
    # load model and predict the input email
    model = joblib.load('model.joblib')
    y_pred = model.predict_proba(X)
    # get argmax for y_pred probabilities
    idx = np.argmax(y_pred)
    prediction['age'] = 'unsure'
    prediction['score'] = y_pred[0][idx]
    if idx == 0:
        prediction['age'] = 'young'
    elif idx == 1:
        prediction['age'] = 'medium'
    elif idx == 2:
        prediction['age'] = 'old'

    return prediction

if __name__ == '__main__':

    email = input('Enter email: ')
    if is_valid_email(email):
        prediction = get_prediction(email)
        print(prediction)
    else:
        print('Invalid email format')
        
        
