##################################################
# Load and test the model
#################################################
import pickle
import pandas as pd
import logging
from utils import extract_features4test, is_valid_email, domain_extraction
import argparse
import numpy as np
import json
import os


def main(email, model_path, params_file_path, scaler_path, one_hot_encoder_path):
    logging.info('Provided email address: {}'.format(email))

    # Check if provided email is valid
    is_valid = is_valid_email(email)

    output = {
        'email': email,
        'age': 'unsure',
        'score': 0
    }
    if is_valid:
        email_ser = pd.Series([email], name='emails')
        # Extract age and store in a dataframe
        age = email_ser.apply(extract_features4test)

        d = {'estimated_age': age}
        age = pd.DataFrame(data=d)

        # Extract domain
        domain = email_ser.apply(domain_extraction)
        test_data = {
            'domain': [domain[0]]
        }
        test_df = pd.DataFrame(test_data)
        encoder = pickle.load(open(one_hot_encoder_path,'rb'))
        domain_encoded_test = encoder.transform(test_df)

        # Load the standard scaler saved during training
        loaded_scaler = pickle.load(open(scaler_path,'rb'))
        # Find scaled age
        scaled_age = loaded_scaler.transform(age)

        # Insert scaled age as the first element
        X_test_list = np.insert(domain_encoded_test, 0, scaled_age)
        # Load and assign the same feature names
        params = pickle.load(open(params_file_path, 'rb'))
        # Load the class categories
        age_classes = params['class_categories']
        X_test = pd.DataFrame([X_test_list], columns=params['feature_names'])

        # Load the classification model
        loaded_model = pickle.load(open(model_path,'rb'))
        # Make prediction
        y_pred = loaded_model.predict(X_test)
        predicted_class = age_classes[y_pred[0]]
        # Find the probabilities of all classes
        y_pred_proba = np.round(loaded_model.predict_proba(X_test), 2)
        # probability of predicted class
        score = y_pred_proba[0][y_pred][0]

        output['age'] = predicted_class
        output['score'] = float(score)

    else:
        logging.info('Provided email is invalid.')

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--email', type=str,
                        required=True, help="email address")
    parser.add_argument('-m', '--model_path', type=str,
                        required=True, help="Path of model pickle file")
    parser.add_argument('-i', '--info_metadata_path', type=str,
                        required=True, help="Path of metadata pickle file")
    parser.add_argument('-s', '--scaler_path', type=str,
                        required=True, help="Path of scaler pickle file")
    parser.add_argument('-o', '--one_hot_encoder', type=str,
                        required=True, help="Path of one_hot_encoder pickle file")
    parser.add_argument('-f', '--email_file_path', type=str,
                        required=False, help="Path of the email file")
    args = parser.parse_args()

    #Log file
    # Log file to be saved in 'results' directory
    path = "log_files"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory 'results' is created!")
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    filename= 'log_files/load_and_test_model' + date_time + '.log'

    logging.basicConfig(filename=filename, level=logging.WARN, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # set json output file path
    output_dir = "output"
    # Check whether the specified path exists or not
    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(output_dir)
        print("The new directory 'output' is created!")

    json_output_file_path = output_dir + '/' + 'output.json'
    l_json_output = []
    if args.email_file_path:
        with open(args.email_file_path) as file:
            while line := file.readline():
                json_output = main(line.rstrip(), args.model_path, args.info_metadata_path, args.scaler_path, args.one_hot_encoder)
                print(json_output)
                l_json_output.append(json_output)

    else:
        l_json_output = main(args.email, args.model_path, args.info_metadata_path, args.scaler_path, args.one_hot_encoder)
        print(l_json_output)

    # Write to json file
    with open(json_output_file_path, 'w') as f:
        json.dump(l_json_output, f)




