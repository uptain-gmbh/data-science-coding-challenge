###########################################################
#   Data preprocessing and Model building
###########################################################
# Import packages
import pandas as pd
import logging
from utils import is_valid_email, domain_extraction, get_cv_results, extract_info
import pickle
import os
import time
import psutil
import argparse
from sklearn.preprocessing import OneHotEncoder


def main(emails_txt_file):

    ########################################
    # Log file
    ########################################
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
    filename='log_files/training' + date_time + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    print('Log file can be found in {}'.format(filename))
    ########################################
    # Read emails.txt file
    df = pd.read_csv(emails_txt_file, sep=" ", names=["emails"])
    logging.info('There are {} emails in {}'.format(df.shape[0], emails_txt_file))
    #########################################
    # Data Cleaning
    #########################################
    # Find duplicate rows
    duplicates = df[df.duplicated()]
    logging.info('Number of duplicated emails: {} '.format(duplicates.shape[0]))
    # Drop duplicates
    df = df.drop_duplicates()
    logging.info('Number of emails after removing duplicates: {} '.format(df.shape[0]))

    # Check the validity of email
    df['valid_email'] = df['emails'].apply(is_valid_email)
    invalid_email = df[df['valid_email'] == False]
    logging.info('There are {} invalid emails'.format(invalid_email.shape[0]))
    # removed invalid emails
    df = df[df['valid_email'] == True]
    logging.info('Number of emails after removing invalid emails: {} '.format(df.shape[0]))

    ##############################################
    # Feature Engineering
    ###############################################
    df[['username_len', 'estimated_age', 'age_range']] = df['emails'].apply(extract_info)
    # Extract domain
    df['domain'] = df['emails'].apply(domain_extraction)
    # Reset the index of dataframe
    df = df.reset_index(drop=True)
    ###########################################
    # One Hot Encoding for domain feature
    ###########################################
    # Split the data
    # X = df[['username_length', 'estimated_age', 'domain']].copy()
    X = df[['estimated_age', 'domain']].copy()

    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    #Initialize OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    # Apply one-hot encoding to the categorical columns
    one_hot_encoded = encoder.fit_transform(df[categorical_columns])

    #Create a DataFrame with the one-hot encoded columns
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))

    # save one hot encoder to file
    metadata_dir = "saved_metadata"
    save_encoder_path = metadata_dir + '/' + 'encoder.pkl'
    # Check whether the specified directory exists or not
    isExist = os.path.exists(metadata_dir)
    if not isExist:
        os.makedirs(metadata_dir)
        print(f'{metadata_dir} directory created')
        pickle.dump(encoder, open(save_encoder_path, 'wb'))
        loaded_encoder = pickle.load(open(save_encoder_path,'rb')) # open(model_path,'rb')
    ####################################################
    # Concatenate the one-hot encoded dataframe with the original dataframe
    X_encoded = pd.concat([X, one_hot_df], axis=1)

    # Drop the original categorical columns
    X_encoded = X_encoded.drop(categorical_columns, axis=1)

    # Target variable
    y = df['age_range']

    # Run the 10 fold cross validation for the following classifiers
    classifiers = ['random_forest', 'lgbm_classifier', 'xgboost_classifier', 'knn', 'logistic_regression', 'NuSVC']
    num_classes = 4
    for clf in classifiers:
        logging.info('\n')
        # Measure the start time and memory before cross validation
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / (1024 * 1024)  # in MB

        # Get CV results
        get_cv_results(X_encoded, y, clf, num_classes, metadata_dir)

        # Measure the end time and memory after training
        end_time = time.time()
        end_memory = process.memory_info().rss / (1024 * 1024)  # in MB
        # Print the resource usage
        logging.info('10 Fold Cross validation time and Memory usuage are as follows:')
        logging.info('time: {:.2f} seconds.'.format(end_time - start_time))
        logging.info('Memory usage: {:.2f} MB'.format(end_memory - start_memory))
        logging.info('-----------------------------------------------------')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--emails_txt_file', type=str,
                    required=True, help="email address")
    args = parser.parse_args()
    main(args.emails_txt_file)