import random
import re
from datetime import datetime, date
import pandas as pd

import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb
from sklearn.svm import NuSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import pickle
import os
import time
import psutil
import re
from sklearn.preprocessing import StandardScaler


def is_valid_email(email):
    """Check if the email is a valid format."""
    regex = r'^[A-Za-z0-9._-]+[@][A-Za-z0-9.]+[.]\w+$'
    # If the string matches the regex, it is a valid email
    if re.fullmatch(regex, email):
        return True
    else:
        return False


def extract_info(x):

    # Initialize age
    age = 0
    username, domain = x.split('@')
    username_len = len(username)
    domain = domain.split('.')[0]

    # extract numbers in email
    numbers_list = re.findall(r'[0-9]+', x)

    if numbers_list:
        int_list = [int(numbers_list[e]) for e in range(0, len(numbers_list))]
        int_list = sorted(int_list, reverse=True)
        extracted_number = str(int_list[0])
        # If there are numbers in email compute age
        today = date.today()
        if len(str(extracted_number)) == 4:
            if int(extracted_number) <= today.year:
                if extracted_number[0] == '0':
                    print('Most significant digit is zero ')
                elif int(extracted_number[:1]) < 19:
                    # consider as day,month,year format
                    if int(extracted_number[2:]) >= 20:
                        birth_year = '19' + extracted_number[2:]
                    else:
                        birth_year = '20' + extracted_number[2:]
                    age = today.year - int(birth_year)
                else:
                    age = today.year - int(extracted_number)

            elif int(extracted_number) > today.year:
                if int(extracted_number[2:]) >= 20:
                    birth_year = '19' + extracted_number[2:]
                else:
                    birth_year = '20' + extracted_number[2:]
                age = today.year - int(birth_year)
        elif len(str(extracted_number)) == 3:
            if int(extracted_number) >= 20:
                birth_year = '19' + extracted_number[1:]
            else:
                birth_year = '20' + extracted_number[1:]
            age = today.year - int(birth_year)
        elif len(str(extracted_number)) == 2:
            if int(extracted_number) >= 0 and int(extracted_number) <= 6:
                birth_year = '20' + extracted_number
                age = today.year - int(birth_year)
            elif int(extracted_number) > 6 and int(extracted_number) <= 59:
                age = int(extracted_number)
            elif int(extracted_number) >= 60 and int(extracted_number) <= 99:
                birth_year = '19' + extracted_number
                age = today.year - int(birth_year)

        elif len(str(extracted_number)) == 1: # Assume young generation (random assignment)
            return pd.Series([username_len, random.randint(18, 30), 'young'])#18-30'
        else:
            #  'young' assumption if no other rules apply
            return pd.Series([username_len, random.randint(18, 30), 'young'])

        # Assign age group
        if age >= 51:
            return pd.Series([username_len, age, 'old']) #'51+'
        elif age >= 31:
            return pd.Series([username_len, age,'medium']) #31-50'
        elif age >= 18:
            return pd.Series([username_len, age, 'young']) #18-30'
        else:
            return pd.Series([username_len, age, 'minor']) #'0-17'

    # If there are no numbers in email, assign age group according to domain, assuming some domains used by certain age group
    # Assumption based on domain
    else:

        if domain in ['aol', 'hotmail', 'yahoo']:
            return pd.Series([username_len, random.randint(51, 100), 'old']) #'51+'
        elif domain in ['protonmail', 'outlook', 'gmail']:
            return pd.Series([username_len, random.randint(18, 50), random.choice(['young', 'medium'])])
        else: # any other domain assign to young group
            return pd.Series([username_len, random.randint(18, 50), 'young']) #'18-30'


def domain_extraction(x):
    username, domain = x.split('@')
    domain = domain.split('.')[0]
    return domain


def extract_features4test(x):

    username, domain = x.split('@')
    domain = domain.split('.')[0]
    today = date.today()
    age = 0

    numbers_list = re.findall(r'[0-9]+', x)
    if numbers_list:
        int_list = [int(numbers_list[e]) for e in range(0, len(numbers_list))]
        int_list = sorted(int_list, reverse=True)
        extracted_number = str(int_list[0])
        if len(str(extracted_number)) == 4:
            if int(extracted_number) <= today.year:
                if extracted_number[0] == '0':
                    print('Most significant digit is zero')
                elif int(extracted_number[:1]) < 19:
                    # consider as day,month,year format
                    if int(extracted_number[2:]) >= 20:
                        birth_year = '19' + extracted_number[2:]
                    else:
                        birth_year = '20' + extracted_number[2:]
                    age = today.year - int(birth_year)
                else:
                    age = today.year - int(extracted_number)

            elif int(extracted_number) > today.year:
                if int(extracted_number[2:]) >= 20:
                    birth_year = '19' + extracted_number[2:]
                else:
                    birth_year = '20' + extracted_number[2:]
                age = today.year - int(birth_year)
        elif len(str(extracted_number)) == 3:
            if int(extracted_number[1:]) >= 20:
                birth_year = '19' + extracted_number[1:]
            else:
                birth_year = '20' + extracted_number[1:]
            age = today.year - int(birth_year)
        elif len(str(extracted_number)) == 2:
            if int(extracted_number) >= 0 and int(extracted_number) <= 6:
                birth_year = '20' + extracted_number
                age = today.year - int(birth_year)
            elif int(extracted_number) > 6 and int(extracted_number) <= 59:
                age = int(extracted_number)
            elif int(extracted_number) >= 60 and int(extracted_number) <= 99:
                birth_year = '19' + extracted_number
                age = today.year - int(birth_year)
        elif len(str(extracted_number)) == 1: # Assume young generation (random assignment)
            return random.randint(18, 30)

        else:
            age = random.randint(18, 30)

    else: # assuming certain domains are used by particular age groups
        if domain in ['aol', 'hotmail', 'yahoo']:
            return random.randint(51, 100)
        elif domain in ['protonmail', 'outlook']:
            return random.randint(18, 50)
        elif domain in ['gmail']:
            return random.randint(18, 50)
        else: # any other domain assign to young group
            return random.randint(18, 50)
    return age

##################################################################
#                   Find best parameter
##################################################################

def get_bestparam(train_index, X, y, classifier):
    X_train = X.loc[train_index]
    y_train = y[train_index]

    # Initialize the standard scaler
    scaler = StandardScaler()
    #features_to_scale = ['username_len', 'estimated_age']
    features_to_scale = ['estimated_age']
    # Fit scalar on training data
    scaler.fit(X_train[features_to_scale])
    # Transform the training data
    X_train[features_to_scale] = scaler.transform(X_train[features_to_scale])

    le = LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)

    ###########################################################
    #                   Grid Search
    ##########################################################
    bestparams = {}
    if classifier == 'random_forest':
        # Use grid Search to find the best parameters of the classifier
        estimator = RandomForestClassifier(random_state=0)
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(1, 10, 1)
        }
    elif classifier == 'decision_tree':
        estimator = DecisionTreeClassifier(random_state=0)
        params = {
            'criterion': ['gini', 'entropy'],
            'max_depth': np.arange(1, 15, 1)
        }
    elif classifier == 'knn':
        estimator = KNeighborsClassifier()
        k_range = list(range(1, 8))
        weight_options = ['uniform', 'distance']
        params = dict(n_neighbors=k_range, weights=weight_options)

    elif classifier == 'logistic_regression':
        estimator = LogisticRegression(solver='saga', max_iter=10000)
        c_space = np.logspace(-3, 3, 7)
        print('c_space is ', c_space)
        params = dict(C=c_space)

    elif classifier == 'NuSVC':
        estimator = NuSVC(probability=True)
        params = {
            #'nu': [0.5, 0.8]#, 0.7, 0.9]
            'nu': [0.1]
        }
    elif classifier == 'lgbm_classifier':
        estimator = LGBMClassifier(objective="multiclass", num_class=4, random_state=0)
        params = {
            'num_leaves': [5, 20, 31],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100, 150]
        }
    elif classifier == 'xgboost_classifier':
        estimator = xgb.XGBClassifier(objective="multi:softmax", random_state=0)
        params = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01, 0.001],
            'subsample': [0.5, 0.7, 1]
        }

    #if classifier == 'random_forest' or classifier == 'decision_tree' or classifier == 'knn' or classifier == 'logistic_regression' or classifier == 'NuSVC' or classifier == 'lgbm_classifier' or classifier == 'xgboost_classifier':
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=params,
        scoring='f1_macro',
        cv=5,
        n_jobs=5,
        verbose=1
    )
    gs.fit(X_train, y_train)
    # Printing the best parameters
    logging.info('Best parameters from Gridsearch')
    logging.info(gs.best_params_)
    bestparams = gs.best_params_

    return bestparams

###################################################################
#                   Validation function
# This function computes and returns all the metrics of the
# classifier and fpr and tpr for each batch of data
##################################################################

def do_validation(train_index, test_index, X, y, classifier, bestparams, fold_number, metadata_dir):
    print('fold number: {}'.format(fold_number))
    X_train = X.loc[train_index]
    y_train = y[train_index]
    X_test = X.loc[test_index]
    y_test = y[test_index]

    # Initialize the standard scaler
    scaler = StandardScaler()
    features_to_scale = ['estimated_age']
    # Fit scalar on training data
    scaler.fit(X_train[features_to_scale])
    # Transform the training data
    X_train[features_to_scale] = scaler.transform(X_train[features_to_scale])
    X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])

    # Perform Label encoding for target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    le.fit(y_train)
    le_classes = le.classes_
    if fold_number == 1 and classifier == 'random_forest':
        logging.info('le.classes_: {}'.format(le_classes))
        #num_classes = len(le_classes)
        #print('num_classes: {}'.format(num_classes))

    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    ##############################################################
    #                   Classifier
    ##############################################################
    if classifier == 'random_forest':
        clf = RandomForestClassifier(criterion=bestparams['criterion'], max_depth=bestparams['max_depth'],
                                     random_state=0)
    elif classifier == 'decision_tree':
        clf = DecisionTreeClassifier(criterion=bestparams['criterion'], max_depth=bestparams['max_depth'],
                                     random_state=0)
    elif classifier == 'lgbm_classifier':
        clf = LGBMClassifier(**bestparams, random_state=0)
    elif classifier == 'xgboost_classifier':
        clf = xgb.XGBClassifier(**bestparams, random_state=0)
    elif classifier == 'NuSVC':
        clf = NuSVC(probability=True, nu=bestparams['nu'])
    elif classifier == 'knn':
        clf = KNeighborsClassifier(n_neighbors=bestparams['n_neighbors'], weights=bestparams['weights'])
    elif classifier == 'logistic_regression':
        clf = LogisticRegression(C=bestparams['C'],
                                 class_weight='balanced', solver='saga',
                                 penalty='l1', random_state=0, max_iter=10000)
    ############################################################
    # Fit the model
    clf.fit(X_train, y_train)
    if fold_number == 1:
        # save metadata (feature names and class label)
        # save the model for the first fold of cross validation
        params_path = metadata_dir + '/' +  'metadata.pkl'
        if classifier == 'lgbm_classifier':
            paramaters = {
                'feature_names': clf.feature_name_,
                'class_categories': le_classes
            }
        else:
            paramaters = {
                'feature_names': clf.feature_names_in_,
                'class_categories': le_classes
            }
        #print(paramaters)
        pickle.dump(paramaters, open(params_path, 'wb'))

        # Save standard scaler to a pickle file
        scaler_path = metadata_dir + '/' + 'scaler.pkl'
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)
        # save the model for the first fold of cross validation
        save_dir = "models"
        # Check whether the specified directory exists or not
        isExist = os.path.exists(save_dir)
        if not isExist:
            os.makedirs(save_dir)
            print(f'{save_dir} directory created')
        model_path = save_dir + '/' + classifier + '_model.pkl'
        pickle.dump(clf, open(model_path, 'wb'))

    # Make predictions
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)

    # Confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    print("confusion matrix: ")
    print(matrix)

    # Store the metrics
    results = {}
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    results['accuracy'] = [accuracy]
    results['roc_auc'] = [roc_auc]
    results['precision'] = [precision]
    results['recall'] = [recall]
    results['f1'] = [f1]

    # store the results to a dataframe
    df = pd.DataFrame.from_dict(results)
    return df


def get_cv_results(X, y, classifier, num_classes, metadata_dir):
    logging.info("Classifier : {}".format(classifier))
    bestparams = {}
    # Create StratifiedKFold object.
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1,)
    results = pd.DataFrame()

    fold_number = 0
    for train_index, test_index in skf.split(X, y):
        fold_number += 1
        # For the first fold perform gridsearch and find the best parameters for classifier
        if fold_number == 1:
            bestparams = get_bestparam(train_index, X, y, classifier)

        results_temp = do_validation(train_index, test_index, X, y, classifier, bestparams, fold_number, metadata_dir)
        results = pd.concat([results, results_temp], ignore_index=True)

    # store accuracy, auc, precision and recall from 10 fold cv in data frames
    accuracy_df = results['accuracy']
    auc_df = results['roc_auc']
    precision_df = results['precision']
    recall_df = results['recall']

    logging.info('-----------------------------------------------------')
    logging.info('    10 Fold Cross Validation Results                 ')
    logging.info('-----------------------------------------------------')
    logging.info('Accuracy mean  : {} '.format(str(np.round(accuracy_df.mean(), 2))))
    logging.info('Accuracy Std   : {} '.format(str(np.round(accuracy_df.std(), 2))))
    logging.info('AUC mean       : {} '.format(str(np.round(auc_df.mean(), 2))))
    logging.info('AUC Std        : {} '.format(str(np.round(auc_df.std(), 2))))
    logging.info('Precision mean : {} '.format(str(np.round(precision_df.mean(), 2))))
    logging.info('Precision Std  : {} '.format(str(np.round(precision_df.std(), 2))))
    logging.info('Recall mean    : {} '.format(str(np.round(recall_df.mean(), 2))))
    logging.info('Recall Std     : {} '.format(str(np.round(recall_df.std(), 2))))

    return results

def get_outlier_bounds(feature):

    statistics = {}
    mean = np.round(np.mean(feature), 2)
    median = np.round(np.median(feature), 2)
    min_value = np.round(feature.min(), 2)
    max_value = np.round(feature.max(), 2)
    Q1 = np.round(feature.quantile(0.25), 2)
    Q3 = np.round(feature.quantile(0.75), 2)
    # Interquartile range
    iqr = np.round(Q3 - Q1, 2)

    # store results in dictionary
    statistics['min'] = min_value
    statistics['mean'] = mean
    statistics['median'] = median
    statistics['max'] = max_value
    statistics['Q1'] = Q1
    statistics['Q3'] = Q3
    statistics['IQR'] = iqr

    # Lowerbound = Q1- 1.5 * IQR, Upperbound : Q3 + 1.5*IQR
    lower_bound = statistics['Q1'] - (1.5 * statistics['IQR'])
    upper_bound = statistics['Q3'] + (1.5 * statistics['IQR'])

    return lower_bound, upper_bound





