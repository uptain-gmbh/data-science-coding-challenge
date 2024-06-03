import re
import pandas as pd

def create_features(email):
    """
        Takes the email string and creates features for trained ML model to predict

        Args:
            - email (str): The email address to validate.

        Returns:
            - features (dict): Features created using email
    """
    feature_dict = {}
    try:
        # extract digits
        digit = re.findall(r'\d+',email)
        if len(digit) > 0:
            feature_dict['digits'] = digit[0]
        else:
            feature_dict['digits'] = 0

        # check if an email has a digit
        feature_dict['has_digit'] = int(bool(re.search(r'\d',email)))
        # username and domain split
        result = email.split('@')
        # check if the email has underscore/period special character in the username 
        username = result[0]
        feature_dict['has_underscore'] = int(bool(re.search(r'[_]',username)))
        feature_dict['has_period'] = int(bool(re.search(r'[.]',username)))
        feature_dict['username_length'] = len(username)
        # extract domain name
        feature_dict['domain'] = result[-1]
    
    except:
        print('Email format not supported')
    
    return feature_dict

def encode_domain(domain_name):
    main_domains = ('gmail.com','yahoo.com','protonmail.com','hotmail.com','aol.com','icloud.com','outlook.com','mail.com','tutanota.com')
    res_domain = domain_name
    if domain_name not in main_domains:
        res_domain = 'others'

    return res_domain

def is_valid_email(email):
    # Regular expression pattern for validating email addresses
    pattern = r'[^@]+@[^@]+\.[^@]+'

    # Check if the email matches the pattern
    if re.match(pattern, email):
        return True
    else:
        return False

def create_domain_features(feature_dict,true_domain):
    # remove the domain key itself
    del feature_dict['domain']
    domain_features = ['domain_aol.com', 'domain_gmail.com',
       'domain_hotmail.com', 'domain_icloud.com', 'domain_mail.com',
       'domain_others', 'domain_outlook.com', 'domain_protonmail.com',
       'domain_tutanota.com', 'domain_yahoo.com']

    for name in domain_features:
        if name.split('_')[-1] == true_domain:
            feature_dict[name] = 1
        else:
            feature_dict[name] = 0

    return feature_dict