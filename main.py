import numpy as np
import pandas as pd
import re
import pickle
import json

# Load the trained KMeans model
with open('kmeans_model.pkl', 'rb') as model_file:
    kmeans = pickle.load(model_file)

# Define the label mapping
label_mapping = {0: 'medium', 1: 'young', 2: 'unsure', 3: 'old'}

def preprocessing_input(email: str) -> str:
    """
    Preprocess email address to separate local name and domain,
    correct domain name, and reassemble clean email address.

    Returns:
        str: str of cleaned email address.
    """
    # Define index words and domain corrections
    index_words = [
        'yahoo', 'hotmail', 'gmail', 'outlook', 'protonmail', 'icloud', 'aol', 
        'tutanota', 'sub.company', 'company', 'organization', 'college', 
        'sub.domain', 'mail'
    ]
  
    # Split each email into local_name and domain
    for word in index_words:
        if word in email:
            parts = email.split(word, 1)
            local_name = parts[0]
            domain = word + parts[1]
            break
        else:
            parts = email.split('@')
            local_name = parts[0]
            domain = parts[1]

    # correct so that domain is valid domain
    if 'yahoo' in domain and domain != 'yahoo.com':
        domain = 'yahoo.com'
    elif 'outlook' in domain and domain != 'outlook.com':
        domain = 'outlook.com'
    elif 'hotmail' in domain and domain != 'hotmail.com':
        domain = 'hotmail.com'
    elif 'gmail' in domain and domain != 'gmail.com':
        domain = 'gmail.com'
    elif 'protonmail' in domain and domain != 'protonmail.com':
        domain = 'protonmail.com'  
    elif 'icloud' in domain and domain != 'icloud.com':
        domain = 'icloud.com'
    elif 'aol' in domain and domain != 'aol.com':
        domain = 'aol.com'   
    elif 'tutanota' in domain and domain != 'tutanota.com':
        domain = 'tutanota.com'
    elif 'sub.company' in domain and domain != 'sub.company.com':
        domain = 'sub.company.com'
    elif 'company.co.' in domain and domain != 'company.co.uk':
        domain = 'company.co.uk'
    elif 'organization' in domain and domain != 'organization.org':
        domain = 'organization.org'
    elif 'college' in domain and domain != 'college.edu':
        domain = 'college.edu'
    elif 'sub.domain' in domain and domain != 'sub.domain.edu':
        domain = 'sub.domain.edu'
    elif 'mail' in domain and domain != 'gmail.com' and 'protonmail' not in domain and 'mail.service' not in domain:
            domain = 'gmail.com'
    else:
        domain = 'gmail.com'
       

    # Remove invalid characters from local names
    def remove_invalid_chars(email_part: str) -> str:
        invalid_characters = [' ', ',', ':', ';', '<', '>', '(', ')', '[', ']', '\\', '"', '@']
        for char in invalid_characters:
            email_part = email_part.replace(char, '')
        return email_part

    corrected_ln = remove_invalid_chars(local_name)
    corrected_dom = remove_invalid_chars(domain)

    # Combine local names and domains to create cleaned email addresses
    corrected_email = corrected_ln + '@' + corrected_dom
    corrected_email = corrected_email.lower()

    return corrected_email


# Function to extract features from email
def format_features(preprocessed_email: str) -> dict:
 
    # Extract username and domain
    email_username, email_domain = preprocessed_email.split('@')
    # Extract number from username
    numbers_in_username = re.findall(r'\d+', email_username)
    
    email_providers = ['aol.com', 'college.edu', 'company.co.uk', 'gmail.com', 'hotmail.com', 'icloud.com', 'mail.service.com',
                       'organization.org', 'outlook.com', 'protonmail.com', 'sub.company.com', 'sub.domain.edu', 'tutanota.com',
                       'yahoo.com']

    # Create provider feature dictionary
    provider_features = {f'email_provider_{provider}': int(email_domain == provider) for provider in email_providers}
  
    # Create feature dictionary
    features = {
        # 'email': preprocessed_email,
        'username_length': len(email_username),
        'has_numbers': bool(numbers_in_username),
        'has_underscore': '_' in email_username,
        'has_dot': '.' in email_username,
        **provider_features
    }
    return features


def create_features_single_input(clean_df: pd.DataFrame) -> pd.DataFrame:
    # Extract features for each email in the DataFrame
    feature_data = clean_df['email'].apply(format_features).tolist()
    feat_df = pd.DataFrame(feature_data)
    feat_df = feat_df.astype(int)

    feat_df[['username_length']] = ((feat_df[['username_length']])-4)/26
    feat_df['email_provider_gmail.com'] *= .5
    feat_df['email_provider_outlook.com'] *= .75
    feat_df['email_provider_yahoo.com'] *= .8
    feat_dataframe = feat_df
    
    return feat_dataframe


def main():
    # loop to keep running until 'exit' is written
    while True:
        email = input("please enter an email address ('exit' to quit): ")
        if email != 'exit':
            try:
                input_email = create_features_single_input(pd.DataFrame({'email': [preprocessing_input(email)]}))
                # Predict the cluster
                cluster_label = kmeans.predict(input_email)[0]
                cluster_name = label_mapping[cluster_label]
                cluster_center = kmeans.cluster_centers_[cluster_label]
                distance_to_center = np.sqrt(np.sum((input_email.values - cluster_center) ** 2))

                # calculate the distance to the other clusters, use the closest one to determine the probability of
                # that it is in the correct one
                distances = []
                for other_cluster in range(kmeans.n_clusters):
                    if other_cluster != cluster_label:
                        other_center = kmeans.cluster_centers_[other_cluster]
                        distance_to_other = np.sqrt(np.sum((input_email.values - other_center) ** 2))
                        distances.append(distance_to_other)

                # calculate the probability that it is in the correct one
                score = 1 - (distance_to_center/(distance_to_center+min(distances)))
                
                # Output results
                results = {"age": cluster_name, "score": round(score, 2)}
                # Convert results to JSON and print as a single line
                print(json.dumps(results))
            except ValueError:
                print("Please enter a valid email")
        else:
            break
        

if __name__ == '__main__':
    main()