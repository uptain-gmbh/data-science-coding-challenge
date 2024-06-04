import json
import os
import re
from datetime import datetime

import numpy as np
import whois
from transformers import BertTokenizer, BertModel

from utils import softmax

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')


def extract_username(email):
    """
    Extracts the username part from an email address.

    Args:
        email (str): The email address.

    Returns:
        str: The username part of the email.
    """
    return email.split('@')[0]


def extract_year_from_username(username):
    """
    Extracts potential year from the username part of an email address.

    This function searches for numeric patterns in the username that could represent a year.
    It supports 1, 2, and 4-digit year formats and attempts to convert them to a full year.

    Args:
        username (str): The username part of the email.

    Returns:
        int: The extracted year or 0 if no valid year is found.
    """
    numbers = re.findall(r'\d+', username)
    current_year = datetime.now().year
    valid_years = []

    for number in numbers:
        num_len = len(number)
        number = int(number)
        if num_len == 4 and 1900 <= number <= current_year:
            valid_years.append(number)
        elif num_len == 2:
            if 60 <= number <= 99:
                valid_years.append(1900 + number)
            elif 0 <= number <= int(str(current_year)[-2:]) - 10:
                valid_years.append(2000 + number)
        elif num_len == 1:
            if number == 0:
                valid_years.append(2000)
            elif 1 <= number <= 9 and number <= int(str(current_year)[-1]) - 10:
                valid_years.append(2000 + number)

    return max(valid_years) if valid_years else 0


def infer_age_from_year(birth_year):
    """
    Infers age group from birth year.

    The function calculates the current age based on the birth year and categorizes it
    into one of the predefined age groups: young, medium, old, or unsure.

    Args:
        birth_year (int): The birth year.

    Returns:
        dict: Dictionary containing age group, score, and weight.
    """
    current_year = datetime.now().year
    age = current_year - birth_year
    if 10 < age < 30:
        return {"age": "young", "score": 1.0, "weight": 0.8}
    elif 30 <= age <= 50:
        return {"age": "medium", "score": 1.0, "weight": 0.8}
    elif age > 50:
        return {"age": "old", "score": 1.0, "weight": 0.8}
    else:
        return {"age": "unsure", "score": 1.0, "weight": 0.6}


def infer_age_from_domain_type(domain):
    """
    Infers age group based on domain type.

    The function uses the domain suffix to infer probable age group. Educational (.edu) and government (.gov)
    domains are specifically checked to make assumptions about the user's age group.

    Args:
        domain (str): The domain of the email.

    Returns:
        dict: Dictionary containing age group, score, and weight.
    """
    if domain.endswith('.edu'):
        return {"age": "young", "score": 1.0, "weight": 0.3}
    elif domain.endswith('.gov'):
        return {"age": "old", "score": 1.0, "weight": 0.3}
    return {"age": "unsure", "score": 1.0, "weight": 0.1}


def get_domain_age(domain, whois_data):
    """
    Gets the age of a domain from WHOIS data.

    The function calculates the domain age by comparing the current date with the domain creation date
    retrieved from the WHOIS data.

    Args:
        domain (str): The domain.
        whois_data (dict): WHOIS data dictionary.

    Returns:
        float: The age of the domain in years or None if not available.
    """
    try:
        domain_info = whois_data.get(domain, {})
        creation_date = domain_info.get('creation_date')

        if isinstance(creation_date, list):
            creation_date = creation_date[0]
        if isinstance(creation_date, datetime):
            return (datetime.now() - creation_date).total_seconds() / (365.25 * 24 * 3600)
    except Exception as e:
        print(f"Error processing domain age for {domain}: {e}")

    return None


def load_whois_data(unique_domains):
    """
    Loads WHOIS data for a list of domains, fetching data if not already cached.

    This function checks for a cache file and loads data from it if available. If not, it fetches WHOIS data
    for the domains that are not in the cache and then updates the cache file.

    Args:
        unique_domains (list): List of unique domains.

    Returns:
        dict: WHOIS data for the domains.
    """
    data_dir = 'data'
    cache_file = os.path.join(data_dir, 'whois_data.json')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    whois_data = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as file:
            whois_data = json.load(file)

    domains_to_fetch = [domain for domain in unique_domains if domain not in whois_data]

    for domain in domains_to_fetch:
        try:
            whois_info = whois.whois(domain)
            whois_data[domain] = whois_info
        except Exception as e:
            print(f"Error retrieving WHOIS info for {domain}: {e}")

    with open(cache_file, 'w') as file:
        json.dump(whois_data, file, default=str)

    return whois_data


def infer_age_from_domain_age(domain_age):
    """
    Infers age group based on the age of the domain.

    The function uses the age of the domain to infer probable age group. Different age ranges of the domain
    correspond to different age group probabilities.

    Args:
        domain_age (float): The age of the domain in years.

    Returns:
        dict: Dictionary containing age groups and their respective scores and weights.
    """
    if domain_age is None:
        return {"age": "unsure", "score": 0.5, "weight": 0.1}

    if domain_age < 7:
        return {"age": "young", "score": 1.0, "weight": 0.4}
    elif domain_age < 20:
        return {
            "young": {"age": "young", "score": 0.3, "weight": 0.2},
            "medium": {"age": "medium", "score": 0.5, "weight": 0.2},
            "old": {"age": "old", "score": 0.2, "weight": 0.2}
        }
    else:
        return {
            "young": {"age": "young", "score": 0.1, "weight": 0.1},
            "medium": {"age": "medium", "score": 0.3, "weight": 0.2},
            "old": {"age": "old", "score": 0.6, "weight": 0.3}
        }


def combine_ages(age_sources, temperature=0.5):
    """
    Combines age inferences from different sources using weighted scores.

    This function aggregates scores from different age inference sources and applies a softmax function
    to generate probabilities for each age group.

    Args:
        age_sources (list): List of age source dictionaries.
        temperature (float): Temperature parameter for softmax function.

    Returns:
        tuple: Contains max age group, its probability, combined scores, and uncertainty.
    """
    combined_score = {"young": 0.0, "medium": 0.0, "old": 0.0, "unsure": 0.0}
    total_weight = 0.0

    for source in age_sources:
        if isinstance(source, dict):
            combined_score[source["age"]] += source["score"] * source["weight"]
            total_weight += source["weight"]

    if total_weight > 0:
        for age in combined_score:
            combined_score[age] /= total_weight

    scores_array = np.array(
        [combined_score["young"], combined_score["medium"], combined_score["old"], combined_score["unsure"]])
    probabilities = softmax(scores_array, temperature)

    combined_score["young"], combined_score["medium"], combined_score["old"], combined_score["unsure"] = probabilities

    max_age_group = max(combined_score, key=lambda k: combined_score[k])
    uncertainty = 1 - combined_score[max_age_group]

    return max_age_group, combined_score[max_age_group], combined_score, uncertainty


def get_bert_embeddings(usernames):
    """
    Generates BERT embeddings for a list of usernames.

    This function uses a pre-trained BERT model to convert usernames into vector embeddings.

    Args:
        usernames (list): List of usernames.

    Returns:
        np.ndarray: Numpy array of embeddings.
    """
    embeddings = []
    for username in usernames:
        inputs = tokenizer(username, return_tensors='pt')
        outputs = model(**inputs)
        embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.vstack(embeddings)


def infer_age_from_knn(knn, new_username_vector):
    """
    Infers age using K-Nearest Neighbors based on BERT embeddings.

    This function uses a trained KNN model to predict the birth year from the BERT embeddings of the username,
    and then infers the age group based on the predicted year.

    Args:
        knn (KNeighborsClassifier): Trained KNN model.
        new_username_vector (np.ndarray): BERT embedding of the new username.

    Returns:
        tuple: Contains predicted year and age inference dictionary.
    """
    neighbors = knn.kneighbors(new_username_vector, return_distance=False)[0]
    neighbor_ages = [knn._y[i] for i in neighbors]

    if all(age == 0 for age in neighbor_ages):
        return 0, {"age": "unsure", "score": 0.0, "weight": 0.0}

    predicted_year = knn.predict(new_username_vector)[0]
    if predicted_year == 0:
        return 0, {"age": "unsure", "score": 0.0, "weight": 0.0}

    return predicted_year, infer_age_from_year(predicted_year)
