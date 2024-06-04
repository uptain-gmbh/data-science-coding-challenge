from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from age_inference import (
    extract_username, extract_year_from_username,
    get_bert_embeddings, infer_age_from_knn,
    infer_age_from_year, infer_age_from_domain_type,
    get_domain_age, load_whois_data, combine_ages,
    infer_age_from_domain_age
)
from utils import clean_emails


def process_emails(emails):
    """
    Processes a list of email addresses to infer the age group of each user.

    This function cleans the email addresses, extracts usernames, predicts birth years,
    generates BERT embeddings, loads WHOIS data, and combines multiple age inference
    methods to predict the most likely age group for each email address.

    Args:
        emails (list): List of email addresses to be processed.

    Returns:
        list: List of dictionaries containing the inferred age group and its score for each email.
    """
    # Clean the emails and extract usernames
    original_emails, cleaned_emails = clean_emails(emails)
    usernames = [extract_username(email) for email in cleaned_emails]

    # Extract potential birth years from usernames
    birth_years = [extract_year_from_username(username) for username in usernames]

    # Generate BERT embeddings for usernames
    X = get_bert_embeddings(usernames)

    # Get unique domains and load WHOIS data
    unique_domains = list(set(email.split('@')[1] for email in cleaned_emails))
    whois_data = load_whois_data(unique_domains)

    # Train K-Nearest Neighbors model on the embeddings and birth years
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X, birth_years)

    results = []

    # Process each email to infer the age group
    for i, (original_email, cleaned_email, username, birth_year) in tqdm(enumerate(
            zip(original_emails, cleaned_emails, usernames, birth_years))):
        age_sources = []

        # Infer age from birth year
        if birth_year != 0:
            age_sources.append(infer_age_from_year(birth_year))

        # Infer age using KNN based on BERT embeddings
        username_vector = X[i].reshape(1, -1)
        knn_inferred_year, knn_age = infer_age_from_knn(knn, username_vector)
        if knn_inferred_year != 0:
            age_sources.append(knn_age)

        # Infer age from the domain type
        domain = cleaned_email.split('@')[1]
        age_sources.append(infer_age_from_domain_type(domain))

        # Infer age from the age of the domain
        domain_age = get_domain_age(domain, whois_data)
        age_sources.append(infer_age_from_domain_age(domain_age))

        # Combine the inferences from different sources
        combined_age, combined_score, all_scores, uncertainty = combine_ages(age_sources, temperature=0.5)

        result = {
            "age": combined_age,
            "score": combined_score
        }
        results.append(result)

    return results
