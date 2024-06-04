from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from age_inference import (
    extract_username, extract_year_from_username,
    get_bert_embeddings, infer_age_from_knn,
    infer_age_from_year, infer_age_from_domain_type,
    get_domain_age, load_whois_data, combine_ages
)
from utils import clean_emails

def process_emails(emails):
    original_emails, cleaned_emails = clean_emails(emails)
    usernames = [extract_username(email) for email in cleaned_emails]
    birth_years = [extract_year_from_username(username) for username in usernames]
    X = get_bert_embeddings(usernames)
    unique_domains = list(set(email.split('@')[1] for email in cleaned_emails))
    whois_data = load_whois_data(unique_domains)

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    knn.fit(X, birth_years)

    results = []
    for i, (original_email, cleaned_email, username, birth_year) in tqdm(enumerate(
            zip(original_emails, cleaned_emails, usernames, birth_years))):
        age_sources = []

        if birth_year != 0:
            age_sources.append(infer_age_from_year(birth_year))

        username_vector = X[i].reshape(1, -1)
        knn_inferred_year, knn_age = infer_age_from_knn(knn, username_vector)
        if knn_inferred_year != 0:
            age_sources.append(knn_age)

        domain = cleaned_email.split('@')[1]
        age_sources.append(infer_age_from_domain_type(domain))
        domain_age = get_domain_age(domain, whois_data)
        age_sources.append(infer_age_from_domain_age(domain_age))

        combined_age, combined_score, all_scores, uncertainty = combine_ages(age_sources, temperature=0.5)

        result = {
            "age": combined_age,
            "score": combined_score
        }
        results.append(result)

    return results
