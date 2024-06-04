# import json
# import os
# import re
# from datetime import datetime
#
# import numpy as np
# import whois
# from sklearn.neighbors import KNeighborsClassifier
# from tqdm import tqdm
# from transformers import BertTokenizer, BertModel
#
# # Initialize BERT tokenizer and model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')
#
#
# def softmax(scores, temperature=1.0):
#     """Compute softmax values for each set of scores in x with temperature."""
#     scores = np.array(scores) / temperature
#     e_x = np.exp(scores - np.max(scores))
#     return e_x / e_x.sum()
#
#
# def extract_username(email):
#     return email.split('@')[0]
#
#
# def is_valid_email(email):
#     pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
#     return re.match(pattern, email) is not None
#
#
# def fix_email(email):
#     if email.count('@') != 1:
#         return email
#
#     local_part, domain_part = email.split('@')
#     domain_part = domain_part.replace(',', '.')
#     if domain_part.endswith('..'):
#         domain_part = domain_part.rstrip('.') + '.com'
#     domain_part = re.sub(r'\.\.+', '.', domain_part)
#
#     return f"{local_part}@{domain_part}"
#
#
# def clean_emails(emails):
#     cleaned_emails = []
#     original_emails = []
#     for email in emails:
#         email = email.strip().lower()
#         fixed_email = fix_email(email)
#         if is_valid_email(fixed_email):
#             cleaned_emails.append(fixed_email)
#             original_emails.append(email)
#     return original_emails, cleaned_emails
#
#
# def extract_year_from_username(username):
#     numbers = re.findall(r'\d+', username)
#     current_year = datetime.now().year
#     valid_years = []
#
#     for number in numbers:
#         num_len = len(number)
#         number = int(number)
#         if num_len == 4 and 1900 <= number <= current_year:
#             valid_years.append(number)
#         elif num_len == 2:
#             if 60 <= number <= 99:
#                 valid_years.append(1900 + number)
#             elif 0 <= number <= int(str(current_year)[-2:]) - 10:
#                 valid_years.append(2000 + number)
#         elif num_len == 1:
#             if number == 0:
#                 valid_years.append(2000)
#             elif 1 <= number <= 9 and number <= int(str(current_year)[-1]) - 10:
#                 valid_years.append(2000 + number)
#
#     return max(valid_years) if valid_years else 0
#
#
# def infer_age_from_year(birth_year):
#     current_year = datetime.now().year
#     age = current_year - birth_year
#     if age > 10 and age < 30:
#         return {"age": "young", "score": 1.0, "weight": 0.8}
#     elif 30 <= age <= 50:
#         return {"age": "medium", "score": 1.0, "weight": 0.8}
#     elif age > 50:
#         return {"age": "old", "score": 1.0, "weight": 0.8}
#     else:
#         return {"age": "unsure", "score": 1.0, "weight": 0.6}
#
#
# def infer_age_from_domain_type(domain):
#     if domain.endswith('.edu'):
#         return {"age": "young", "score": 1.0, "weight": 0.3}
#     elif domain.endswith('.gov'):
#         return {"age": "old", "score": 1.0, "weight": 0.3}
#     return {"age": "unsure", "score": 1.0, "weight": 0.1}
#
#
# def get_domain_age(domain, whois_data):
#     try:
#         domain_info = whois_data.get(domain, {})
#         creation_date = domain_info.get('creation_date')
#
#         if isinstance(creation_date, list):
#             creation_date = creation_date[0]
#         if isinstance(creation_date, datetime):
#             return (datetime.now() - creation_date).total_seconds() / (365.25 * 24 * 3600)
#     except Exception as e:
#         print(f"Error processing domain age for {domain}: {e}")
#
#     return None
#
#
# def load_whois_data(unique_domains):
#     data_dir = 'data'
#     cache_file = os.path.join(data_dir, 'whois_data.json')
#
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#
#     whois_data = {}
#     if os.path.exists(cache_file):
#         with open(cache_file, 'r') as file:
#             whois_data = json.load(file)
#
#     domains_to_fetch = [domain for domain in unique_domains if domain not in whois_data]
#
#     for domain in domains_to_fetch:
#         try:
#             whois_info = whois.whois(domain)
#             whois_data[domain] = whois_info
#         except Exception as e:
#             print(f"Error retrieving WHOIS info for {domain}: {e}")
#
#     with open(cache_file, 'w') as file:
#         json.dump(whois_data, file, default=str)
#
#     return whois_data
#
#
# def infer_age_from_domain_age(domain_age):
#     if domain_age is None:
#         return {"age": "unsure", "score": 0.5, "weight": 0.1}
#
#     if domain_age < 7:
#         return {"age": "young", "score": 1.0, "weight": 0.4}
#     elif domain_age < 20:
#         return {
#             "young": {"age": "young", "score": 0.3, "weight": 0.2},
#             "medium": {"age": "medium", "score": 0.5, "weight": 0.2},
#             "old": {"age": "old", "score": 0.2, "weight": 0.2}
#         }
#     else:
#         return {
#             "young": {"age": "young", "score": 0.1, "weight": 0.1},
#             "medium": {"age": "medium", "score": 0.3, "weight": 0.2},
#             "old": {"age": "old", "score": 0.6, "weight": 0.3}
#         }
#
#
# def combine_ages(age_sources, temperature=0.5):
#     combined_score = {"young": 0.0, "medium": 0.0, "old": 0.0, "unsure": 0.0}
#     total_weight = 0.0
#
#     for source in age_sources:
#         if isinstance(source, dict):
#             combined_score[source["age"]] += source["score"] * source["weight"]
#             total_weight += source["weight"]
#
#     if total_weight > 0:
#         for age in combined_score:
#             combined_score[age] /= total_weight
#
#     scores_array = np.array(
#         [combined_score["young"], combined_score["medium"], combined_score["old"], combined_score["unsure"]])
#     probabilities = softmax(scores_array, temperature)
#
#     combined_score["young"], combined_score["medium"], combined_score["old"], combined_score["unsure"] = probabilities
#
#     max_age_group = max(combined_score, key=lambda k: combined_score[k])
#     uncertainty = 1 - combined_score[max_age_group]
#
#     return max_age_group, combined_score[max_age_group], combined_score, uncertainty
#
#
# def get_bert_embeddings(usernames):
#     embeddings = []
#     for username in usernames:
#         inputs = tokenizer(username, return_tensors='pt')
#         outputs = model(**inputs)
#         embeddings.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
#     return np.vstack(embeddings)
#
#
# def infer_age_from_knn(knn, new_username_vector):
#     neighbors = knn.kneighbors(new_username_vector, return_distance=False)[0]
#     neighbor_ages = [knn._y[i] for i in neighbors]
#
#     if all(age == 0 for age in neighbor_ages):
#         return 0, {"age": "unsure", "score": 0.0, "weight": 0.0}
#
#     predicted_year = knn.predict(new_username_vector)[0]
#     if predicted_year == 0:
#         return 0, {"age": "unsure", "score": 0.0, "weight": 0.0}
#
#     return predicted_year, infer_age_from_year(predicted_year)
#
#
# def process_emails(emails):
#     original_emails, cleaned_emails = clean_emails(emails)
#     usernames = [extract_username(email) for email in cleaned_emails]
#     birth_years = [extract_year_from_username(username) for username in usernames]
#     X = get_bert_embeddings(usernames)
#     unique_domains = list(set(email.split('@')[1] for email in cleaned_emails))
#     whois_data = load_whois_data(unique_domains)
#
#     knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
#     knn.fit(X, birth_years)
#
#     results = []
#     for i, (original_email, cleaned_email, username, birth_year) in tqdm(enumerate(
#             zip(original_emails, cleaned_emails, usernames, birth_years))):
#         age_sources = []
#
#         if birth_year != 0:
#             age_sources.append(infer_age_from_year(birth_year))
#
#         username_vector = X[i].reshape(1, -1)
#         knn_inferred_year, knn_age = infer_age_from_knn(knn, username_vector)
#         if knn_inferred_year != 0:
#             age_sources.append(knn_age)
#
#         domain = cleaned_email.split('@')[1]
#         age_sources.append(infer_age_from_domain_type(domain))
#         domain_age = get_domain_age(domain, whois_data)
#         age_sources.append(infer_age_from_domain_age(domain_age))
#
#         combined_age, combined_score, all_scores, uncertainty = combine_ages(age_sources, temperature=0.5)
#
#         result = {
#             "age": combined_age,
#             "score": combined_score
#         }
#         results.append(result)
#
#     # Save results to a JSON file
#     with open('data/output/results.json', 'w') as f:
#         for result in results:
#             f.write(json.dumps(result) + '\n')
#
#
# # Main function to read emails from file and process them
# if __name__ == "__main__":
#     with open('data/raw/emails.txt', 'r') as file:
#         emails = file.read().splitlines()
#
#     process_emails(emails)
