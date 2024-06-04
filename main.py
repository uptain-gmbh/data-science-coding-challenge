from preprocessing import Preprocess
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
import json
import numpy as np
import re

def is_valid_email(email: str) -> bool:
  """
  This function checks if the input string is a valid email address.
  """
  regex = r"^([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)$"
  return bool(re.match(regex, email))


def predictor(model: RandomForestClassifier, preprocessor: Preprocess, vectorizer: CountVectorizer):
    """
    Predicts user inputs
    :param model: The trained model
    :param preprocessor: The preprocessor
    :param vectorizer: The domain vectorizer
    :returns: label and score
    """
    while True:
        user_input = input("Enter the desired email address: ")
        if not isinstance(user_input, str) or not is_valid_email(user_input):
            print("Not a valid email address. Try again...")
            continue
        user_input = user_input.strip()
        user_input = user_input.lower()
        year_feature = preprocessor.extract_year(user_input)
        domain = preprocessor.extract_domain(user_input)
        domain_features = vectorizer.transform([domain])
        year_feature = np.expand_dims(np.array([year_feature]), axis=0)
        features = hstack([domain_features, year_feature])
        print("age:", ['young', 'medium', 'old', 'unsure'][model.predict(features)[0]],
              " Score:", model.predict_proba(features).max(axis=1)[0])



# Format output as JSON
if __name__ == "__main__":
    with open('emails.txt', 'r') as file:
        emails = file.readlines()

    preprocessing_client = Preprocess()

    emails = [email.strip() for email in emails]
    data = [preprocessing_client.run(email) for email in emails]

    df = pd.DataFrame(data)
    df = preprocessing_client.label(df)

    # Making some labeled data
    labeled_data = df.sample(600, random_state=42)
    labeled_data['label'] = labeled_data['age_class'].apply(
lambda x: {'young': 0, 'medium': 1, 'old': 2, 'unsure': 3}[x])

    vectorizer = CountVectorizer()
    domain_feature = vectorizer.fit_transform(labeled_data['domain'])

    year_age_features = labeled_data[['year']].values
    x = hstack([domain_feature, year_age_features])
    y = labeled_data['label']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['young', 'medium', 'old', 'unsure']))

    # Predict for the entire dataset
    x_full_domain = vectorizer.transform(df['domain'])
    x_full_year = df[['year']].values
    x_full = hstack([x_full_domain, x_full_year])
    df['prediction'] = model.predict(x_full)
    df['prediction_proba'] = model.predict_proba(x_full).max(axis=1)

    # Map predictions to age classes
    df['predicted_age_class'] = df['prediction'].apply(lambda x: ['young', 'medium', 'old', 'unsure'][x])
    df['score'] = df['prediction_proba']

    output = df.apply(lambda row: {"age": row['predicted_age_class'], "score": row['score']}, axis=1).tolist()
    with open('output.json', 'w') as file:
        for item in output:
            file.write(json.dumps(item) + "\n")

    # Serving user inputs
    predictor(model, preprocessing_client, vectorizer)