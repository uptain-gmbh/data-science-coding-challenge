# import sklearn
import json
from joblib import load
import numpy as np
from utils import tokenize_and_pad, is_valid_email

with open("mapping.json", 'r') as f:
    mapping = json.load(f)

with open('scores.json', 'r') as f:
    scores = json.load(f)


forest = load('forest_model.joblib')

email = input("enter Email address: ")
assert is_valid_email(email), "please input a valid email address."

if len(email) > 32:
    print("warning: Email length is greater than 32 characters. This might affect the output result.")

processed = tokenize_and_pad(email, mapping)

predictions = forest.predict_proba([processed])
label = str(int(np.argmax(predictions)))
score = np.max(predictions)
result = {"age": scores[label], "score": score}
print(f'{result}')