# Email Data Processor and Classifier

This repository features a Python script designed to predict the age of a person based on their email address. The script processes a file named `emails.txt`, which contains an unsorted list of email addresses. Each email has potential attributes that can be linked to the user's age. The script extracts these attributes and uses them to build a model that predicts the age class based on the email address.

## Features

- **Data Extraction and Processing**: Parses emails from a file and organizes extracted details into a structured DataFrame, saved to a CSV for further use.
- **Model Training**: Trains and serializes several machine learning models to classify data into predefined age classes.
- **Model Testing**: Allows testing trained models with sample email inputs for practical validation of the classifier's effectiveness.
- **Alternative Non-ML Approach**: Includes methods to handle and evaluate email data without using machine learning, focusing on basic data handling and processing techniques.

## How to Run

1. Please ensure `requirements.txt` is run to install all necessary packages.
2. Modify the sample email ID in the `test_email_with_model` function to test different email IDs.
3. Execute the script to process the data, train the models, and test the outputs.

This solution offers both a machine learning approach and a simpler alternative that does not rely on ML techniques, providing flexibility depending on the user's needs or computational resources.


