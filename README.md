# Uptain Data Science Coding Challenge
 I have considered this task as a classification problem, because we're basically assigning
the email addresses to its right class [young, medium, old, unsure].

For solving this problem, I have done a feature engineering step to extract some features of the data
the extracted features are as follows:
- **Year**: As a heuristic, I have assumed the number inside the email addresses, might indicate the year 
  of creation. For this purpose I have first looked for 4-digit numbers like 19xx and 20xx, if it is not
  found, I then looked for 2-digit numbers and converted them to 19xx or 20xx. If there is no number, this feature
  would be None (0 in this case).
- **Domain**: Domains can also be assumed as a feature, because some domain are old, some are newer
  so they may containt valuable information. I have used a vectorizer to convert these domain to tokens.

**Classifier**: The **Random Forest** is chosen for this task, because the nature of the dataset; to prevent
overfitting, outlier handling and also because of its interpretability.

I didn't use any **Deep learning model** for solving this challenge, because we have limited amount of data.
Also the features we have use are not complicated, so fine-tuning a pretrained model is also not a good choice.

To train the model, I have labeled apart of the data  (600 samples), to have enough data for training the model.
The I have used this model to label all the data. This is the evaluation of the trained model:

              precision    recall  f1-score   support

       young       1.00      0.95      0.97        20
      medium       0.96      1.00      0.98        24
         old       1.00      1.00      1.00        19
      unsure       1.00      1.00      1.00        57

    accuracy                          0.99       120
    macro avg      0.99     0.99      0.99       120
    weighted avg   0.99     0.99      0.99       120

Once you run the code, everything would be done from scratch, and it waits for you to insert 
new email addresses.


### How to run without Docker
`pip3 install -r requirements.txt`

`python3 main.py`

### How to run with Docker
`docker build -t uptain .`

`docker container run -it --name uptain uptain`