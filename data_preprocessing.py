import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from read_data import read_data



class DataPreprocessing:
    def __init__(self):
        self.reader = read_data()
        self.category_mappings = {}
        self.dom_prob = {}
        self.domain_data_csv = pd.read_csv('domain_data.csv')
        self.label_encoder = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.data = self.reader.data

    ## Scale the data between 0 and 1 for numerical values.
    ## As numerical values are not accepted by random forest classifier
    ## categorical values are replaced with numerical values
    def scale_data(self, data: pd.DataFrame) -> pd.DataFrame:     
        ## Because the input email id also needs to be scaled
        ## store the categorical mappings in a dict for later use.   
        for col in list(data.select_dtypes(['object']).columns):
            ## When this is not the first time the modified
            ## The mappings for a given column are already present
            ## In case of using the model for prediction
            if col in self.category_mappings.keys():
                val = data[col].iloc[0]
                if val in self.category_mappings[col].keys():
                    data.loc[:,col] = self.category_mappings[col][val]
                else:
                    data.loc[:,col] = len(self.category_mappings[col]) 
            else:
                ## When the data is being scaled for the training the model
                self.label_encoder.fit(data[col]) 
                self.category_mappings[col] = dict(zip(self.label_encoder.classes_, 
                                                       self.label_encoder.transform(self.label_encoder.classes_)))
                data.loc[:, col] = self.label_encoder.fit_transform(data[col]) 

        ## Sclare the numerical values between 0 and 1
        for col in list(data.select_dtypes(['int']).columns):
            data = data.astype({col: 'float'})
            data.loc[:, col] = self.scaler.fit_transform(data[[col]]) 
        return data
    

    ## Label the data based on the domain data and the username prediction
    def label_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data['label'] = data['probable_age_cat_digits']
        data['label'] = np.where(data['probable_age_cat_digits'] == 'unsure', 
                                 data['probable_cat_domain'], data['probable_age_cat_digits'])

        data['label_score'] = 0
        unique_domains = self.domain_data['domain'].unique()
        ## For each domain, calculate the probabilty that the user belongs to a certain age group
        ## This probability is the confidence of labelling for each of the email id.
        for domain in unique_domains:
            if domain in self.dom_prob.keys():
                data['label_score'] = np.where((data['domain'] == domain) & (data['label'] == 'young'), 
                                        (self.dom_prob[domain]['young'])/100, data['label_score'])
                data['label_score'] = np.where((data['domain'] == domain) & (data['label'] == 'medium'),
                                        (self.dom_prob[domain]['medium'])/100, data['label_score'])
                data['label_score'] = np.where((data['domain'] == domain) & (data['label'] == 'old'),
                                        (self.dom_prob[domain]['old'])/100, data['label_score'])
        ## When digits are considered for the prediction of tha age category,
        ## If there is a mismatch between the maximum age category of the domain and that of digits category,
        ## The category predicted using the digits is considered. Hence the probability is set to that of digits category being true.
        ## The probability of it being true is 0.33333. Rounding it off to 0.34
        data.loc[(data['probable_age_cat_digits'] == data['label']) & (data['probable_cat_domain'] != data['label']), 
                 'label_score'] = 0.33
        return data


    def filter_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[['id_len', 'digits_count', 'probable_age_digits', 
                     'domain', 'probable_age_cat_digits',
                     'probable_cat_domain', 'label', 'label_score']]
        return data


    def prep(self, data: pd.DataFrame) -> pd.DataFrame:
        ## Extract domain names and the usernames
        data['username'] = [email.split("@")[0] for email in data["email"]]
        data['domain'] = [email.split("@")[1] for email in data["email"]]
        data['id_len'] = [len(username) for username in data["username"]]
        data['digits'] = [re.sub("[^0-9]", "", username) for username in data["username"]]
        data['digits_count'] = [len(digits) for digits in data["digits"]]
        data['digits'] = pd.to_numeric(data['digits'], errors='coerce')
        data['top_domain'] = [domain.split(".")[-1] for domain in data["domain"]]
        ## If digits count is 2 or 4: If 4, should be between 1910 and 2006 for age
        ## between 1985 and 2024 for year of creation. probability is 1/3
        ## If 2, considering between between 10 and 99 or 00 and 06 for age
        ## Here, the assumption is being made that a person who is born before 1910 does not have an email id
        data['probable_age_digits'] = np.where((data['digits_count'] == 4) & (data['digits'] >= 1910) & (data['digits'] <= 2006), 
                                        2024 - data['digits'], 0)
        data['probable_age_digits'] = np.where((data['digits_count'] == 2) & ((data['digits'] >= 10) & (data['digits'] <= 99)),  
                                        2024 - (1900 + data['digits']), data['probable_age_digits'])
        data['probable_age_digits'] = np.where(((data['digits_count'] == 2) | (data['digits_count'] == 1))  & ((data['digits'] >= 0) & (data['digits'] <= 6)),
                                        2024 - (2000 + data['digits']), data['probable_age_digits'])
        data['probable_age_cat_digits'] = 'unsure'
        data['probable_age_cat_digits'] = np.where((data['probable_age_digits'] >= 18) & (data['probable_age_digits'] <= 30), 
                                            'young', data['probable_age_cat_digits'])
        data['probable_age_cat_digits'] = np.where((data['probable_age_digits'] >= 31) & (data['probable_age_digits'] <= 50), 
                                            'medium', data['probable_age_cat_digits'])
        data['probable_age_cat_digits'] = np.where((data['probable_age_digits'] >= 51), 'old', data['probable_age_cat_digits'])

        data = self.add_domain_data(data)
        ## The data is labelled even in the case of prediction 
        ## for the input email id. This is to obtain the label score.
        ## Even though the label is calculated, 
        ## the predicted label by the model need not be same
        data = self.label_data(data)
        return data
    

    def get_domain_data(self) -> pd.DataFrame:
        ## Get the domain category values
        domain_data = self.domain_data_csv
        domain_data = domain_data[['domain', 'total_users_in_mil', 
                                'young', 'medium', 'old', 
                                'probable_cat_domain']]
        unique_domains = domain_data['domain'].unique()
        for i in range(len(unique_domains)):
            if domain_data[domain_data['domain'] == unique_domains[i]].iloc[0]['probable_cat_domain'] != 'unsure':
                self.dom_prob[unique_domains[i]] = {}
                self.dom_prob[unique_domains[i]]['young'] = domain_data[domain_data['domain'] == unique_domains[i]].iloc[0]['young']
                self.dom_prob[unique_domains[i]]['medium'] = domain_data[domain_data['domain'] == unique_domains[i]].iloc[0]['medium']
                self.dom_prob[unique_domains[i]]['old'] = domain_data[domain_data['domain'] == unique_domains[i]].iloc[0]['old']
        return domain_data


    def add_domain_data(self, data: pd.DataFrame ) -> pd.DataFrame:
        self.domain_data = self.get_domain_data()
        data = pd.merge(self.domain_data, data, on=["domain"], how='right')
        return data