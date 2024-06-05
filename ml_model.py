import pandas as pd
import numpy as np
from data_preprocessing import DataPreprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier



class MLModel:
    def __init__(self):
        print("Thank you for your patience.. model will be ready in a moment.")
        self.data_processor = DataPreprocessing()
        self.model = None
        self.data = self.data_processor.data
        self.data = self.data_processor.prep(self.data)
        self.build_model(self.data)


    ## Because this data is skewed, weights are assigned 
    ## inversely proportional to the number of samples in each class.
    ## This will ensure that the model is not biased towards the majority class.
    def get_weights(self, data: pd.DataFrame):
        weights = {}
        for label in data['label'].unique():
            weights[label] = 1/data[data['label'] == label].shape[0]
        return weights


    ## ML model is build using RandomForestClassifier
    ## For the model, scaled data with categorical values replaced, is used.
    def build_model(self, data: pd.DataFrame):
        ## Features used for the model
        data_features = data[['id_len', 'digits_count', 'probable_age_digits', 
                              'domain', 'probable_age_cat_digits', 
                              'probable_cat_domain', 'label_score']]
        data_features = self.data_processor.scale_data(data_features)
        
        ## Target variable used for the model
        data_labels = data[['label']]
        weights = self.get_weights(data_labels)
        self.model = RandomForestClassifier(random_state=0, class_weight=weights)
        #self.model = DecisionTreeClassifier(random_state=0, class_weight=weights)
        self.model.fit(data_features, data_labels.values.ravel())


    def predict_age(self, emailid: str):
        ## Check entered email is valid or not
        if not self.data_processor.reader.check_email(emailid):
            print( 'Email id entered is invalid.')
            return 'invalid email id', -1
        
        ## Preprocess the ipemailid
        processedIpEmailid = self.data_processor.prep(pd.DataFrame([emailid], columns=['email']))
        input_params = processedIpEmailid[['id_len', 'digits_count', 'probable_age_digits', 
                              'domain', 'probable_age_cat_digits', 
                              'probable_cat_domain' , 'label_score']]
        
        ## The categories and the scale should match the ones with the model is trained
        input_params = self.data_processor.scale_data(input_params)
        category = self.model.predict(input_params)[0]
        score = np.max(self.model.predict_proba(input_params), axis=1)[0]
        return category, score
    

    