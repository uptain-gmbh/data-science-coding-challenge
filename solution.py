import json
from ml_model import MLModel

class Solution:
    def __init__(self):
        self.model = MLModel()

    ## Form the json using the output
    def build_json(self, age, score):
        json_data = {'age': age, 'score': score}
        json_data = json.dumps(json_data)
        return json_data


    ## Solution executed the model to 
    ## predict the age of the input email address
    ## input requests: denotes the number of times the model is executed
    def solution(self, requests: int):
        total_requests = requests
        while requests > 0:
            ## Predict the age of the user
            input_string = input("\nPlease enter an email id to predict age: ")
            ipemailid = input_string.strip()
            ## Call the trained model to predict for the input email id
            (age_class, score) = self.model.predict_age(ipemailid)
            res = self.build_json(str(age_class), float(score))
            print(res)
            requests -= 1
        print('\n \n Handled '+str(total_requests)+' requests. Exiting now')
        return


if __name__ == "__main__":
    sol = Solution()
    ## User enters the number of times they wish to test the model.
    ## This is to avoid the programme from running infinitely.
    input_string = input("\nPlease enter the number of requests you want to process: ")
    requests = int(input_string.strip())
    sol.solution(requests)
    print("Solution executed successfully!")