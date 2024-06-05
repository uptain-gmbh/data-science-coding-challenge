import pandas as pd
from email_validator import validate_email, EmailNotValidError


class read_data:
    def __init__(self):
        self.data = pd.DataFrame()
        self.invalid_emails = pd.DataFrame()
        self.read_data()

    ## Check if the email id passed as input is valid or not
    def check_email(self, email: str) -> bool:
        try:
            # validate the email and get normalized form
            result = validate_email(email) 
            email = result.email
            return True
        except EmailNotValidError as e:
            # If email is not valid, return false
            return False
        

    ## Read the data from the input csv file.
    ## Validate each of the email id read 
    ## and separate the invalid ones
    def read_data(self) -> pd.DataFrame:
        data = pd.DataFrame(columns=(['email']))
        invalid_emails = pd.DataFrame(columns=(['email']))
        ## First clean data
        ## Separate any emails that are not valid
        ## Separate any emails that are not in the right format
        with open('emails.txt') as file:
            for line in file:
                line = line.rstrip()
                if (self.check_email(line)):
                    ## Add line to the dataframe
                    new_email = {'email': line}
                    data = pd.concat([data, pd.DataFrame([new_email])], ignore_index=True)
                else:
                    ## Add the email to the invalid email list
                    new_email = {'email': line}
                    invalid_emails = pd.concat([invalid_emails, pd.DataFrame([new_email])], ignore_index=True)
        self.data = data.reset_index(drop=True)
        self.invalid_emails = invalid_emails.reset_index(drop=True)
        return

    ## Returns the list of valid email addresses 
    ## that was read from the input file
    @property
    def get_data(self):
        return self.data