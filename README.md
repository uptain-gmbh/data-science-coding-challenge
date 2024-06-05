## Input validation
The given email addresses are screened for valid email address format. Only those which are valid are considered for  further steps.
Even when the input is entered by the user to test the model, the input is validated for the proper email address format.


## Attribute selection
Email addresses comprise of two main components: username and domain name.
username was used to derive as much insight as possible regarding the age. 

Using username, I have derived the following attributes: 
    -- length of the username
    -- digits
    -- length of digits
    -- year of birth 
    -- probable age 
    -- probable age group 

    Using the length of digits, the possible age of the email address user is predicted (year of birth, probable age, probable age category). 
    (Only those cases with the length of digits 2 or 4 are considered to predict the year. This is due to the shortage of time and ease of coding for this project.)
    For any derived year, it can mean the following:
        -- year of creation of email ID
        -- year of birth of the user
        -- age of the user when email ID is created. (in case of 2 digits)
    Hence, using the derived year, probability that the predicted age / age category is true is only 0.33 (assuming uniform distribution among the above three cases.)
    ** Please note that an assumption is made that the users of the email ID are all above 18 years of age. Hence, people who are born in the last 18 years from now do not have an email address.
    


domain name was used to derive the age distribution of the users with the email ID linked to the given domain.

Some of the sources used to gain the statistics of the users in each of the category are listed below: 
https://www.demandsage.com/gmail-statistics/
https://www.similarweb.com/website/hotmail.com/#demographics
https://wifitalents.com/statistic/apple-icloud/#:~:text=%22Over%2070%25%20of%20iCloud%27s%20user,group%20of%2018%2D49.%22
https://www.sellcell.com/blog/most-popular-email-provider-by-number-of-users/
https://market.us/statistics/internet/email-users/

In the above sources, the distribution of age is made into 5 categories. An assumption is made that in each of these categories, the number of users are same for every age. Based on this assumption, the age distribution for the required 3 categories in this problem is derived. 
Some of the above sources also provide country-specific data. This data is assumed to be true for all the countries and reflects the world-wide distribution of the users.
Using the above sources, information was derived for only few domains. For other domains, the data was extrapolated based on the year of foundation.

Using the domain name, following attributes were derived:
    -- top domain
    -- year of foundation of domain
    -- percentage of users in age group young
    -- percentage of users in age group medium
    -- percentage of users in age group old
    -- maximum probable age category


## Machine learning model
The given data is labeled and converted to a supervised learning problem.
To label the data, few key attributes are used.
    -- probable age category (derived from user name) (first attribute)
    -- maximum probable age category (derived from domain name) (second attribute)

The probability of the first attribute to be true is 0.33. 
The probability of the second attribute to be true is (percentage of maximum probable age category) / 100.
If the first attribute is present, and if the first and the second attributes differ, the label assigned is the same as that of the first attribute with the label_score = 0.33
Otherwise, the label assigned is the that of the second attribute with the label_score = (probability of the second category in the domain)
This also helped derive another attribute: label_score.
Using the above attributes, RandomForestClassifier is used to classify the instances. 


## Input of the model
The model takes in the number(integer) of email ids (n) the tester/user of the solution wants to predict.
Followed by this, the model requests "n" email IDs (string) to be entered and classifies each of these email ID to the predefined age categories.


## How to run the model
Run the solution.py file
`python3 solution.py`


## Libraries used
Pandas, Numpy, scikit-learn, json and email_validator