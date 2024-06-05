# Uptain Data Science Coding Challenge


# Description of solution

My first step was to load the data to understand its structure. Upon seeing that it consisted of typical email addresses, I contemplated the potential features to use for predicting age. To explore this, I consulted ChatGPT for suggestions on features that could help determine a person's age from their email address. ChatGPT suggested several features, such as the length of the username, the presence of special characters (underscores or dots), the inclusion of numbers, and the email domain/provider.

These suggestions were intriguing, particularly in terms of understanding what insights into age these attributes could provide. However, I had reservations about some assumptions made by ChatGPT. For instance, it assumed that the presence of numbers in the username likely indicated a birth year. It also suggested that older email domains might correspond to older users. While I partially agreed with this, I noted that younger users might also use older domains, which could introduce bias if these assumptions were overly relied upon. Additionally, ChatGPT's suggestion to use these features in a supervised ML model seemed redundant; if the features alone could determine age, a simple rule-based system might suffice.

Through further online research, I found that the structure of email addresses has evolved over the years. Early email addresses tended to be short, often comprising just a first name and a number, whereas more recent ones typically have a more professional look and are longer. While this observation was considered, it was not used in the initial model training but was kept in mind for later analysis.

Next, I examined the data more closely and performed data cleaning since many emails were improperly formatted. Notably, the dataset contained only 1073 email addresses, which posed a challenge for building a robust ML model given the limited features. With cleaned data, I observed that approximately 45% of the emails were from Gmail domains, potentially biasing the model. Given that this was an unsupervised learning problem without labeled data, I opted for a KMeans model to classify the emails into four age groups. To handle feature importance, I used IsolationForest.

A challenge arose with the high importance assigned to domains, affecting the results. To address this, I had to tune the weights so that all features were considered with relative importance. The final features selected were: username length, and boolean indicators for the presence of dots, underscores, numbers, and the email domain. This resulted in five features.

To prepare the data for the model, I normalized the username length using MinMaxScaler and one-hot-encoded the categorical domain features, resulting in 18 features for the model. I created four clusters corresponding to the four age categories and examined how the model grouped the email addresses. To mitigate the bias from the overrepresented Gmail domain, I adjusted the weights accordingly.

Analyzing the clusters created by the model revealed trends in email address patterns over time, matching the previous internet research. Shorter addresses were common in the 90s and early 2000s, often using domains like AOL, Hotmail, and Yahoo. More recent trends include longer addresses with underscores and the use of dots between first and last names, along with domains like iCloud, ProtonMail, and Tutanota. The 'unsure' cluster consisted of emails that combined various features, making them harder to classify into a specific group. With this in mind, the names for each cluster are determined, making this the first and only assumption used for the model.

Finally, I tested a few personal email addresses from friends and family, obtaining some correct predictions, but there were inaccuracies. I believe additional features could improve the model’s accuracy. Firstly, having more data would enhance learning and better classify the 'unsure' cases. Additionally, incorporating a web scraping algorithm to find users' social media profiles, such as Facebook, could provide more direct age information from their bio. However, this raises ethical concerns, so it was not implemented in this project but could potentially improve the prediction score.


# Guide to start model

- open a terminal window
- pip3 install -r requirements.txt
- python3 main.py
(You will be asked to write an email address)
- write email address and press enter
- results are presented
- write 'exit' to quit program