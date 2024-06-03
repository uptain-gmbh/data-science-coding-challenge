# Uptain Data Science Coding Challenge

# Steps for prediction of the age with email as input data
- Install Python (my_version=3.10.12) and other dependencies (requirements.txt)
- Run command -> python3 main.py
- User input a real-world and valid email address -> Eg:anishm.92@gmail.com, anis_m91@aol.com
- Get a json formatted output prediction -> {'age': 'medium', 'score': 1.0}

# Problem Statement
First of all, let me mention the problem statement. We have a list of email address which are the only input data for the machine learning models to be trained on. The goal is to use them to predict the **age** of the email address holder. We have 4 classes for prediction which are the age groups of email address holders. The classes are as follows:
- young - a person is relatively young (18-30)
- medium - a person is middle-aged (30-50)
- old - a person is old (50+)
- unsure - the age can't be determined

# Data Exploration and Annotation
- There are 1073 emails in the dataset (emails.txt).
- There are 46 types email domains in total. However, after standardization, we have 15 types. **"gmail.com"** has the highest frequency of 368. Their distribtion can be observed in a barplot in the notebook (main.ipynb).
- After observing the dataset, we see that there is major gap in the dataset we have. It doesn't have any target labels which means data annotation is missing. So, to tackle this, we need to find a way to label them.
- There is no precise way to label the emails to a certain age group. The only real way is to actually know the age of the person the email address belongs too but this is not an option for us.
- Now, the only way I see is to make some assumptions about the dataset. After observing the emails, we see there are few assumptions we can make to label them. The observations and assumptions are as follows:
    - We can see that, there are 719 data samples which has digits in their emails. As observed in real-world, it is quite common to include your date of birth in your emails. Both full 4 digits year and 2 digits year are common. Leverage this pattern, the age of the email holders are calculated and thus categorized accordingly. For eg: samantha_turner1987@gmail.com would be categorized as **medium** age group and Elody_OConner51@gmail.com as **old**.
    -  Email domains are another potential source of information. Using their date of foundation and popular peak times information to calculate the age of email address doesn't sound too absurd. With some internet research, it was found that domains *aol.com*, *mail.com* were quite popular in early 90s and late 90s respectively and, domains such as *protonmail.com* and *icloud.com* are from 2010s. So, it is likely that the emails match their holder's age. Therefore, I categorized some emails which couldn't be categorized based on digits, based on their respective domains.
    - For emails where I couldn't leverage the above two paterns, I categorized them as **unsure**. 
    - After labeling all data samples, it is observed that **unsure** category has the most samples of 372. The barplot of their frequency can be observed in the notebook.

# Feature Engineering, Encoding and Scaling
- After the annotation, I created more features such as length of username, has_digit(if an email has a digit) , has_underscore(if an email has a underscore), has_period(if an eamil has a underscore), digits (digits in the email). These are some patterns I thought that the machine learning models could make some sense out of.
- Then, another feature domain was created according to the domain of the email address. There are 8 domains with significant frequency. They were expanded as dummy features and the rest of others were bunched together as *others* domain feature.
- All boolean features were encoded to 1(True) and 0(False).
- The two features digits and username length are in different scales, thus are standardized by using standard scaler. The same scaler is also used during inference time.

# Machine Learning Models Training and Evaluation
- Before training models, I splitted the dataset into train and test in the ratio of 0.2 (test/train).
- There are so many machine learning models to use but I started with the most common algorithm, logisitic regression as a baseline classifier. 64% accuracy was observed with it at it's default hyperparameters.
- I chose balanced accuracy as my metric to evaluate the performance of models. Because, the data samples between classes are not balanced and accuracy makes sense for this problem.
- Majority of features we have are categorical and the data itself seems to follow some intuitive rules thus tree-based algorithms like Decision Tree and Random Forest are picked for training. Likewise, KNearestNeighbors are also good choice for low dimensional data so it is also picked.
- The performance of a machine learning model should not rely on just one data split and iteration. To ensure that I decided to use cross validation. Cross validation can also be used to find optimum set of hyperparameters of a machine learning model. Thus, to tune best sets of hyperparameters, I ran machine learning models at a diverse range of hyperparameters and observed the range of their peak performances. Based on those ranges, I created better sets of hyperparameters and used nested cross-validation to observe the performance of machine learning models in the most unbiased and reliable way. Nested CV ensures that we run our models at different splits of the data and also tune the best hyperparameters without any data leakage.
- After Nested CV for KNN, Decision Tree and Random Forest, it is observed that **decision tree** has the best accuracy(98.3%[mean]) and also significantly lesser training time(23s[mean]) as compared to Random Forest. 

# Conclusion on Machine Learning Models' Performance
- The performance of tree-based models like Decision Tree and Random Forest seems to be too good to be true. However, it is not something very unusual. At the initial preprocessing steps, we assumed two main conditions for creating our target data using the email address.
- The email domain names and digits were used for categorizing the email address to certain age group.
- Therefore, these simple assumptions are something to obvious to decision tree classifier that it performs so well.
- Having said, the winner model no where simulates the production level machine learning model.
- But, if the new unseen real-world data follows similar patterns that we have assumed then the winner model will certainly be a great choice.

# Implementation details
All implementation details can be read on a jupyter notebook (main.ipynb). It consists of detail explanation of the steps and decisions taken.

# Disclaimer

This repository contains a list of generated test emails. Any real match with existing emails is purely coincidental and unintentional. All the emails here were generated for testing purposes only.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.