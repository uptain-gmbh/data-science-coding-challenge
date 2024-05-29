# Uptain Data Science Coding Challenge

It seems like you're trying out for a position at [Uptain](https://uptain.de) or you've found this and would like to apply.
We're excited to see your creativity and skills in action — we 💚 those things at [Uptain](https://uptain.de)!

Your goal is to build an ML model that can detect the age of a person based on their email address. 
Once you've completed the challenge, please create a Pull Request and we will get in touch. 🤙

Fork this repo and get started 🥷

## Brief

This repository contains a file [emails.txt](./emails.txt), which has an unsorted list of emails. 
Each email has a possible association with an age based on different attributes. 
Your task is to find these attributes in the emails and build a model that can predict the age of a person based on their email address.

## Technology Selection

It is up to you to select your stack. Feel free to choose the one that enables you to complete the challenge.
*   You can use any libraries, task runners, or frameworks you like; however, we expect the solution to be written in Python.

## Requirements

*   The output of the model must produce a single JSON line like:
    * ```{ "age": "{age_class}", "score": {score_value} }``` 

    For example:
    1.   ```{ "age": "young", "score": 1 }``` 
    2.   ```{ "age": "medium", "score": 0.5 }``` 
    3.   ```{ "age": "old", "score": 0.75 }``` 
    4.   ```{ "age": "unsure", "score": 0 }``` 

    Where `age` can be one of four options:

    * young - a person is relatively young (18-30)
    * medium - a person is middle-aged (30-50)
    * old - a person is old (50+)
    * unsure - the age can't be determined

    The `score` should be a float value between `0` and `1`, where `1` is the most confident prediction 
    and `0` is the least confident prediction. 

*   Please provide a description of your solution and the decisions you made in the `README.md` file. 
    * This must include the method of finding the attributes in the emails and the model training process you used to predict the age.
    * And a guide of how to start the ML model from terminal, correctly provide input and receive an output.
    * You can also include any additional information you think is relevant, possible the minimal RAM and CPU requirements, etc.



# GitHub
* [How to fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
* [How to create a pull request from fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

# Disclaimer

This repository contains a list of generated test emails. Any real match with existing emails is purely coincidental and unintentional. All the emails here were generated for testing purposes only.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Solution and methodology

The approach taken was to build a classifier using either shallow methods or deep learning methods.
First, Invalid email adresses were filtered out using RegEx. Then, the potential dates in the email adresses were extracted also using RegEx. Then, based on certain criteria the labels were created.
The email adresses are then stripped off the Domain name and converted to lowercase for consistency. Afterwards, the addresses were converted into a character-wise numeric representations(tokens) and padded to uniform length of 32 and then split into training and testing sets. 

Random forests, a fully connected neural network and convolutional neural networks were trained and had their performance tested. The random forest outperformed both neural networks, which is probably due to the small training data. 

To run the model from the terminal, simply navigate to the ```data-science-coding-challenge-main``` folder then run ```python inference.py```. you will then be prompted to enter an email address. and here the full email address must be entered. the purification and tokenization process is taken care of internally.


## Extra info

```Playground.py``` contains the exploration and the pre-processing of the dataset as well as training the classifier. All steps are underneath their respective markdown.

```training_data.csv```, ```validation_data.csv``` and ```test_data.csv``` contain the split dataframes used to train the pytorch models. However, due to the poor performance the models were not used in the final inference. Yet, the model is still present in ```playground.py``` alongside the train and validation loops.

Some intermediate text files and CSVs are kept in the project to showcase the progress made.