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

*   The output of the model must produce a JSON value like:
    *   ```{ "age": "young", "score": 1 }``` 
    *   ```{ "age": "medium", "score": 0.5 }``` 
    *   ```{ "age": "old", "score": 0.75 }``` 
    *   ```{ "age": "unsure" }``` 

    Where `age` can be one of four options:

    * young - a person is relatively young (18-30)
    * medium - a person is middle-aged (30-50)
    * old - a person is old (50+)
    * unsure - the age can't be determined

    The `score` should be a float value between `0` and `1`, where `1` is the most confident prediction 
    and `0` is the least confident prediction. 

*   Please provide a description of your solution and the decisions you made in the `README.md` file. 
    This must include the method of finding the attributes in the emails and the model training process you used to predict the age.

# Disclaimer

This repository contains a list of generated test emails. Any real match with existing emails is purely coincidental and unintentional. All the emails here were generated for testing purposes only.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.