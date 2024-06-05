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

---
---
---

# Solution
**TODO**

## Requirements
This solution is pretty lightweight and does not require specific CPU or RAM constraint.

## Installation
First navigate to the solution folder.
``` 
: cd solution 
```
Then make an virtual environment.
``` 
: python -m venv .venv 
```
Depending on the OS, switch to the environment. Here is for Windows:
```
: .venv\Scripts\activate
``` 
Install requirements
```
(.venv): pip install -r requirements.txt
```

## Inference
The main entry point for inference is `main.py`. The module requires one of two argumants: `--file` or `--email`. The file format must be a text file just like what was intially provided.
```
-f FILE, --file FILE  Path to a text file with emails.
-e EMAIL, --email EMAIL Email address for inference. i.e. 'john.smith@example.com'
```
### Example for an email
```
(.venv): python main.py -e john.smith@example.com
{'email': 'john.smith@example.com', 'age': 'unsure', 'score': 0.68}
```

### Example for a text file
We can use the provided `emails.txt` file located in root directory of this repository. Since we are in `solution` directory, we must supply the file as follow:
```
(.venv): python main.py -f ../emails.txt 
Started prediction ...
Prediction done.
```
Depending on your hardware, the inferance may take betwee 1 to 3 minutes. After completion you can check `result.json` file.