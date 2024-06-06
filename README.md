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
## Solution: KNN improve labeling + LightGBM +Optuna
### Data cleaning
   * Duplicate Removal: Duplicates are identified and removed to ensure each data entry is unique and valuable.
     83 duplicated emails that need to be deleted.
   * Typo Fixing: Instead of dropping entries with typos, they are corrected to preserve every piece of data, acknowledging that each data point is valuable.
     * Missing dot or duplicate dot before "com"
     * "," instead of "."
     * Missing "com"
### Initial Labeling
   * Use heuristic rules to label a portion of the data based on the extracted birth years from the usernames.
   * This initial labeling step provides a small amount of labeled data, the "unsure" labels could be further labeling with the help of the initial labeling.
     
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/4aca3b1b-eae6-421f-a1af-ffab6cb596e5)
### Exploratory Data Analysis (EDA)
   * Perform EDA to validate and refine the initial labels. This helps in understanding the data distribution and identifying potential issues.
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/9a409e5d-0b7f-496d-ade8-c5fa7f970848)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/3f8ad018-457a-40ac-8435-832c9045b318)
     * Young individuals tend to favor hotmail.com and organization.org.
     * Older individuals often prefer aol.com, tutanota.com, and mail.com.
     * Middle-aged individuals generally prefer yahoo.com.
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/b36e3281-975a-4e81-9967-201bb89ab93b)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/aa35f9e9-1704-4df0-8229-856b15dce983)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/881e2bca-2440-41f6-bad0-600f2a438ab3)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/b209b17c-ff4f-42cf-865a-3a8481335f81)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/3ce247cd-831a-443a-a24a-d09f19eada5d)
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/27b3b64c-adaa-4ea9-a5d3-27a28e67147a)
     * Old people do not like underscore.
     * Old people prefer capital.
### Semi-Supervised Learning with KNN-based Labeling:
   * Apply a K-Nearest Neighbors (KNN) algorithm to label more data based on the initial labeled data.
   * Use a high confidence threshold (e.g., >0.9) to further label some of the unlabeled data, ensuring that only highly confident predictions are used.
   * This step effectively leverages the labeled data to make predictions on the unlabeled data, thereby increasing the amount of labeled data.
     ![image](https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/0554b943-cb13-4ca0-afb2-b25cb989e9ac)
### Model Training with LightGBM + Optuna:
   * Use the expanded labeled dataset to train a LightGBM model.
   * Apply Optuna for hyperparameter tuning to ensure optimal model performance.
     <img width="414" alt="image" src="https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/6db343f5-90c1-465f-b0ed-e5d76df0a028">
### Model Inference
   Single and Batch Prediction: The model supports both single email predictions and batch predictions from a CSV file.
## Running the Project
### Running Locally

#### 1. Set Up and Activate Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```
#### 2. Install Required Packages
```bash
pip install -r requirements.txt
```
#### 3. Train the model
```bash
python3 model_train.py df_cleaned.csv
```
#### 4. Run Inference
* For Batch Prediction
```bash
python3 inference.py df_cleaned.csv
```
* For Single Prediction
```bash
python3 inference.py email@example.com
```
<img width="826" alt="image" src="https://github.com/boyuwangpsu/data-science-coding-challenge/assets/49320567/bc1366b5-b233-40d3-a172-c2a640cd2d5a">

### Running with FastAPI
#### 1. Ensure Virtual Environment is Activated
#### 2. Start the FastAPI Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
#### 3. Interact with the API
* For Batch Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch/" -H "Content-Type: application/json" -d '{"data_path": "df_cleaned.csv"}'
```
* For Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"email": "email@example.com"}'
```
### Running with Docker
#### 1. Build the Docker Image
```bash
docker build -t my_lightgbm_app .
```
#### 2. Run the Docker Container
```bash
docker run -d -p 8000:8000 my_lightgbm_app
```
#### 3. Interact with the API (from Docker)
* For Batch Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict_batch/" -H "Content-Type: application/json" -d '{"data_path": "df_cleaned.csv"}'
```
* For Single Prediction
```bash
curl -X POST "http://127.0.0.1:8000/predict/" -H "Content-Type: application/json" -d '{"email": "email@example.com"}'
```

# GitHub
* [How to fork a repository](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)
* [How to create a pull request from fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)

# Disclaimer

This repository contains a list of generated test emails. Any real match with existing emails is purely coincidental and unintentional. All the emails here were generated for testing purposes only.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
