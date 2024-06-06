# Uptain Data Science Coding Challenge

## Solution Description

This solution builds a machine learning model to predict the age category of a person based on their email address. The steps include data preprocessing, feature extraction, model training, and evaluation.

### Features Extracted
To predict the age category from email addresses, several features were extracted:

1. **Email domain**: Common domains (e.g., gmail.com, yahoo.com, hotmail.com) can indicate different age groups. For example, older generations may be more likely to use domains like aol.com, while younger users may prefer gmail.com.
2. **Presence of numbers**: Younger people often include birth years in their email addresses. This feature captures whether numeric characters are present in the email address.
3. **Length of the email**: The total length of the email address may correlate with different age groups, as different generations may prefer shorter or more concise addresses.
4. **Character Counts**: This includes counts of numeric, alphabetic, and special characters. These counts help understand the structure and complexity of the email address.
5. **Year extraction**: Extracting years from email addresses can provide direct indicators of birth years, which are crucial for age prediction.

### Preprocessing Steps

#### Detailed Explanation:

1. **Data Extraction**: 
   - The email data is read from a provided text file (`emails.txt`). Each email address is standardized to lowercase to maintain uniformity.

2. **Feature Engineering**:
   - **Domain Indicators**: Boolean features are created for popular email domains. This helps in identifying common patterns associated with different age groups.
   - **Numeric Presence**: A boolean feature that indicates if numeric characters are present in the email. This can hint at the inclusion of birth years or other significant numbers.
   - **Length**: The length of the email address is calculated, which might correlate with user preferences across different age groups.
   - **Character Counts**: Counts of numeric, alphabetic, and special characters are computed to understand the composition of the email address.
   - **Year Extraction**: The first sequence of 2-4 digits is extracted from the email, assuming it represents a year, which is often related to the user’s birth year.

3. **Handling Missing Data**: 
   - In this scenario, missing data handling is not required as all extracted features are derived directly from the email addresses.

4. **Label Simulation**: 
   - For the purpose of demonstration, age labels are simulated randomly. In a real-world scenario, actual age labels would be necessary for accurate model training and validation.

### Model Training
Three different models were selected for training to provide a comprehensive comparison and understand which model performs best for this specific task:

1. **Random Forest Classifier**:
   - **Reason for Selection**: Random Forest is robust and can handle non-linear relationships well. It is also useful for understanding feature importance, which can provide insights into which features are most influential in predicting the age category.
   - **Implementation**: This model was trained using the default settings with a moderate number of trees to balance between computational efficiency and performance.

2. **Logistic Regression**:
   - **Reason for Selection**: Logistic Regression is simple and interpretable, making it a good baseline model. It is particularly well-suited for binary classification but can be extended to multi-class classification.
   - **Implementation**: The model was trained with standard parameters, and we used multi-class settings to adapt it for this classification task.

3. **Support Vector Machine (SVM)**:
   - **Reason for Selection**: SVM is effective for high-dimensional spaces and can perform well with a clear margin of separation between classes.
   - **Implementation**: The SVM model was trained with a linear kernel to manage computational complexity while still providing good performance for the classification task.

### Evaluation
The models were evaluated using a classification report to understand the precision, recall, f1-score, and support for each class. These metrics provide a comprehensive overview of how well each model performs in predicting the age categories.

### Visualization and Analysis

#### Distribution of Age Labels
A bar plot was created to visualize the distribution of age labels. This helps in understanding the balance of different age categories in the dataset. The dataset appears to be relatively balanced, with a slightly higher number of 'unsure' labels.

#### Feature Correlation Matrix
A heatmap was used to visualize the correlation matrix of numeric features. This helps in identifying relationships between features. Notable correlations include:
- `length` and `alpha_count`: High positive correlation, indicating that longer emails tend to have more alphabetic characters.
- `num_count` and `year`: Moderate positive correlation, suggesting that years in the email often contribute to the numeric character count.

#### Feature Distributions
Histograms and count plots were used to visualize the distributions of various features. These plots provide insights into the variability and distribution of each feature:
- **Length of Email**: Most emails are around 20-30 characters long.
- **Numeric Character Count**: Many emails have a low count of numeric characters, with a few having higher counts.
- **Alphabetic Character Count**: The distribution is similar to the length of emails.
- **Special Character Count**: Many emails have few special characters, but there is a notable number with higher counts.
- **Year Extracted**: The years extracted show peaks around common recent years.

### Results
The performance of the models was evaluated using classification reports. Below are the metrics for each model:

#### Random Forest Classifier
- **Precision, Recall, F1-Score**: The metrics indicate that the model struggles with predicting certain classes, particularly 'medium' and 'old'. The 'unsure' category has the highest recall, suggesting it is easier to identify.
- **Support**: Indicates the number of occurrences of each class in the test data.

#### Logistic Regression
- **Precision, Recall, F1-Score**: Similar to Random Forest, Logistic Regression also struggles with predicting 'medium' and 'old' categories. This model serves as a baseline for comparison.
- **Support**: Provides the number of true instances for each class.

#### Support Vector Machine (SVM)
- **Precision, Recall, F1-Score**: The SVM model shows similar performance patterns with high recall for the 'unsure' category but struggles with 'medium' and 'old' categories.
- **Support**: Reflects the distribution of the test dataset.

### Usage
To start the ML model from the terminal, follow these steps:

1. **Preprocess the Data**
    1. Ensure you have the `emails.txt` file in your working directory.
    2. Run the preprocessing script to extract features and save the processed data:
        ```bash
        python preprocess.py
        ```

2. **Train the Models**
    1. Run the training script to train and evaluate the models. This script will save the trained models and the evaluation results:
        ```bash
        python train_models.py
        ```

3. **Predict Ages**
    1. Ensure the `emails.txt` file is in the working directory.
    2. Run the prediction script to predict the age categories for the emails. By default, it uses the Random Forest model:
        ```bash
        python predict.py
        ```

### Detailed Guide to Start the ML Model from Terminal
1. **Setup the Environment**:
    1. **Create a Virtual Environment**:
        ```bash
        python -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        ```
    2. **Install Required Libraries**:
        ```bash
        pip install pandas scikit-learn numpy matplotlib seaborn
        ```

2. **Preprocess the Data**:
    1. Ensure the `emails.txt` file is in the same directory as the script.
    2. Run the preprocessing script:
        ```bash
        python preprocess.py
        ```
    3. This will generate `processed_emails.csv` and save visualization plots in the `images` folder (if you modify the script to save the plots).

3. **Train the Models**:
    1. Run the training script:
        ```bash
        python train_models.py
        ```
    2. This script trains Random Forest, Logistic Regression, and SVM models, evaluates them, and saves the models as `random_forest_model.pkl`, `logistic_regression_model.pkl`, and `svm_model.pkl`. The evaluation results are saved in `model_results.json`.

4. **Predict Ages**:
    1. Run the prediction script:
        ```bash
        python predict.py
        ```
    2. This script reads the `emails.txt` file, uses the Random Forest model to predict the age categories, and prints the results.

### Requirements
- Python 3.x
- pandas
- scikit-learn
- numpy
- re
- matplotlib
- seaborn

### System Requirements
Minimal requirements:
- RAM: 4GB
- CPU: Dual-core

### Additional Notes:
- **UndefinedMetricWarning**: The warnings indicate that some labels had no predicted samples. This is common in imbalanced datasets or when the model isn't fitting well. To address this, consider:
  - Scaling the data (e.g., using `StandardScaler`).
  - Increasing the number of iterations (`max_iter`) for the Logistic Regression model.
  - Exploring alternative solvers for Logistic Regression as mentioned in the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression).

- **Model Performance**: The results indicate poor performance across all models, likely due to class imbalance or insufficient feature engineering. Additional steps to improve model performance might include:
  - Balancing the dataset using techniques like SMOTE.
  - Adding more meaningful features.
  - Tuning hyperparameters.


