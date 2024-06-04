
# Age Prediction from Email Addresses


**Created by:** 🎨 **Yatin Arora**  
**Email:** ✉️ [yatin.arora@outlook.de](mailto:yatin.arora@outlook.de)

## Overview

This project aims to infer the probable age group of users based on their email addresses. The possible age groups are:
- Young (18-30)
- Medium (30-50)

## Overview

This project aims to infer the probable age group of users based on their email addresses. The possible age groups are:
- Young (18-30)
- Medium (30-50)
- Old (50+)
- Unsure

The prediction is made

## Overview

This project aims to infer the probable age group of users based on their email addresses. The possible age groups are:
- Young (18-30)
- Medium (30-50)
- Old (50+)
- Unsure

The prediction is made using various features derived from the email address, such as the username and domain. We utilize BERT embeddings for usernames and K-Nearest Neighbors (KNN) for classification.

## Project Structure

- `main.py`: Entry point for the script, reads emails, processes them, and saves results.
- `email_processor.py`: Contains the function for processing emails.
- `age_inference.py`: Contains the core logic for extracting features and inferring age.
- `utils.py`: Contains utility functions for cleaning emails and computing softmax probabilities.

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have a file named `emails.txt` in the `data/raw/` directory containing the emails to be processed, one per line.

## Running the Script

Run the main script to process the emails and generate age predictions:
```sh
python main.py
```

The results will be saved in `data/output/results.json`.

## Explanation of Predictions

### Feature Extraction

1. **Username Extraction**: The part of the email before the `@` symbol is considered the username.
    
    **Example**:
    - Email: `john.doe1980@example.com`
    - Username: `john.doe1980`

2. **Year Extraction from Username**: We extract numbers from the username, considering valid years as:
    - Four-digit numbers between 1900 and the current year.
    - Two-digit numbers, interpreted as years in the 1900s or 2000s.
    - Single-digit numbers, interpreted as years in the 2000s.
    
    This helps in identifying potential birth years.

    **Examples**:
    - Username: `john.doe1980`
        - Extracted Year: 1980
    - Username: `susan.75`
        - Extracted Year: 1975
    - Username: `kid.9`
        - Extracted Year: 2009

### Age Inference

1. **Age from Birth Year**: The extracted birth year is used to infer age.
    - **Young**: 18-30 years
    - **Medium**: 30-50 years
    - **Old**: 50+ years
    - **Unsure**: Age <= 10 years or invalid year
    
    **Example**:
    - Birth Year: 1980
        - Age: 44 (in 2024)
        - Age Group: Medium

    **Weight Assignment**:
    - **Weight**: 0.8
    - **Reason**: Birth year extracted from the username is a strong indicator of the user's age, hence given a high weight.

2. **Age from Domain Type**: The domain type can provide hints about the user's age.
    - `.edu`: Likely young (students)
    - `.gov`: Likely old (government employees)
    - Other domains: Age is unsure
    
    **Example**:
    - Email: `john.doe@example.edu`
        - Domain: `.edu`
        - Age Group: Young

    **Weight Assignment**:
    - **Weight**: 0.3
    - **Reason**: Domain type can indicate age groups but is less reliable than birth year, hence given a moderate weight.

3. **Domain Age**: The age of the domain is also considered.
    - Domain age < 7 years: Likely young
    - Domain age 7-20 years: Medium
    - Domain age > 20 years: Old
    
    **Example**:
    - Domain: `example.com`
        - Creation Date: 1995
        - Domain Age: 29 years (in 2024)
        - Age Group: Old

    **Weight Assignment**:
    - **Weight for Young**: 0.4
    - **Weight for Medium**: 0.2
    - **Weight for Old**: 0.3
    - **Reason**: Domain age can indicate the user's age group, but with varying reliability. Older domains might belong to older users, but newer domains can belong to users of any age group, thus weights are adjusted accordingly.

### Combined Age Prediction

- **Softmax Normalization**: We use softmax to combine scores from different sources, ensuring probabilities sum to 1.
    
    **Example**:
    - Scores: [1.0, 0.5, 0.2, 0.1]
    - Softmax Probabilities: [0.617, 0.228, 0.091, 0.064]

- **Weight Calculation**: Weights are assigned to different age sources based on confidence levels.
    
    **Weights**:
    - Birth Year: 0.8
    - Domain Type: 0.3
    - Domain Age: 0.4 (young), 0.2 (medium), 0.3 (old)
    
    **Example**:
    - Combined Score for Young: 0.8 (birth year) + 0.3 (domain type) + 0.4 (domain age) = 1.5 (before normalization)

### Edge Cases

1. **Invalid or Missing Data**: Emails without valid birth years or domains result in an `unsure` classification.
    - **Example**: `invalid@domain`
        - Result: Unsure

2. **Multiple Potential Birth Years**: We take the most plausible year considering the context and predefined rules.
    - **Example**: `john19801990`
        - Extracted Years: [1980, 1990]
        - Selected Year: 1990

3. **Empty Usernames**: These are handled gracefully and classified as `unsure`.
    - **Example**: `@domain.com`
        - Result: Unsure

## Parameter Choices and Reasoning

- **Age > 10**: We assume people younger than 10 are unlikely to have their own email addresses. This is a reasonable cutoff considering internet usage patterns.
- **Weights**: Higher weights for more reliable sources (e.g., birth year from username) ensure better prediction accuracy. Weights are empirically determined based on the perceived reliability of each source.
- **Softmax Temperature**: Adjusting temperature in softmax helps in controlling the confidence spread across classes. Lower temperatures make the model more confident in its predictions.

## Future Enhancements

- Improve domain-specific age prediction by incorporating more domain patterns.
- Enhance username parsing with advanced NLP techniques.

---
