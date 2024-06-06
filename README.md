# Uptain Data Science Coding Challenge

# Email Age Classifier

Python script processes a list of email addresses to extract, classify, and predict the age of a person based on email content. It features a comprehensive approach for data transformation, utilizing both machine learning models and an alternative mathematical approach for data processing.

Three machine learning models are utilized for age prediction:
- *Logistic Regression*
- *Random Forest Classifier*
- *Decision Tree Classifier*

These models are trained using the processed data, which is split into training and testing sets. After training, each model is saved to a file using `pickle`, making them reusable without the need for retraining.

## Alternative Data Processing Method
While working on the data processing, an alternative method using a mathematical equation was conceptualized and implemented in the `test_email_with_confidence` function. This method is intended to provide a quick assessment of the data’s completeness and the confidence level of the outputs based on the available email attributes.

## Testing the Models
- **test_email_with_model Function**: This function allows for testing the pre-trained models with any given email to predict the age class, providing output along with a confidence score.
  ```python
  test_email_with_model('your_email@example.com', 'model_filename.pkl')

# Disclaimer

This repository contains a list of generated test emails. Any real match with existing emails is purely coincidental and unintentional. All the emails here were generated for testing purposes only.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.