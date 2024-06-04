import re

import numpy as np


def softmax(scores, temperature=1.0):
    """
    Compute softmax values for each set of scores with temperature scaling.

    Args:
        scores (list or np.ndarray): Input scores to be normalized.
        temperature (float): Temperature parameter to control the sharpness of the softmax distribution.

    Returns:
        np.ndarray: Softmax probabilities.
    """
    scores = np.array(scores) / temperature
    e_x = np.exp(scores - np.max(scores))  # Stability improvement
    return e_x / e_x.sum()


def is_valid_email(email):
    """
    Validates an email address using regex.

    Args:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email is valid, False otherwise.
    """
    pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    return re.match(pattern, email) is not None


def fix_email(email):
    """
    Fixes common issues in email addresses.

    This function fixes issues like commas instead of dots in the domain part and
    removes consecutive dots.

    Args:
        email (str): The email address to be fixed.

    Returns:
        str: The fixed email address.
    """
    if email.count('@') != 1:
        return email  # Return as is if there's not exactly one '@'

    local_part, domain_part = email.split('@')
    domain_part = domain_part.replace(',', '.')  # Replace commas with dots
    if domain_part.endswith('..'):
        domain_part = domain_part.rstrip('.') + '.com'  # Replace ending '..' with '.com'
    domain_part = re.sub(r'\.\.+', '.', domain_part)  # Replace consecutive dots with a single dot

    return f"{local_part}@{domain_part}"


def clean_emails(emails):
    """
    Cleans a list of email addresses.

    This function normalizes the email addresses by stripping whitespace, converting to lowercase,
    fixing common issues, and validating them.

    Args:
        emails (list): List of email addresses to be cleaned.

    Returns:
        tuple: Two lists - original cleaned emails and valid cleaned emails.
    """
    cleaned_emails = []
    original_emails = []
    for email in emails:
        email = email.strip().lower()  # Normalize email by stripping whitespace and converting to lowercase
        fixed_email = fix_email(email)
        if is_valid_email(fixed_email):
            cleaned_emails.append(fixed_email)
            original_emails.append(email)
    return original_emails, cleaned_emails
