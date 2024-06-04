import re

import numpy as np


def softmax(scores, temperature=1.0):
    """Compute softmax values for each set of scores in x with temperature."""
    scores = np.array(scores) / temperature
    e_x = np.exp(scores - np.max(scores))
    return e_x / e_x.sum()


def is_valid_email(email):
    pattern = re.compile(r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)")
    return re.match(pattern, email) is not None


def fix_email(email):
    if email.count('@') != 1:
        return email

    local_part, domain_part = email.split('@')
    domain_part = domain_part.replace(',', '.')
    if domain_part.endswith('..'):
        domain_part = domain_part.rstrip('.') + '.com'
    domain_part = re.sub(r'\.\.+', '.', domain_part)

    return f"{local_part}@{domain_part}"


def clean_emails(emails):
    cleaned_emails = []
    original_emails = []
    for email in emails:
        email = email.strip().lower()
        fixed_email = fix_email(email)
        if is_valid_email(fixed_email):
            cleaned_emails.append(fixed_email)
            original_emails.append(email)
    return original_emails, cleaned_emails
