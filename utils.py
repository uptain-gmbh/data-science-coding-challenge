import re
import json
with open("mapping.json", 'r') as f:
    mapping = json.load(f)

def is_valid_email(email):
    """Check if the email is a valid format."""

    # Regular expression for validating an Email

    # regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w+$'
    # regex = re.compile(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+')
    # regex = re.compile(r"^[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$" )
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'

    # If the string matches the regex, it is a valid email

    if re.match(regex, email):

        return True

    else:

        return False
def pad_tokens(token_list: list):
    # inputs to the network must be of uniform length.
    len_tokens = len(token_list)
    if len_tokens > 32 :
        return token_list[:32]
    else:
        pad_val = [0] * 32
        extended = token_list + pad_val # length of (original_length + max_len)
        return extended[:-len_tokens]
    
def tokenize_and_pad(email: str, mapping):
    
    email = email.split('@')[0].lower()
    tokens = [mapping[ch] for ch in email]
    tokens = pad_tokens(tokens)
    return tokens
