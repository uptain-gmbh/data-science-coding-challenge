import re


def extract_email_domain(email):
    split_list = email.split('@')
    domain = ''.join(split_list[-1])
    #rest are username
    username = ''.join(split_list[:-1])
    return username, domain

def check_email_format(domain):
    # only 1 . after @, and not the last character
    if domain.count('.') != 1:
        return False
    elif domain[-1] == '.':
        return False
    else:
        return True
    
def extract_digits(username):
    digit = re.findall(r'\d+', username)
    if digit:
        return digit[0], len(digit[0])
    else:
        return 12345, 0