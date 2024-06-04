import json
import os

from email_processor import process_emails

if __name__ == "__main__":
    """
    Main script to process email addresses and infer age groups.

    This script reads email addresses from a file, processes them to infer age groups,
    and saves the results to a JSON file.
    """
    # Read email addresses from a file
    with open('data/raw/emails.txt', 'r') as file:
        emails = file.read().splitlines()

    # Process emails to infer age groups
    results = process_emails(emails)

    # Ensure the output directory exists
    os.makedirs('data/output', exist_ok=True)

    # Save results to a JSON file, one result per line
    with open('data/output/results.json', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
