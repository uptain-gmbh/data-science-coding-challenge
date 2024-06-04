import json
import os

from email_processor import process_emails

if __name__ == "__main__":
    with open('data/raw/emails.txt', 'r') as file:
        emails = file.read().splitlines()

    results = process_emails(emails)

    # Ensure the output directory exists
    os.makedirs('data/output', exist_ok=True)

    # Save results to a JSON file
    with open('data/output/results.json', 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
