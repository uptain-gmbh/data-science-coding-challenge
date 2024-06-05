import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from utils import is_valid_email, domain_extraction, extract_info, get_outlier_bounds
#import warnings
import os
import argparse


def main(emails_txt_file):

    #Log file
    path = "log_files"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory 'results' is created!")
    from datetime import datetime
    now = datetime.now()
    date_time = now.strftime("%m%d%Y_%H%M%S")
    filename='log_files/data_analysis' + date_time + '.log'
    logging.basicConfig(filename=filename, level=logging.INFO, filemode='w', format='%(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())

    # Tp suppress future warning from Seaborn
    #warnings.simplefilter(action='ignore', category=FutureWarning)

    df = pd.read_csv(emails_txt_file, sep=" ", names=["emails"])
    # Find duplicate rows
    duplicates = df[df.duplicated()]
    logging.info('Number of duplicated emails: {} '.format(duplicates.shape))
    # Drop duplicates
    df = df.drop_duplicates()
    logging.info('Number of emails after removing duplicates: {} '.format(df.shape[0]))

    # Check the validity of email
    df['valid_email'] = df['emails'].apply(is_valid_email)
    print(df.to_string())
    invalid_email = df[df['valid_email'] == False]
    #print(invalid_email.to_string())
    # removed invalid emails
    df = df[df['valid_email'] == True]

    #df['extracted_numbers'] = df['emails'].str.extract(r'(\d+)')
    #print(df.to_string())

    df[['username_len', 'estimated_age', 'age_range']] = df['emails'].apply(extract_info)

    # Extract domain
    df['domain'] = df['emails'].apply(domain_extraction)
    print(df.to_string())

    # To check if plots directory exists or not
    save_dir = "plots"
    # Check whether the specified directory exists or not
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
        print(f'{save_dir} directory created')

    # domain cout plot
    plt.figure(figsize=(8, 8))
    sns.countplot(data=df, x='domain')
    save_domain_countplot_path = save_dir + '/' + 'domain_historam.png'
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.title('Histogram of domain')
    plt.savefig(save_domain_countplot_path)

    # Pie chart
    target_counts = df['age_range'].value_counts()
    # Create the pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%', startangle=140)
    save_piechart_path = save_dir + '/' + 'target_pie_chart.png'
    plt.title('Target Pie Chart')
    plt.savefig(save_piechart_path)


    # Analyze the age distribution
    plt.figure(figsize=(8, 8))
    plt.hist(df['estimated_age'], bins=10)
    plt.xlabel('estimated_age')
    plt.ylabel('count')
    plt.title('Histogram of estimated_age')
    save_histogram_path = save_dir + '/' + 'age_histogram.png'
    plt.savefig(save_histogram_path)

    # Statistical analysis
    logging.info('Analyze the statistical data')
    logging.info(df['estimated_age'].describe())
    lower_bound, upper_bound = get_outlier_bounds(df['estimated_age'])
    logging.info('Upper bound of  estimated_age: {}'.format(upper_bound))

    # Box plot
    plt.figure(figsize=(8, 8))
    sns.boxplot(df['estimated_age'])
    save_boxplot_path = save_dir + '/' + 'boxplot.png'
    plt.title('Box Plot')
    plt.savefig(save_boxplot_path)
    '''
    # Bivariate analysis
    plt.figure(figsize=(8, 8))
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    sns.stripplot(data=df, x='domain', y='estimated_age', jitter=True, hue='domain', palette='Set2', marker='o', size=8)
    save_stripplot_path = save_dir + '/' + 'stripplot.png'
    plt.title('Strip plot between domain and estimated_age')
    plt.savefig(save_stripplot_path)
    '''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--emails_txt_file', type=str,
                        required=True, help="email address")
    args = parser.parse_args()
    main(args.emails_txt_file)