import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import joblib
import math

# Age thresholds
# young: 18-29 | 2006 - 1995
# middle: 30-49 | 1994 - 1975
# old: 50 - 84 | 1974 - 1940


def extract_domain(email):
    return (email.rsplit("@")[-1]).rsplit(".")[0]


def extract_username(email):
    return email.rsplit("@")[0]


def extract_birth_year(username):

    # Find all sequences of digits
    digit_sequences = re.findall(r"\d+", username)

    # If there are multiple sequences, the first one will be used.
    for seq in digit_sequences:
        year = int(seq)
        # Check if the sequence is 4 digits long
        if len(seq) == 4:
            if 1940 <= year <= 2006:
                return int(year)
        # Check if the sequence is 2 digits long
        elif len(seq) == 2:
            if 40 <= year <= 99:
                return int(1900 + year)
            elif str(year) in ["00", "01", "02", "03", "04", "05", "06"]:
                return int(2000 + year)

    return None


def extract_names(username):
    regex = r"[a-zA-Z]{3,}"
    result = []
    matches = re.finditer(regex, username)
    for matchNum, match in enumerate(matches):
        result.append(match.group())
    return result


def email_validation(email):
    regex = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"

    if re.fullmatch(regex, email):
        return True
    else:
        return False


def preprocess_emails_to_df(file_path):

    # Read the file line by line and strip whitespace from each line
    with open(file_path, "r") as file:
        emails = [line.strip().rstrip(",") for line in file]

    # Convert the cleaned list to a DataFrame
    df = pd.DataFrame(emails, columns=["email"])

    df["email"] = df["email"].str.lower()

    df.drop_duplicates(inplace=True)

    mask = df.email.apply(email_validation)
    df = df[mask]

    return df


def sum_features(dict_a, dict_b):
    if dict_a.keys() != dict_b.keys():
        raise ValueError("Features must have the same keys.")

    # Create a new dictionary to store the averaged values
    sum_dict = {}
    for key in dict_a:
        sum_dict[key] = dict_a[key] + dict_b[key]

    return sum_dict


def average_features(dict_a, dict_b):
    if dict_a.keys() != dict_b.keys():
        raise ValueError("Features must have the same keys.")

    # Create a new dictionary to store the averaged values
    avg_dict = {}
    for key in dict_a:
        avg_dict[key] = dict_a[key] + dict_b[key] / 2

    return avg_dict


def preprocess_names_to_dict(file_path):
    df_names = pd.read_csv(file_path)

    # Lower case column names
    df_names.columns = map(str.lower, df_names.columns)

    # Remove years before 1940 and after 2006
    df_names = df_names[df_names.year > 1939]
    df_names = df_names[df_names.year < 2007]
    df_names.name = df_names.name.str.lower()

    # The dict will years as keys
    names_dict = {}
    for year in df_names.year.unique():
        names_dict[year] = df_names[df_names.year == year].copy()

    # We scale count of names within a year between 0 and 1.
    # We also scale for two geneders since ...
    for year in names_dict.keys():
        m_scaler = MinMaxScaler()
        f_scaler = MinMaxScaler()
        temp = names_dict[year]

        m_scaler.fit(temp[temp.gender == "M"][["count"]])
        f_scaler.fit(temp[temp.gender == "F"][["count"]])

        temp.loc[temp.gender == "M", ["count_scaled"]] = m_scaler.transform(
            temp[temp.gender == "M"][["count"]]
        )
        temp.loc[temp.gender == "F", ["count_scaled"]] = f_scaler.transform(
            temp[temp.gender == "F"][["count"]]
        )

    return names_dict


def get_scaled_features_by_name(names_dict, name):
    result = {"young": 0, "middle": 0, "old": 0}

    for year in range(1995, 2006 + 1):
        result["young"] += names_dict[year][
            names_dict[year].name == name
        ].count_scaled.sum()
    result["young"] = result["young"] / 12

    for year in range(1975, 1994 + 1):
        result["middle"] += names_dict[year][
            names_dict[year].name == name
        ].count_scaled.sum()
    result["middle"] = result["middle"] / 20

    for year in range(1940, 1974 + 1):
        result["old"] += names_dict[year][
            names_dict[year].name == name
        ].count_scaled.sum()
    result["old"] = result["old"] / 35

    return result


def get_scaled_features_by_birth_year(birth_year):
    result = {"young": 0, "middle": 0, "old": 0}

    if birth_year is None:
        return None
    elif 1940 <= birth_year <= 1974:
        result["old"] = 2
    elif 1975 <= birth_year <= 1994:
        result["middle"] = 2
    else:
        result["young"] = 2

    return result


def get_scaled_features_by_domain(domain):
    result = {"young": 0, "middle": 0, "old": 0}
    if domain in ["aol", "hotmail", "mail"]:
        result["old"] = 0.3
        return result
    else:
        return None


def normalize_feature(feature):
    sum = feature["young"] + feature["middle"] + feature["old"]
    if sum == 0:
        return feature
    return {
        "young": feature["young"] / sum,
        "middle": feature["middle"] / sum,
        "old": feature["old"] / sum,
    }


def get_scaled_features(names_dict, email):
    username = extract_username(email)
    domain = extract_domain(email)
    strings = strings = extract_names(username)
    birth_year = extract_birth_year(username)

    feature_result = {"young": 0, "middle": 0, "old": 0}
    feature_all = {"email": email}
    selected_name = None
    for str in strings:
        feature_current = get_scaled_features_by_name(names_dict, str)
        if sum(feature_result.values()) < sum(feature_current.values()):
            feature_result = feature_current
            selected_name = str

    # feature_result = normalize_feature(feature_result)
    feature_all["selected_name"] = selected_name
    feature_all["feature_name"] = feature_result

    feature_birth_year = get_scaled_features_by_birth_year(birth_year)
    if feature_birth_year is not None:
        feature_all["feature_birth_year"] = feature_birth_year
        feature_all["selected_birth_year"] = birth_year
        feature_result = sum_features(feature_birth_year, feature_result)

    feature_domain = get_scaled_features_by_domain(domain)
    if feature_domain is not None:
        feature_all["feature_domain"] = feature_domain
        feature_all["selected_domain"] = domain
        feature_result = sum_features(feature_domain, feature_result)

    return normalize_feature(feature_result), feature_all


def make_dataset(emails_df, names_dict):
    dataset = pd.DataFrame()
    for i in range(0, len(emails_df)):
        temp = pd.DataFrame(
            [
                {
                    "email": emails_df.iloc[i]["email"],
                    **(get_scaled_features(names_dict, emails_df.iloc[i]["email"])[0]),
                }
            ]
        )
        dataset = pd.concat([dataset, temp], ignore_index=True)
    return dataset


def gmm_fit_save(dataset):

    X = dataset.loc[:, ["young", "middle", "old"]]
    gmm = GaussianMixture(
        n_components=4, random_state=0, covariance_type="tied", tol=5e-1, reg_covar=5e-2
    )

    gmm.fit(X)
    joblib.dump(gmm, "gmm_model.pkl")


def gmm_predict(dataset):
    gmm = joblib.load("gmm_model.pkl")

    X = dataset.loc[:, ["young", "middle", "old"]]

    result = dataset.copy()
    result["cluster"] = gmm.predict(X)
    result["cluster"] = result["cluster"].astype("category")

    def calc_conf(row):
        match row["cluster"]:
            case 0:
                row["conf"] = 1 - math.dist(
                    (0, 1, 0), tuple(row[["young", "middle", "old"]])
                )
            case 1:
                row["conf"] = 1 - math.dist(
                    (0.5, 0.5, 0.5), tuple(row[["young", "middle", "old"]])
                )
            case 2:
                row["conf"] = 1 - math.dist(
                    (1, 0, 0), tuple(row[["young", "middle", "old"]])
                )
            case 3:
                row["conf"] = 1 - math.dist(
                    (0, 0, 1), tuple(row[["young", "middle", "old"]])
                )
        return row

    result = result.apply(calc_conf, axis=1)
    result["conf"] = result["conf"].round(2)

    return result
