import argparse
import os
import pandas as pd
import json

from utils import (
    gmm_predict,
    preprocess_emails_to_df,
    preprocess_names_to_dict,
    make_dataset,
    email_validation,
)

cluster_lookup = ["medium", "unsure", "young", "old"]


def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def main():
    parser = argparse.ArgumentParser(description="Process a file or an email.")
    parser.add_argument(
        "-f",
        "--file",
        type=validate_file,
        required=False,
        help="Path to a text file with emails.",
    )
    parser.add_argument(
        "-e",
        "--email",
        type=str,
        required=False,
        help="Email address for inference. i.e. 'john.smith@example.com'",
    )

    args = parser.parse_args()

    if not args.file and not args.email:
        parser.error("At least one of --file or --email is required.")

    if args.file is not None and args.email is not None:
        parser.error("Only one of --file or --email can be processed.")

    if args.file is not None:
        emails_df = preprocess_emails_to_df(args.file)
        names_dict = preprocess_names_to_dict("./names_by_birth_year.csv")
        dataset = make_dataset(emails_df, names_dict)
        print("Started prediction ...")
        result = gmm_predict(dataset)
        print("Prediction done.")

        result["age"] = result.apply(lambda row: cluster_lookup[row["cluster"]], axis=1)
        result["score"] = result["conf"]

        list_of_dicts = result[["email", "age", "score"]].to_dict(orient="records")
        json_list = json.dumps(list_of_dicts, indent=2)

        with open("result.json", "w") as f:
            f.write(json_list)

    else:
        email = args.email.lower()
        if email_validation(email) is False:
            print("Email is not valid")
            exit()

        email_df = pd.DataFrame({"email": [email]})
        names_dict = preprocess_names_to_dict("./names_by_birth_year.csv")
        dataset = make_dataset(email_df, names_dict)

        result = gmm_predict(dataset)

        result = {
            "email": result["email"][0],
            "age": cluster_lookup[result["cluster"][0]],
            "score": result["conf"][0],
        }
        print(result)


if __name__ == "__main__":
    main()
