import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "sqlalchemy==1.4.46", "pymysql"])

import argparse
import json
import boto3
import pandas as pd
from sqlalchemy import create_engine, text


# -----------------------------
# Get Secret
# -----------------------------
def get_secret(secret_id, region):
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_id)
    return json.loads(response["SecretString"])


def get_engine(secret):
    return create_engine(
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )


# -----------------------------
# CSV RAW → PSA STRUCTURED
# -----------------------------
def csv_raw_to_psa(engine, stg_table, psa_table):

    # Read raw CSV text
    df_raw = pd.read_sql(f"SELECT raw_csv FROM {stg_table}", engine)

    # Split CSV string into columns
    df = df_raw["raw_csv"].str.split(",", expand=True)

    print("Detected column count:", df.shape[1])

    # ---- Adjust based on actual column count ----
    if df.shape[1] == 13:
        df.columns = [
            "StudyID",
            "PatientID",
            "DrugName",
            "Age",
            "Gender",
            "WeightKg",
            "HeightCm",
            "BMI",
            "DosageMg",
            "BiomarkerLevel",
            "ResponseScore",
            "AdverseEventFlag",
            "EventDate"
        ]
    else:
        raise ValueError(f"Unexpected column count: {df.shape[1]}")

    # Replace PSA table
    df.to_sql(psa_table, engine, if_exists="replace", index=False)

    print("✅ PSA Clinical Trial table created successfully")


# -----------------------------
# JSON RAW → PSA STRUCTURED
# -----------------------------
def json_raw_to_psa(engine, stg_table, psa_table):
    df_raw = pd.read_sql(f"SELECT raw_json FROM {stg_table}", engine)

    records = df_raw["raw_json"].apply(json.loads)
    df = pd.DataFrame(records.tolist())

    df.to_sql(psa_table, engine, if_exists="replace", index=False)

    print("PSA Clinical Safety table created")


# -----------------------------
# MAIN
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--stg_trials_table", required=True)
    parser.add_argument("--stg_safety_table", required=True)
    parser.add_argument("--psa_trials_table", required=True)
    parser.add_argument("--psa_safety_table", required=True)

    args = parser.parse_args()

    secret = get_secret(args.db_secret_arn, args.region)
    engine = get_engine(secret)

    csv_raw_to_psa(engine, args.stg_trials_table, args.psa_trials_table)
    json_raw_to_psa(engine, args.stg_safety_table, args.psa_safety_table)

    print("STG → PSA completed successfully")


if __name__ == "__main__":
    main()