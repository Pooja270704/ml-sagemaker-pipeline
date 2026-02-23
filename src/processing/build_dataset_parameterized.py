import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install", "boto3", "pandas", "sqlalchemy==1.4.46", "pymysql", "pytz"])






import argparse
import json
import boto3
import pandas as pd
import numpy as np
import pytz

from datetime import datetime
from sqlalchemy import create_engine, text


# =========================================================
# 🔐 Secrets Manager
# =========================================================

def get_secret(secret_name, region):
    print("🔐 Fetching DB credentials...")
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def build_db_engine(secret_name, region):
    secret = get_secret(secret_name, region)
    connection_string = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    return create_engine(connection_string)


# =========================================================
# 📥 Load from S3
# =========================================================

def load_csv_from_s3(bucket, key):
    print(f"📥 Loading CSV from S3: s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"])


def load_json_from_s3(bucket, key):
    print(f"📥 Loading JSON from S3: s3://{bucket}/{key}")
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_json(obj["Body"], lines=True)


# =========================================================
# 📥 Load from DB (STG or PSA)
# =========================================================

def load_table_from_db(engine, table_name):
    print(f"📥 Loading table from DB: {table_name}")
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


# =========================================================
# 🧹 Cleaning
# =========================================================

def clean_trials(df):
    df["EnrolledDate"] = pd.to_datetime(df["EnrolledDate"], errors="coerce")
    df = df[df["BMI"].between(15, 50)]
    df = df.dropna(subset=["BiomarkerLevel"])
    df = df.drop_duplicates()
    return df


def clean_safety(df):
    df["Severity"] = df["Severity"].str.title()
    df = df[df["EventDurationDays"] <= 365]
    df = df.drop_duplicates()
    return df


# =========================================================
# 🧠 Feature Engineering
# =========================================================

def feature_engineering(df):
    print("🧠 Applying feature engineering...")

    df = df.dropna(subset=["Severity"])

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 45, 60, 75, 100],
        labels=["Young", "Middle", "Senior", "Elderly"]
    )

    df["HighBiomarker"] = (df["BiomarkerLevel"] > 220).astype(int)
    df["HighDosage"] = (df["DosageMg"] > 130).astype(int)

    critical_events = [
        "Cardiac Arrest",
        "Respiratory Failure",
        "Seizure"
    ]
    df["CriticalEvent"] = df["EventType"].isin(critical_events).astype(int)

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    return df


# =========================================================
# 🗄 Save ML-ready to DB (if DB involved)
# =========================================================

def save_to_db(df, engine, table_name, view_name):
    print("🗄 Saving ML-ready dataset to DB with IST load_time...")

    ist = pytz.timezone("Asia/Kolkata")
    df["load_time"] = datetime.now(ist)

    df.to_sql(table_name, engine, if_exists="replace", index=False)

    view_sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT *
        FROM {table_name}
        WHERE load_time = (
            SELECT MAX(load_time)
            FROM {table_name}
        );
    """

    with engine.begin() as conn:
        conn.execute(text(view_sql))

    print("✅ DB table + view updated.")


# =========================================================
# 🚀 MAIN
# =========================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--db_secret_arn")
    parser.add_argument("--region")
    parser.add_argument("--raw_bucket")
    parser.add_argument("--trials_key")
    parser.add_argument("--safety_key")

    parser.add_argument("--trials_source")
    parser.add_argument("--trials_format")
    parser.add_argument("--safety_source")
    parser.add_argument("--safety_format")

    parser.add_argument("--stg_trials_table")
    parser.add_argument("--stg_safety_table")
    parser.add_argument("--psa_trials_table")
    parser.add_argument("--psa_safety_table")

    parser.add_argument("--ml_ready_table")
    parser.add_argument("--ml_ready_view")

    args = parser.parse_args()

    print("======================================")
    print(" BUILD DATASET PARAMETERIZED STARTED ")
    print("======================================")

    engine = None
    if args.trials_source == "DB" or args.safety_source == "DB":
        engine = build_db_engine(args.db_secret_arn, args.region)

    # --------------------------
    # Load Trials
    # --------------------------
    if args.trials_source == "S3":
        trials_df = load_csv_from_s3(args.raw_bucket, args.trials_key)
    else:
        trials_df = load_table_from_db(engine, args.stg_trials_table)

    # --------------------------
    # Load Safety
    # --------------------------
    if args.safety_source == "S3":
        safety_df = load_json_from_s3(args.raw_bucket, args.safety_key)
    else:
        safety_df = load_table_from_db(engine, args.stg_safety_table)

    print("Trials Shape:", trials_df.shape)
    print("Safety Shape:", safety_df.shape)

    # --------------------------
    # Cleaning
    # --------------------------
    trials_df = clean_trials(trials_df)
    safety_df = clean_safety(safety_df)

    # --------------------------
    # Join
    # --------------------------
    print("🔗 Joining datasets...")
    df = trials_df.merge(
        safety_df,
        on=["StudyID", "PatientID", "DrugName"],
        how="inner"
    )

    print("After Join Shape:", df.shape)

    # --------------------------
    # Feature Engineering
    # --------------------------
    df = feature_engineering(df)

    print("Final Dataset Shape:", df.shape)
    print("Severity Distribution:")
    print(df["Severity"].value_counts())

    # --------------------------
    # Save to DB (if used)
    # --------------------------
    if engine:
        save_to_db(df, engine, args.ml_ready_table, args.ml_ready_view)

    # --------------------------
    # Save for SageMaker Training
    # --------------------------
    #
    output_path = "/opt/ml/processing/output/train.csv"
    #output_path = "local_train.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Dataset saved to {output_path}")
    print("======================================")
    print(" BUILD DATASET COMPLETED ")
    print("======================================")


if __name__ == "__main__":
    main()