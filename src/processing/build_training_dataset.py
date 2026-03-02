import subprocess
import sys

# Install dependencies inside processing container
subprocess.check_call(
    [sys.executable, "-m", "pip", "install",
     "boto3", "pandas", "sqlalchemy==1.4.46", "pymysql", "pytz", "scikit-learn"]
)

import argparse
import json
import boto3
import pandas as pd
import pytz
from datetime import datetime
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split


# =========================================================
# Expected Schemas
# =========================================================
TRIALS_EXPECTED_COLS = [
    "StudyID", "PatientID", "DrugName",
    "Age", "Gender", "WeightKg", "HeightCm", "BMI",
    "DosageMg", "BiomarkerLevel", "AdverseEventFlag",
    "EnrolledDate"
]

SAFETY_EXPECTED_COLS = [
    "StudyID", "PatientID", "DrugName",
    "EventType", "Severity", "EventDurationDays", "Hospitalized"
]


# =========================================================
# Secrets Manager + DB Engine
# =========================================================
def get_secret(secret_name, region):
    print("🔐 Fetching DB credentials...")
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def build_engine(secret_name, region):
    secret = get_secret(secret_name, region)

    conn_str = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )

    return create_engine(conn_str, pool_pre_ping=True)


# =========================================================
# Cleaning Functions
# =========================================================
def clean_trials(df):

    print("🧹 Cleaning trials data...")

    # Keep only expected columns
    df = df[[c for c in TRIALS_EXPECTED_COLS if c in df.columns]]

    numeric_cols = [
        "Age", "WeightKg", "HeightCm",
        "BMI", "DosageMg",
        "BiomarkerLevel", "AdverseEventFlag"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["StudyID", "PatientID", "DrugName"])
    df = df.drop_duplicates(subset=["StudyID", "PatientID", "DrugName"])

    return df


def clean_safety(df):

    print("🧹 Cleaning safety data...")

    df = df[[c for c in SAFETY_EXPECTED_COLS if c in df.columns]]

    if "EventDurationDays" in df.columns:
        df["EventDurationDays"] = pd.to_numeric(df["EventDurationDays"], errors="coerce")
        df = df[df["EventDurationDays"].fillna(0) <= 365]

    df = df.dropna(subset=["StudyID", "PatientID", "DrugName"])
    df = df.drop_duplicates()

    return df


# =========================================================
# Feature Engineering
# =========================================================
def feature_engineering(df):

    print("🧠 Applying feature engineering...")

    df = df.dropna(subset=["Severity"])
    df["Severity"] = df["Severity"].astype(str).str.title()
    df = df[df["Severity"].isin(["Mild", "Moderate", "Severe"])]

    numeric_cols = ["Age", "BMI", "DosageMg", "BiomarkerLevel"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=numeric_cols)

    df["HighBiomarker"] = (df["BiomarkerLevel"] > 220).astype(int)
    df["HighDosage"] = (df["DosageMg"] > 130).astype(int)

    return df


# =========================================================
# Save Joined Dataset to DB (DMT Layer)
# =========================================================
def save_joined_view(df, engine):

    print("🗄 Saving ML-ready dataset to DB...")

    ist = pytz.timezone("Asia/Kolkata")
    df = df.copy()
    df["load_time"] = datetime.now(ist)

    table_name = "DMT_ML_READY_DATA"
    view_name = "DMT_LATEST_ML_DATA"

    # Replace table each training run (snapshot behavior)
    df.to_sql(
        table_name,
        engine,
        schema="ml_layer",
        if_exists="replace",
        index=False
    )

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE OR REPLACE VIEW ml_layer.{view_name} AS
            SELECT * FROM ml_layer.{table_name};
        """))

    print("✅ DMT table + view updated successfully.")


# =========================================================
# MAIN
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--db_secret_name", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--psa_trials_table", required=True)
    parser.add_argument("--psa_safety_table", required=True)

    args = parser.parse_args()

    print("========================================")
    print(" BUILD TRAINING DATASET STARTED ")
    print("========================================")

    engine = build_engine(args.db_secret_name, args.region)

    # -----------------------------------------------------
    # Load PSA Tables
    # -----------------------------------------------------
    print("📥 Loading PSA tables...")

    trials_df = pd.read_sql(
        f"SELECT * FROM ml_layer.{args.psa_trials_table}",
        engine
    )

    safety_df = pd.read_sql(
        f"SELECT * FROM ml_layer.{args.psa_safety_table}",
        engine
    )

    print("Trials shape:", trials_df.shape)
    print("Safety shape:", safety_df.shape)

    # -----------------------------------------------------
    # Cleaning
    # -----------------------------------------------------
    trials_df = clean_trials(trials_df)
    safety_df = clean_safety(safety_df)

    # -----------------------------------------------------
    # Join
    # -----------------------------------------------------
    print("🔗 Joining trials + safety...")
    df = trials_df.merge(
        safety_df,
        on=["StudyID", "PatientID", "DrugName"],
        how="inner"
    )

    print("After join shape:", df.shape)

    # -----------------------------------------------------
    # Feature Engineering
    # -----------------------------------------------------
    df = feature_engineering(df)

    print("Final dataset shape:", df.shape)
    print("Severity distribution:")
    print(df["Severity"].value_counts())

    # -----------------------------------------------------
    # Save to DB DMT Layer
    # -----------------------------------------------------
    save_joined_view(df, engine)

    # -----------------------------------------------------
    # Train/Test Split
    # -----------------------------------------------------
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["Severity"]
    )

    # Save outputs for training step
    train_df.to_csv("/opt/ml/processing/output/train.csv", index=False)
    test_df.to_csv("/opt/ml/processing/output/test.csv", index=False)

    print("✅ Train/Test datasets saved.")
    print("========================================")
    print(" BUILD TRAINING DATASET COMPLETED ")
    print("========================================")


if __name__ == "__main__":
    main()