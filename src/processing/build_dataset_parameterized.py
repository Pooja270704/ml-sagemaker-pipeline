import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install",
     "boto3", "pandas", "sqlalchemy==1.4.46", "pymysql", "pytz"]
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
# Expected schema (we will filter to these only)
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


def build_db_engine(secret_name, region):
    secret = get_secret(secret_name, region)

    # NOTE: if your password has special chars, this can break.
    # In production, URL-encode it. For now leaving as-is because your current code does this.
    conn_str = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    return create_engine(conn_str, pool_pre_ping=True)


# =========================================================
# S3 Loaders
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
# DB Helpers
# =========================================================
def read_db_table(engine, table_name):
    print(f"📥 Loading table from DB: {table_name}")
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def drop_table_if_exists(engine, table_name):
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
    print(f"🧹 Dropped table if existed: {table_name}")


def write_replace_table(engine, df, table_name):
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"✅ Replaced table: {table_name} (rows={len(df)})")


# =========================================================
# STG (raw) -> PSA (structured)
# =========================================================
def stg_csv_to_psa(engine, stg_table, psa_table, expected_cols):
    """
    STG is assumed to have: raw_csv LONGTEXT
    Each row is a single CSV line (without header).
    """
    raw = read_db_table(engine, stg_table)

    if "raw_csv" not in raw.columns:
        # Already structured (someone loaded structured into STG)
        print(f"⚠ {stg_table} does not have raw_csv; treating as structured.")
        df = raw.copy()
    else:
        lines = raw["raw_csv"].dropna().astype(str)
        # Split by comma into columns
        split_df = lines.str.split(",", expand=True)

        # If the count doesn't match, we must stop (otherwise wrong mapping)
        if split_df.shape[1] != len(expected_cols):
            raise ValueError(
                f"STG CSV column count mismatch in {stg_table}. "
                f"Expected {len(expected_cols)} fields but got {split_df.shape[1]}."
            )

        split_df.columns = expected_cols
        df = split_df

    # Filter strictly to expected columns
    df = df[[c for c in expected_cols if c in df.columns]]

    # Replace PSA
    drop_table_if_exists(engine, psa_table)
    write_replace_table(engine, df, psa_table)
    return df


def stg_json_to_psa(engine, stg_table, psa_table, expected_cols):
    """
    STG is assumed to have: raw_json LONGTEXT
    Each row is a JSON object string (or JSONL line).
    """
    raw = read_db_table(engine, stg_table)

    if "raw_json" not in raw.columns:
        print(f"⚠ {stg_table} does not have raw_json; treating as structured.")
        df = raw.copy()
    else:
        records = raw["raw_json"].dropna().astype(str).apply(json.loads)
        df = pd.DataFrame(records.tolist())

    # Filter strictly to expected columns (ignore old junk columns)
    for c in expected_cols:
        if c not in df.columns:
            df[c] = None
    df = df[expected_cols]

    drop_table_if_exists(engine, psa_table)
    write_replace_table(engine, df, psa_table)
    return df


# =========================================================
# Cleaning
# =========================================================
def normalize_trials_columns(df):
    """
    Handles old column names (EventDate vs EnrolledDate etc.)
    """
    if "EventDate" in df.columns and "EnrolledDate" not in df.columns:
        df = df.rename(columns={"EventDate": "EnrolledDate"})
    return df


def clean_trials(df):
    df = normalize_trials_columns(df)

    print("Columns available in trials_df:", df.columns.tolist())

    numeric_cols = ["Age", "WeightKg", "HeightCm", "BMI", "DosageMg", "BiomarkerLevel", "AdverseEventFlag"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "EnrolledDate" in df.columns:
        df["EnrolledDate"] = pd.to_datetime(df["EnrolledDate"], errors="coerce")
        min_date = df["EnrolledDate"].dropna().min()
        if pd.notna(min_date):
            df["EnrollmentDays"] = (df["EnrolledDate"] - min_date).dt.days
        else:
            df["EnrollmentDays"] = 0
    else:
        df["EnrollmentDays"] = 0

    if "BMI" in df.columns:
        df = df[df["BMI"].between(15, 50)]

    if "BiomarkerLevel" in df.columns:
        df = df.dropna(subset=["BiomarkerLevel"])

    # Ensure keys exist
    df = df.dropna(subset=["StudyID", "PatientID", "DrugName"])

    df = df.drop_duplicates(subset=["StudyID", "PatientID", "DrugName"], keep="first")
    return df


def clean_safety(df):
    # normalize
    if "Severity" in df.columns:
        df["Severity"] = df["Severity"].astype(str).str.title()

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

    # CRITICAL: remove NaN target BEFORE training
    df["Severity"] = df["Severity"].astype(str).str.title()
    df = df.dropna(subset=["Severity"])
    df = df[df["Severity"].isin(["Mild", "Moderate", "Severe"])]

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")
    df["DosageMg"] = pd.to_numeric(df["DosageMg"], errors="coerce")
    df["BiomarkerLevel"] = pd.to_numeric(df["BiomarkerLevel"], errors="coerce")

    df = df.dropna(subset=["Age", "BMI", "DosageMg", "BiomarkerLevel"])

    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 45, 60, 75, 100],
        labels=["Young", "Middle", "Senior", "Elderly"]
    )

    df["HighBiomarker"] = (df["BiomarkerLevel"] > 220).astype(int)
    df["HighDosage"] = (df["DosageMg"] > 130).astype(int)

    critical_events = ["Cardiac Arrest", "Respiratory Failure", "Seizure"]
    df["CriticalEvent"] = df["EventType"].isin(critical_events).astype(int)

    df["BMI_Category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )

    return df


# =========================================================
# Save ML-ready to DB (optional)
# =========================================================
def save_to_db(df, engine, table_name, view_name):
    print("🗄 Saving ML-ready dataset to DB with IST load_time...")

    ist = pytz.timezone("Asia/Kolkata")
    df = df.copy()
    df["load_time"] = datetime.now(ist)

    # Replace every run (you asked table should not persist across runs)
    drop_table_if_exists(engine, table_name)
    write_replace_table(engine, df, table_name)

    view_sql = f"""
        CREATE OR REPLACE VIEW {view_name} AS
        SELECT *
        FROM {table_name};
    """
    with engine.begin() as conn:
        conn.execute(text(view_sql))

    print(f"✅ View created/updated: {view_name}")


# =========================================================
# MAIN
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

    # Validate formats (your business rule)
    if args.trials_format.lower() != "csv":
        raise ValueError("Trials must be csv")
    if args.safety_format.lower() != "json":
        raise ValueError("Safety must be json")

    engine = None
    if args.trials_source.upper() == "DB" or args.safety_source.upper() == "DB":
        engine = build_db_engine(args.db_secret_arn, args.region)

    # --------------------------
    # Load Trials (S3 or DB->STG->PSA)
    # --------------------------
    if args.trials_source.upper() == "S3":
        trials_df = load_csv_from_s3(args.raw_bucket, args.trials_key)
        # filter schema to avoid old junk cols
        for c in TRIALS_EXPECTED_COLS:
            if c not in trials_df.columns:
                trials_df[c] = None
        trials_df = trials_df[TRIALS_EXPECTED_COLS]
    else:
        # DB: STG(raw) -> PSA(structured) -> use PSA
        if engine is None:
            raise RuntimeError("DB selected but engine not created")
        trials_df = stg_csv_to_psa(engine, args.stg_trials_table, args.psa_trials_table, TRIALS_EXPECTED_COLS)
        trials_df = read_db_table(engine, args.psa_trials_table)

    # --------------------------
    # Load Safety (S3 or DB->STG->PSA)
    # --------------------------
    if args.safety_source.upper() == "S3":
        safety_df = load_json_from_s3(args.raw_bucket, args.safety_key)
        for c in SAFETY_EXPECTED_COLS:
            if c not in safety_df.columns:
                safety_df[c] = None
        safety_df = safety_df[SAFETY_EXPECTED_COLS]
    else:
        if engine is None:
            raise RuntimeError("DB selected but engine not created")
        safety_df = stg_json_to_psa(engine, args.stg_safety_table, args.psa_safety_table, SAFETY_EXPECTED_COLS)
        safety_df = read_db_table(engine, args.psa_safety_table)

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
    # Save to DB (if DB was involved)
    # --------------------------
    if engine is not None:
        save_to_db(df, engine, args.ml_ready_table, args.ml_ready_view)

    
    # Split dataset properly
    train_df, test_df = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        stratify=df["Severity"]
    )

    # Save train set
    train_path = "/opt/ml/processing/output/train.csv"
    train_df.to_csv(train_path, index=False)

    # Save test set
    test_path = "/opt/ml/processing/output/test.csv"
    test_df.to_csv(test_path, index=False)

    print(f"✅ Train dataset saved to {train_path}")
    print(f"✅ Test dataset saved to {test_path}")
    print("======================================")
    print(" BUILD DATASET COMPLETED ")
    print("======================================")


if __name__ == "__main__":
    main()