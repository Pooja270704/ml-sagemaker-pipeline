import boto3
import json
import joblib
import pandas as pd

from io import StringIO
from datetime import datetime
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

from src.config import Defaults
from src.data.dataset_builder import build_dataset_parameterized
from src.models.train import make_preprocessor, split_xy, get_candidate_models
from src.models.evaluate import train_and_select_best

import pytz


# =========================================================
# 🔒 INFRASTRUCTURE CONFIG
# =========================================================

S3_BUCKET = "ml-sagemaker-pipeline-demo"
TRIALS_S3_KEY = "data/raw/clinical_trials.csv"
SAFETY_S3_KEY = "data/raw/clinical_safety_events.json"

SECRET_NAME = "abalone-db-app-secret"
AWS_REGION = "ap-south-1"


# =========================================================
# 🔐 SECRETS MANAGER
# =========================================================

def get_secret(secret_name, region):
    print("🔐 Fetching DB credentials from AWS Secrets Manager...")
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def build_db_connection_string():
    secret = get_secret(SECRET_NAME, AWS_REGION)

    encoded_password = quote_plus(secret["password"])

    return (
        f"mysql+pymysql://{secret['username']}:{encoded_password}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )


# =========================================================
# 🧠 USER INPUT
# =========================================================

def ask_user_inputs():
    print("\n====================================")
    print(" PARAMETERIZED MEDICAL MLOPS RUN ")
    print("====================================\n")

    trials_source = input("👉 Load TRIALS data from? (S3/DB): ").strip().upper()
    trials_format = input("👉 What is TRIALS file format? (csv/json): ").strip().lower()

    safety_source = input("👉 Load SAFETY data from? (S3/DB): ").strip().upper()
    safety_format = input("👉 What is SAFETY file format? (csv/json): ").strip().lower()

    # Validation
    if trials_format != "csv":
        raise ValueError("❌ Trials dataset must be CSV.")

    if safety_format != "json":
        raise ValueError("❌ Safety dataset must be JSON.")

    if trials_source not in ["S3", "DB"]:
        raise ValueError("❌ Trials source must be S3 or DB.")

    if safety_source not in ["S3", "DB"]:
        raise ValueError("❌ Safety source must be S3 or DB.")

    print("✅ Input validation passed.\n")

    return trials_source, trials_format, safety_source, safety_format


# =========================================================
# 📦 S3 UPLOAD HELPERS
# =========================================================

def upload_dataframe_to_s3(df, bucket, key):
    print(f"\n⬆️ Uploading cleaned dataset to S3: s3://{bucket}/{key}")

    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

    print("✅ Dataset uploaded successfully.")


def upload_model_to_s3(model, bucket, key):
    print(f"\n⬆️ Uploading trained model to S3: s3://{bucket}/{key}")

    joblib.dump(model, "best_model.joblib")

    s3 = boto3.client("s3")
    s3.upload_file("best_model.joblib", bucket, key)

    print("✅ Model uploaded successfully.")


# =========================================================
# 🗄️ SAVE TO DB WITH VIEW + LOAD_TIME
# =========================================================

def save_dataframe_to_db_with_view(df, db_connection_string):
    print("\n🗄️ Saving cleaned dataset to DB (IST Time)...")

    engine = create_engine(db_connection_string)

    df_copy = df.copy()

    # ------------------------------------------------
    # 🇮🇳 Indian Standard Time (Asia/Kolkata)
    # ------------------------------------------------
    ist = pytz.timezone("Asia/Kolkata")
    load_time_ist = datetime.now(ist)

    df_copy["load_time"] = load_time_ist

    table_name = "PSA_ML_READY_DATA"
    view_name = "DMT_LATEST_ML_DATA"

    df_copy.to_sql(
        table_name,
        engine,
        if_exists="append",
        index=False
    )

    print(f"✅ Data appended to table: {table_name}")
    print(f"🕒 Load Time (IST): {load_time_ist}")

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

    print(f"✅ View created/updated: {view_name}")


# =========================================================
# 🚀 MAIN
# =========================================================

def main():
    d = Defaults()

    trials_source, trials_format, safety_source, safety_format = ask_user_inputs()

    db_connection_string = None

    if trials_source == "DB" or safety_source == "DB":
        db_connection_string = build_db_connection_string()

    print("\n🚀 Building dataset...")

    df = build_dataset_parameterized(
        trials_source=trials_source,
        trials_format=trials_format,
        safety_source=safety_source,
        safety_format=safety_format,
        s3_bucket=S3_BUCKET,
        trials_s3_key=TRIALS_S3_KEY,
        safety_s3_key=SAFETY_S3_KEY,
        db_connection_string=db_connection_string,
        stg_trials_table=d.stg_trials_table,
        stg_safety_table=d.stg_safety_table,
        psa_trials_table=d.psa_trials_table,
        psa_safety_table=d.psa_safety_table,
    )

    # =====================================================
    # 📊 PREVIEW TRANSFORMED DATA
    # =====================================================

    print("\n====================================")
    print(" DATA TRANSFER & TRANSFORMATION DONE ")
    print("====================================")

    print("Final Dataset Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nSample Data:")
    print(df.head())

    print("\nTarget Distribution:")
    print(df["Severity"].value_counts())

    # Save to DB (if DB involved)
    if db_connection_string:
        save_dataframe_to_db_with_view(df, db_connection_string)

    # Upload cleaned dataset to S3
    upload_dataframe_to_s3(
        df,
        S3_BUCKET,
        "data/processed/final_cleaned_dataset.csv"
    )

    # =====================================================
    # 🤖 MODEL TRAINING
    # =====================================================

    print("\n📊 Starting model training...")

    X_train, X_test, y_train, y_test = split_xy(df)
    preprocessor = make_preprocessor()
    models = get_candidate_models()

    best_model, results = train_and_select_best(
        models,
        preprocessor,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Upload best model
    upload_model_to_s3(
        best_model,
        S3_BUCKET,
        "models/best_model.joblib"
    )

    print("\n🏆 Best model selected and stored.")
    print("✅ Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()