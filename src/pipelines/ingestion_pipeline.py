import subprocess
import sys

# Install required libraries inside processing container
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
from io import StringIO


# =========================================================
# ARGUMENTS
# =========================================================
parser = argparse.ArgumentParser()

parser.add_argument("--region", required=True)
parser.add_argument("--db_secret_name", required=True)
parser.add_argument("--source", required=True)        # DB or S3
parser.add_argument("--file_format", required=True)   # csv or json
parser.add_argument("--raw_bucket", required=False)
parser.add_argument("--file_key", required=False)

args = parser.parse_args()

REGION = args.region
SECRET_NAME = args.db_secret_name
SOURCE = args.source.upper()
FILE_FORMAT = args.file_format.lower()

IST = pytz.timezone("Asia/Kolkata")


# =========================================================
# DB CONNECTION
# =========================================================
def get_secret(secret_name, region):
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])


def build_engine():
    secret = get_secret(SECRET_NAME, REGION)

    conn_str = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    return create_engine(conn_str, pool_pre_ping=True)


engine = build_engine()


# =========================================================
# LOAD DATA
# =========================================================
def load_from_db():
    if FILE_FORMAT == "csv":
        return pd.read_sql("SELECT raw_csv FROM source_layer.STG_clinical_trials", engine)
    else:
        return pd.read_sql("SELECT raw_json FROM source_layer.STG_clinical_safety_events", engine)


def load_from_s3():
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=args.raw_bucket, Key=args.file_key)
    content = obj["Body"].read().decode("utf-8")

    if FILE_FORMAT == "csv":
        return pd.DataFrame({"raw_csv": content.splitlines()})
    else:
        return pd.DataFrame({"raw_json": content.splitlines()})


# =========================================================
# STG INSERT (Append Only)
# =========================================================
def insert_to_stg(df):

    stg_table = "STG_clinical_trials" if FILE_FORMAT == "csv" else "STG_clinical_safety_events"
    column_name = "raw_csv" if FILE_FORMAT == "csv" else "raw_json"

    df["created_at"] = datetime.now(IST)

    df.to_sql(
        stg_table,
        engine,
        schema="ml_layer",
        if_exists="append",
        index=False
    )

    print(f"✅ Inserted into ml_layer.{stg_table}")


# =========================================================
# STG -> PSA (Business Key Dedup)
# =========================================================
def transform_to_psa():

    if FILE_FORMAT == "csv":
        stg_table = "ml_layer.STG_clinical_trials"
        psa_table = "PSA_clinical_trials"
        view_name = "PSA_clinical_trials_view"

        dedup_condition = """
            p.StudyID = s.StudyID
            AND p.PatientID = s.PatientID
            AND p.DrugName = s.DrugName
        """

    else:
        stg_table = "ml_layer.STG_clinical_safety_events"
        psa_table = "PSA_clinical_safety_events"
        view_name = "PSA_clinical_safety_events_view"

        dedup_condition = """
            p.StudyID = s.StudyID
            AND p.PatientID = s.PatientID
            AND p.DrugName = s.DrugName
            AND p.EventType = s.EventType
        """

    with engine.begin() as conn:

        # Create PSA table if not exists (structured copy)
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS ml_layer.{psa_table}
            LIKE {stg_table};
        """))

        # Insert only new records
        conn.execute(text(f"""
            INSERT INTO ml_layer.{psa_table}
            SELECT s.*
            FROM {stg_table} s
            WHERE NOT EXISTS (
                SELECT 1 FROM ml_layer.{psa_table} p
                WHERE {dedup_condition}
            );
        """))

        # Create / Replace View
        conn.execute(text(f"""
            CREATE OR REPLACE VIEW ml_layer.{view_name} AS
            SELECT *,
                   CONVERT_TZ(NOW(), 'UTC', 'Asia/Kolkata') AS load_time
            FROM ml_layer.{psa_table};
        """))

    print(f"✅ PSA updated: ml_layer.{psa_table}")
    print(f"✅ View updated: ml_layer.{view_name}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("===================================")
    print(" INGESTION PIPELINE STARTED ")
    print("===================================")

    if SOURCE == "DB":
        df = load_from_db()
    elif SOURCE == "S3":
        df = load_from_s3()
    else:
        raise ValueError("Source must be DB or S3")

    if df.empty:
        raise ValueError("No data found in source")

    insert_to_stg(df)
    transform_to_psa()

    print("===================================")
    print(" INGESTION COMPLETED SUCCESSFULLY ")
    print("===================================")