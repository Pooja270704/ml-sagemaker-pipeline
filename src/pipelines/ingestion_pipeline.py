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
from datetime import datetime
from sqlalchemy import create_engine, text
from io import StringIO

# ==========================
# ARGUMENTS
# ==========================
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

# ==========================
# SECRETS + DB ENGINE
# ==========================
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

# ==========================
# LOAD RAW DATA
# ==========================
def load_from_db():

    if FILE_FORMAT == "csv":
        query = "SELECT raw_csv FROM source_layer.STG_clinical_trials"
        df_raw = pd.read_sql(query, engine)

        if df_raw.empty:
            raise ValueError("No CSV data found in source_layer.")

        combined = "\n".join(df_raw["raw_csv"].tolist())

        header = (
            "StudyID,PatientID,DrugName,Age,Gender,WeightKg,HeightCm,"
            "BMI,DosageMg,BiomarkerLevel,AdverseEventFlag,EnrolledDate"
        )

        final_csv = header + "\n" + combined
        return pd.read_csv(StringIO(final_csv))

    elif FILE_FORMAT == "json":
        query = "SELECT raw_json FROM source_layer.STG_clinical_safety_events"
        df_raw = pd.read_sql(query, engine)

        if df_raw.empty:
            raise ValueError("No JSON data found in source_layer.")

        combined = "\n".join(df_raw["raw_json"].tolist())
        return pd.read_json(StringIO(combined), lines=True)

    else:
        raise ValueError("Unsupported file format.")

def load_from_s3():

    s3 = boto3.client("s3", region_name=REGION)

    bucket = "ml-sagemaker-pipeline-demo"

    if FILE_FORMAT == "csv":
        key = "data/raw/clinical_trials.csv"
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        return pd.read_csv(StringIO(content))

    elif FILE_FORMAT == "json":
        key = "data/raw/clinical_safety_events.json"
        obj = s3.get_object(Bucket=bucket, Key=key)
        content = obj["Body"].read().decode("utf-8")
        return pd.read_json(StringIO(content))

# ==========================
# INSERT INTO STG (AUTO-CREATE)
# ==========================
def insert_to_stg(df):

    if FILE_FORMAT == "csv":
        table_name = "STG_clinical_trials"
    else:
        table_name = "STG_clinical_safety_events"

    df["created_at"] = datetime.utcnow()

    df.to_sql(
        table_name,
        engine,
        schema="ml_layer",
        if_exists="append",
        index=False
    )

# ==========================
# TRANSFORM TO PSA (DEDUP SAFE)
# ==========================
def transform_to_psa():

    if FILE_FORMAT == "csv":
        stg_table = "STG_clinical_trials"
        psa_table = "PSA_clinical_trials"
        view_name = "PSA_clinical_trials_view"

        dedup_condition = """
            p.StudyID = s.StudyID
            AND p.PatientID = s.PatientID
        """

    else:
        stg_table = "STG_clinical_safety_events"
        psa_table = "PSA_clinical_safety_events"
        view_name = "PSA_clinical_safety_events_view"

        dedup_condition = """
            p.StudyID = s.StudyID
            AND p.PatientID = s.PatientID
            AND p.EventType = s.EventType
        """

    with engine.begin() as conn:

        # 1️⃣ Create PSA table if not exists
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS ml_layer.{psa_table}
            LIKE ml_layer.{stg_table};
        """))

        # 2️⃣ Insert only new rows (dedup)
        conn.execute(text(f"""
            INSERT INTO ml_layer.{psa_table}
            SELECT s.* FROM ml_layer.{stg_table} s
            WHERE NOT EXISTS (
                SELECT 1 FROM ml_layer.{psa_table} p
                WHERE {dedup_condition}
            );
        """))

        # 3️⃣ Create or Replace PSA View with Load Time
        conn.execute(text(f"""
            CREATE OR REPLACE VIEW ml_layer.{view_name} AS
            SELECT 
                p.*,
                CURRENT_TIMESTAMP AS load_time
            FROM ml_layer.{psa_table} p;
        """))

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    print("=== Starting Ingestion Pipeline ===")

    if SOURCE == "DB":
        df = load_from_db()

    elif SOURCE == "S3":
        df = load_from_s3()

    else:
        raise ValueError("Invalid source. Use DB or S3.")

    if df.empty:
        raise ValueError("No data found after loading.")

    insert_to_stg(df)
    transform_to_psa()

    print("=== Ingestion Completed Successfully ===")