import subprocess
import sys


subprocess.check_call([sys.executable, "-m", "pip", "install","sqlalchemy==1.4.46", "pymysql"])


import argparse
import boto3
import json
import pandas as pd
from sqlalchemy import create_engine, text


# --------------------------------------------------
# 1️⃣ Get DB credentials from Secrets Manager
# --------------------------------------------------

def get_secret(secret_arn, region):
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


# --------------------------------------------------
# 2️⃣ Create DB Engine
# --------------------------------------------------

def get_engine(secret):
    connection_string = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )

    return create_engine(
        connection_string,
        pool_pre_ping=True
    )


# --------------------------------------------------
# 3️⃣ Create STG Tables (RAW format)
# --------------------------------------------------

def create_stg_tables(engine, stg_trials_table, stg_safety_table):
    with engine.begin() as conn:

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {stg_trials_table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                raw_csv LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {stg_safety_table} (
                id INT AUTO_INCREMENT PRIMARY KEY,
                raw_json LONGTEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

    print("✅ STG tables ensured")


# --------------------------------------------------
# 4️⃣ Truncate Tables
# --------------------------------------------------

def truncate_tables(engine, stg_trials_table, stg_safety_table):
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE TABLE {stg_trials_table}"))
        conn.execute(text(f"TRUNCATE TABLE {stg_safety_table}"))

    print("🗑️ STG tables truncated")


# --------------------------------------------------
# 5️⃣ Load CSV RAW from S3 → STG
# --------------------------------------------------

def load_csv_raw_to_stg(engine, bucket, key, table_name):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    raw_content = obj["Body"].read().decode("utf-8")
    lines = raw_content.strip().split("\n")

    data_lines = lines[1:]  # skip header

    records = [{"raw_csv": line} for line in data_lines if line.strip()]
    df = pd.DataFrame(records)

    with engine.begin() as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="append",
            index=False
        )

    print(f"📄 Inserted {len(df)} raw CSV rows into {table_name}")


# --------------------------------------------------
# 6️⃣ Load JSON RAW from S3 → STG
# --------------------------------------------------

def load_json_raw_to_stg(engine, bucket, key, table_name):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)

    raw_content = obj["Body"].read().decode("utf-8")

    try:
        data = json.loads(raw_content)

        if isinstance(data, list):
            records = [{"raw_json": json.dumps(row)} for row in data]
        else:
            records = [{"raw_json": json.dumps(data)}]

    except json.JSONDecodeError:
        lines = raw_content.strip().split("\n")
        records = [{"raw_json": line} for line in lines if line.strip()]

    df = pd.DataFrame(records)

    with engine.begin() as conn:
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists="append",
            index=False
        )

    print(f"📦 Inserted {len(df)} raw JSON rows into {table_name}")


# --------------------------------------------------
# 7️⃣ MAIN
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--raw_bucket", required=True)
    parser.add_argument("--trials_csv_key", required=True)
    parser.add_argument("--safety_json_key", required=True)
    parser.add_argument("--stg_trials_table", required=True)
    parser.add_argument("--stg_safety_table", required=True)

    args = parser.parse_args()

    print("🔐 Fetching DB credentials...")
    secret = get_secret(args.db_secret_arn, args.region)

    print("🔗 Connecting to DB...")
    engine = get_engine(secret)

    print("📦 Creating STG tables if not exist...")
    create_stg_tables(engine, args.stg_trials_table, args.stg_safety_table)

    print("🗑️ Truncating STG tables...")
    truncate_tables(engine, args.stg_trials_table, args.stg_safety_table)

    print("⬇️ Loading CSV raw from S3...")
    load_csv_raw_to_stg(
        engine,
        args.raw_bucket,
        args.trials_csv_key,
        args.stg_trials_table
    )

    print("⬇️ Loading JSON raw from S3...")
    load_json_raw_to_stg(
        engine,
        args.raw_bucket,
        args.safety_json_key,
        args.stg_safety_table
    )

    print("🎉 RAW → STG completed successfully!")


if __name__ == "__main__":
    main()