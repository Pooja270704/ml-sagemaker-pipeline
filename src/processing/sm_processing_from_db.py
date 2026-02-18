import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Install pymysql inside the processing container
subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql"])

import boto3
import pandas as pd
import pymysql


def _get_db_secret(secret_arn: str, region: str) -> dict:
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(response["SecretString"])

    return {
        "host": secret.get("host") or secret.get("hostname") or secret.get("endpoint"),
        "username": secret.get("username"),
        "password": secret.get("password"),
        "database": secret.get("dbname") or secret.get("database") or secret.get("dbName"),
        "port": int(secret.get("port", 3306)),
    }


def _ensure_timestamp_column(cursor, db_name: str, table_name: str, column_name: str):
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM information_schema.columns
        WHERE table_schema = %s
          AND table_name = %s
          AND column_name = %s
        """,
        (db_name, table_name, column_name),
    )
    exists = cursor.fetchone()[0] > 0

    if not exists:
        cursor.execute(
            f"ALTER TABLE `{table_name}` ADD COLUMN `{column_name}` DATETIME"
        )
        print(f"Added column `{column_name}` to `{table_name}`")
    else:
        print(f"Column `{column_name}` already exists in `{table_name}`")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--db_table", default="clinical_training_clean")
    args = parser.parse_args()

    secret = _get_db_secret(args.db_secret_arn, args.region)
    if not secret["host"] or not secret["username"] or not secret["password"] or not secret["database"]:
        raise ValueError("DB secret is missing required fields (host/username/password/database).")

    connection = pymysql.connect(
        host=secret["host"],
        user=secret["username"],
        password=secret["password"],
        database=secret["database"],
        port=secret["port"],
        connect_timeout=10,
        autocommit=True,   # Important: apply ALTER/UPDATE immediately
    )

    table = args.db_table
    column = "PipelineRunTimestamp"
    run_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    try:
        with connection.cursor() as cursor:
            # 1) Ensure column exists
            _ensure_timestamp_column(cursor, secret["database"], table, column)

            # 2) Update timestamp for this run (updates all rows)
            cursor.execute(
                f"UPDATE `{table}` SET `{column}` = %s",
                (run_time,),
            )
            print(f"Updated `{table}.{column}` to {run_time}")

        # 3) Export data for training
        df = pd.read_sql(f"SELECT * FROM `{table}`", connection)
        print(f"Rows pulled from DB table {table}: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")

    finally:
        connection.close()

    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")


if __name__ == "__main__":
    main()