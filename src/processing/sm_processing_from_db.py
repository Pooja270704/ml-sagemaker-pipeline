import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

# Install pymysql inside processing container
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--db_table", required=True)   # now pass PSA_VIEW_TRAINING
    args = parser.parse_args()

    print("🔐 Fetching DB credentials...")
    secret = _get_db_secret(args.db_secret_arn, args.region)

    connection = pymysql.connect(
        host=secret["host"],
        user=secret["username"],
        password=secret["password"],
        database=secret["database"],
        port=secret["port"],
        connect_timeout=10,
        autocommit=True,
    )

    try:
        print(f"📥 Reading from {args.db_table}...")
        df = pd.read_sql(f"SELECT * FROM `{args.db_table}`", connection)

        print(f"Rows pulled: {len(df)}")

    finally:
        connection.close()

    # ✅ Add runtime dynamically (DO NOT UPDATE DB)
    pipeline_runtime = datetime.now(timezone.utc)
    df["PipelineRunTimestamp"] = pipeline_runtime

    print(f"⏱ PipelineRunTimestamp added: {pipeline_runtime}")

    # Save for training
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(output_path, index=False)

    print(f"✅ Data saved to: {output_path}")
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()