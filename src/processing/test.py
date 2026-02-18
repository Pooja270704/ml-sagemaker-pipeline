import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

subprocess.check_call([sys.executable, "-m", "pip", "install", "pymysql"])

import boto3
import pandas as pd
import pymysql


def _get_db_secret(secret_arn: str, region: str) -> dict:
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    secret = json.loads(response["SecretString"])

    return {
        "host": secret.get("host"),
        "username": secret.get("username"),
        "password": secret.get("password"),
        "database": secret.get("dbname"),
        "port": int(secret.get("port", 3306)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", default="ap-south-1")
    parser.add_argument("--db_table", default="clinical_training_clean")
    args = parser.parse_args()

    secret = _get_db_secret(args.db_secret_arn, args.region)

    connection = pymysql.connect(
        host=secret["host"],
        user=secret["username"],
        password=secret["password"],
        database=secret["database"],
        port=secret["port"],
        connect_timeout=10,
    )

    try:
        cursor = connection.cursor()

        # 🔹 1️⃣ Add column if not exists
        cursor.execute(f"""
        ALTER TABLE `{args.db_table}`
        ADD COLUMN IF NOT EXISTS PipelineRunTimestamp DATETIME;
        """)
        connection.commit()

        # 🔹 2️⃣ Generate timestamp
        current_time = datetime.utcnow()

        # 🔹 3️⃣ Update entire clean table with current run timestamp
        cursor.execute(f"""
        UPDATE `{args.db_table}`
        SET PipelineRunTimestamp = %s;
        """, (current_time,))
        connection.commit()

        print(f"Timestamp updated in {args.db_table}: {current_time}")

        # 🔹 4️⃣ Pull updated table for training
        df = pd.read_sql(f"SELECT * FROM `{args.db_table}`", connection)

    finally:
        connection.close()

    # 🔹 5️⃣ Save to S3 for training step
    output_dir = "/opt/ml/processing/output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_data.csv")
    df.to_csv(output_path, index=False)

    print(f"Training data saved to: {output_path}")


if __name__ == "__main__":
    main()