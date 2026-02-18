import boto3
import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine

# ----------------------------------
# 1️⃣ Get DB credentials
# ----------------------------------
def get_secret():
    client = boto3.client("secretsmanager", region_name="ap-south-1")
    response = client.get_secret_value(
        SecretId="abalone-db-app-secret"
    )
    return json.loads(response["SecretString"])


# ----------------------------------
# 2️⃣ Create DB connection
# ----------------------------------
def create_connection(secret):
    connection_string = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    engine = create_engine(connection_string)
    return engine


# ----------------------------------
# 3️⃣ Main Loader
# ----------------------------------
def main():

    print("Fetching DB credentials...")
    secret = get_secret()

    print("Connecting to database...")
    engine = create_connection(secret)

    # Use raw connection for insert
    connection = engine.raw_connection()
    cursor = connection.cursor()

    print("Truncating table...")
    cursor.execute("TRUNCATE TABLE clinical_safety_events_raw;")

    print("Reading JSON file...")
    df = pd.read_json(
        "data/raw/clinical_safety_events.json",
        lines=True
    )

    print("Inserting raw JSON records...")

    for _, row in df.iterrows():
        json_string = json.dumps(row.to_dict())
        cursor.execute(
            "INSERT INTO clinical_safety_events_raw (raw_json) VALUES (%s)",
            (json_string,)
        )

    connection.commit()
    cursor.close()
    connection.close()

    print("Raw JSON data loaded successfully!")
    print("Total rows inserted:", len(df))


if __name__ == "__main__":
    main()