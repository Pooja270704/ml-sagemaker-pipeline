import json
import boto3
import pandas as pd
from sqlalchemy import create_engine

# -----------------------------
# STEP 1: Get DB credentials from Secrets Manager
# -----------------------------

def get_db_credentials():
    client = boto3.client("secretsmanager", region_name="ap-south-1")
    response = client.get_secret_value(
        SecretId="abalone-db-app-secret"
    )
    secret = json.loads(response["SecretString"])
    return secret


# -----------------------------
# STEP 2: Connect to MySQL
# -----------------------------

def create_db_engine(secret):
    connection_string = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    engine = create_engine(connection_string)
    return engine


# -----------------------------
# STEP 3: Load Raw Tables
# -----------------------------

def load_data(engine):
    trials_df = pd.read_sql("SELECT * FROM clinical_trials", engine)
    safety_df = pd.read_sql("SELECT * FROM clinical_safety_events", engine)
    return trials_df, safety_df


# -----------------------------
# STEP 4: Join + Clean
# -----------------------------

def transform_data(trials_df, safety_df):

    # Join
    merged_df = pd.merge(
        trials_df,
        safety_df,
        on=["StudyID", "PatientID", "DrugName"],
        how="left"
    )

    # Remove null biomarker
    merged_df = merged_df[merged_df["BiomarkerLevel"].notna()]

    # Remove extreme BMI
    merged_df = merged_df[
        (merged_df["BMI"] >= 15) & (merged_df["BMI"] <= 45)
    ]

    # Remove duplicates
    merged_df = merged_df.drop_duplicates(
        subset=["StudyID", "PatientID"]
    )

    return merged_df


# -----------------------------
# STEP 5: Write Clean Table
# -----------------------------

def save_transformed_data(engine, df):
    df.to_sql(
        "clinical_training_clean",
        con=engine,
        if_exists="replace",
        index=False
    )


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("Fetching DB credentials...")
    secret = get_db_credentials()

    print("Connecting to database...")
    engine = create_db_engine(secret)

    print("Loading raw tables...")
    trials_df, safety_df = load_data(engine)

    print("Transforming data...")
    clean_df = transform_data(trials_df, safety_df)

    print("Saving transformed data...")
    save_transformed_data(engine, clean_df)

    print("Processing completed successfully!")

    print("Raw rows:", len(trials_df))
    print("Clean rows:", len(clean_df))
    print("Rows removed:", len(trials_df) - len(clean_df))


if __name__ == "__main__":
    main()