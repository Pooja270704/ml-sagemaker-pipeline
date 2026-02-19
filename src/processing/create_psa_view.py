import subprocess
import sys


subprocess.check_call([sys.executable, "-m", "pip", "install","sqlalchemy==1.4.46", "pymysql"])


import argparse
import boto3
import json
from sqlalchemy import create_engine, text


# --------------------------------------------------
# Get DB credentials
# --------------------------------------------------
def get_secret(secret_arn, region):
    client = boto3.client("secretsmanager", region_name=region)
    response = client.get_secret_value(SecretId=secret_arn)
    return json.loads(response["SecretString"])


def get_engine(secret):
    connection_string = (
        f"mysql+pymysql://{secret['username']}:{secret['password']}"
        f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
    )
    return create_engine(connection_string)


# --------------------------------------------------
# Create PSA View (NO DUPLICATES)
# --------------------------------------------------
def create_psa_view(engine, trials_table, safety_table, view_name):

    view_sql = f"""
    CREATE OR REPLACE VIEW {view_name} AS
    SELECT *
    FROM (
        SELECT
            t.StudyID,
            t.PatientID,
            t.DrugName,

            CAST(t.Age AS SIGNED) AS Age,
            t.Gender,
            CAST(t.WeightKg AS DECIMAL(10,2)) AS WeightKg,
            CAST(t.HeightCm AS DECIMAL(10,2)) AS HeightCm,
            CAST(t.BMI AS DECIMAL(10,2)) AS BMI,
            CAST(t.DosageMg AS DECIMAL(10,2)) AS DosageMg,
            CAST(t.BiomarkerLevel AS DECIMAL(10,2)) AS BiomarkerLevel,
            CAST(t.ResponseScore AS DECIMAL(10,2)) AS ResponseScore,
            CAST(t.AdverseEventFlag AS SIGNED) AS AdverseEventFlag,
            t.EventDate,

            s.EventType,
            s.Severity,
            s.Hospitalized,
            s.EventDurationDays,

            ROW_NUMBER() OVER (
                PARTITION BY t.StudyID, t.PatientID
                ORDER BY t.EventDate DESC
            ) AS rn

        FROM {trials_table} t
        LEFT JOIN {safety_table} s
            ON t.StudyID = s.StudyID
            AND t.PatientID = s.PatientID
            AND t.DrugName = s.DrugName

        WHERE t.BiomarkerLevel IS NOT NULL
          AND CAST(t.BMI AS DECIMAL(10,2)) BETWEEN 15 AND 45
    ) x
    WHERE x.rn = 1;
    """

    with engine.begin() as conn:
        conn.execute(text(view_sql))

    print("✅ PSA view created successfully.")


# --------------------------------------------------
# MAIN 
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--psa_trials_table", required=True)
    parser.add_argument("--psa_safety_table", required=True)
    parser.add_argument("--psa_view_name", required=True)

    args = parser.parse_args()

    secret = get_secret(args.db_secret_arn, args.region)
    engine = get_engine(secret)

    create_psa_view(
        engine,
        args.psa_trials_table,
        args.psa_safety_table,
        args.psa_view_name
    )


if __name__ == "__main__":
    main()