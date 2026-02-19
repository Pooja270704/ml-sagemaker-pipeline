import argparse
import json
import boto3
from sqlalchemy import create_engine, text


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


def main():
    parser = argparse.ArgumentParser()

    # REQUIRED because pipeline passes these
    parser.add_argument("--db_secret_arn", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--psa_trials_table", required=True)
    parser.add_argument("--psa_safety_table", required=True)
    parser.add_argument("--psa_view_name", required=True)

    args = parser.parse_args()

    print("🔐 Fetching DB credentials...")
    secret = get_secret(args.db_secret_arn, args.region)

    print("🔗 Connecting to DB...")
    engine = get_engine(secret)

    view_sql = f"""
    CREATE OR REPLACE VIEW {args.psa_view_name} AS
    SELECT
        t.StudyID,
        t.PatientID,
        t.DrugName,
        t.Age,
        t.Gender,
        t.WeightKg,
        t.HeightCm,
        t.BMI,
        t.DosageMg,
        t.BiomarkerLevel,
        t.ResponseScore,
        t.AdverseEventFlag,
        t.EventDate,
        s.EventType,
        s.Severity,
        s.Hospitalized,
        s.EventDurationDays
    FROM {args.psa_trials_table} t
    LEFT JOIN {args.psa_safety_table} s
        ON t.StudyID = s.StudyID
        AND t.PatientID = s.PatientID
        AND t.DrugName = s.DrugName
    WHERE
        t.BiomarkerLevel IS NOT NULL
        AND t.BMI BETWEEN 15 AND 45;
    """

    with engine.connect() as conn:
        conn.execute(text(view_sql))
        conn.commit()

    print("✅ PSA View created successfully!")


if __name__ == "__main__":
    main()