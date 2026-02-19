from dotenv import load_dotenv
load_dotenv()

import os

from src.pipelines.pipeline import DEFAULT_BUCKET, ROLE_ARN, get_pipeline


def run_pipeline_with_parameters(
    db_secret_arn: str,
    min_r2: float = 0.10,
):
    print("=" * 80)
    print("CLINICAL SAGEMAKER PIPELINE EXECUTION")
    print("=" * 80)
    print(f"Default bucket: {DEFAULT_BUCKET}")
    print(f"Minimum R2 threshold: {min_r2}")
    print("=" * 80)

    pipeline = get_pipeline()

    pipeline.upsert(
        role_arn=ROLE_ARN,
        description="Clinical full pipeline: RAW → STG → PSA → VIEW → Train → Evaluate → Conditional Register",
    )

    print("Pipeline created/updated successfully.")

    parameters = {
        "DbSecretArn": db_secret_arn,
        "RawBucket": DEFAULT_BUCKET,
        "TrialsCsvKey": "data/raw/clinical_trials.csv",
        "SafetyJsonKey": "data/raw/clinical_safety_events.json",
        "StgTrialsTable": "STG_CLINICAL_TRIALS",
        "StgSafetyTable": "STG_CLINICAL_SAFETY",
        "PsaTrialsTable": "PSA_CLINICAL_TRIAL",
        "PsaSafetyTable": "PSA_CLINICAL_SAFETY",
        "PsaViewName": "PSA_VIEW_TRAINING",
        "MinR2": min_r2,
    }

    execution = pipeline.start(parameters=parameters)

    print("Pipeline execution started.")
    print(f"Execution ARN: {execution.arn}")

    return execution
    
if __name__ == "__main__":
    secret_arn = os.getenv("DB_SECRET_ARN")

    if not secret_arn:
        raise ValueError("Set DB_SECRET_ARN environment variable before running.")

    run_pipeline_with_parameters(
        db_secret_arn=secret_arn,
        min_r2=0.10,
    )