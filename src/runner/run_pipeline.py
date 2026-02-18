from dotenv import load_dotenv 
load_dotenv()

import os

from src.pipelines.pipeline import DEFAULT_BUCKET, ROLE_ARN, get_pipeline


def run_pipeline_with_parameters(
    db_secret_arn: str,
    db_table: str = "clinical_training_clean",
    min_r2: float = 0.10,
):
    print("=" * 80)
    print("CLINICAL SAGEMAKER PIPELINE EXECUTION")
    print("=" * 80)
    print(f"DB table: {db_table}")
    print(f"Minimum R2 threshold: {min_r2}")
    print(f"Default bucket: {DEFAULT_BUCKET}")
    print("=" * 80)

    pipeline = get_pipeline()
    pipeline.upsert(
        role_arn=ROLE_ARN,
        description=(
            "Clinical pipeline with processing, training, evaluation, "
            "conditional model registration"
        ),
    )
    print("Pipeline created/updated successfully.")

    parameters = {
        "DbSecretArn": db_secret_arn,
        "DbTable": db_table,
        "MinR2": min_r2,
    }
    execution = pipeline.start(parameters=parameters)

    print("Pipeline execution started.")
    print(f"Execution ARN: {execution.arn}")
    print("Steps:")
    print("1. ProcessClinicalDataFromDB")
    print("2. TrainClinicalModel")
    print("3. EvaluateClinicalModel")
    print("4. RegisterIfR2Passes")
    print("=" * 80)

    return execution


if __name__ == "__main__":
    secret_arn = os.getenv("DB_SECRET_ARN", "na")
    if secret_arn == "na":
        raise ValueError("Set DB_SECRET_ARN environment variable before running this script.")

    run_pipeline_with_parameters(
        db_secret_arn=secret_arn,
        db_table=os.getenv("DB_TABLE", "clinical_training_clean"),
        min_r2=float(os.getenv("MIN_R2", "0.10")),
    )
