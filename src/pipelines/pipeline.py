import sagemaker
from sagemaker.session import Session
from sagemaker.processing import ScriptProcessor, ProcessingOutput, ProcessingInput
from sagemaker.sklearn.estimator import SKLearn

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet

from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables

import boto3 


def get_pipeline(
    region: str,
    role_arn: str,
    default_bucket: str,
    pipeline_name: str = "ClinicalSeverityPipeline",
    boto_sess = None
) -> Pipeline:
    if boto_sess is None:
        boto_sess = boto3.Session(region_name=region)
    sm_session = Session(default_bucket=default_bucket, boto_session=boto_sess)

    # =========================================================
    # Hardcoded infra config (NOT user parameters)
    # =========================================================
    DEFAULT_BUCKET = "ml-sagemaker-pipeline-demo"
    DEFAULT_TRIALS_KEY = "data/raw/clinical_trials.csv"
    DEFAULT_SAFETY_KEY = "data/raw/clinical_safety_events.json"
    DEFAULT_DB_SECRET = "abalone-db-app-secret"

    STG_TRIALS_TABLE = "STG_clinical_trials"
    STG_SAFETY_TABLE = "STG_clinical_safety_events"
    PSA_TRIALS_TABLE = "PSA_clinical_trials"
    PSA_SAFETY_TABLE = "PSA_clinical_safety_events"
    ML_READY_TABLE = "PSA_ML_READY_DATA"
    ML_READY_VIEW = "DMT_LATEST_ML_DATA"

    MODEL_PACKAGE_GROUP = "ClinicalSeverityModelPackageGroup"

    # =========================================================
    # Pipeline parameters (ONLY what user chooses)
    # =========================================================
    trials_source = ParameterString(name="TrialsSource", default_value="S3")     # S3 | DB
    trials_format = ParameterString(name="TrialsFormat", default_value="csv")   # csv | json (validated in script)
    safety_source = ParameterString(name="SafetySource", default_value="S3")     # S3 | DB
    safety_format = ParameterString(name="SafetyFormat", default_value="json")  # csv | json (validated in script)

    min_accuracy = ParameterFloat(name="MinAccuracy", default_value=0.85)

    # =========================================================
    # Containers / processors
    # =========================================================
    sklearn_image = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
    )

    processing_processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sm_session,
    )

    # =========================================================
    # Step 1: Build dataset (parameterized source/format)
    # =========================================================
    step_build = ProcessingStep(
        name="BuildDatasetParameterized",
        processor=processing_processor,
        code="src/processing/build_dataset_parameterized.py",
        job_arguments=[
            "--db_secret_arn", DEFAULT_DB_SECRET,
            "--region", region,
            "--raw_bucket", DEFAULT_BUCKET,
            "--trials_key", DEFAULT_TRIALS_KEY,
            "--safety_key", DEFAULT_SAFETY_KEY,
            "--trials_source", trials_source,
            "--trials_format", trials_format,
            "--safety_source", safety_source,
            "--safety_format", safety_format,
            "--stg_trials_table", STG_TRIALS_TABLE,
            "--stg_safety_table", STG_SAFETY_TABLE,
            "--psa_trials_table", PSA_TRIALS_TABLE,
            "--psa_safety_table", PSA_SAFETY_TABLE,
            "--ml_ready_table", ML_READY_TABLE,
            "--ml_ready_view", ML_READY_VIEW,
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output",
                destination=Join(
                    on="/",
                    values=[
                        f"s3://{default_bucket}",
                        "clinical",
                        "processed",
                        ExecutionVariables.PIPELINE_EXECUTION_ID
                    ],
                )
            )
        ],
    )

    # =========================================================
    # Step 2: Train
    # =========================================================
    estimator = SKLearn(
        entry_point="src/models/train.py",
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=sm_session,
    )

    step_train = TrainingStep(
        name="TrainSeverityModel",
        estimator=estimator,
        inputs={
            "train": step_build.properties.ProcessingOutputConfig.Outputs[
                "processed"
            ].S3Output.S3Uri
        },
    )

    # =========================================================
    # Step 3: Evaluate
    # =========================================================
    evaluation_report = PropertyFile(
        name="SeverityEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_eval = ProcessingStep(
        name="EvaluateSeverityModel",
        processor=processing_processor,
        code="src/models/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_build.properties.ProcessingOutputConfig.Outputs[
                    "processed"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{default_bucket}/clinical/evaluation/",
            )
        ],
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_eval.properties.ProcessingOutputConfig.Outputs[
                        "evaluation"
                    ].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    # =========================================================
    # Step 4: Register (conditional)
    # =========================================================
    step_register = RegisterModel(
        name="RegisterSeverityModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name=MODEL_PACKAGE_GROUP,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # =========================================================
    # Step 5: Condition gate
    # =========================================================
    step_condition = ConditionStep(
        name="RegisterIfAccuracyPasses",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_eval.name,
                    property_file=evaluation_report,
                    json_path="classification_metrics.accuracy.value",
                ),
                right=min_accuracy,
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            trials_source,
            trials_format,
            safety_source,
            safety_format,
            min_accuracy,
        ],
        steps=[step_build, step_train, step_eval, step_condition],
        sagemaker_session=sm_session,
    )


# =========================================================
# Local upsert (run once)
# =========================================================
if __name__ == "__main__":
    REGION = "ap-south-1"
    ROLE_ARN = "arn:aws:iam::604860203124:role/SageMaker-Execution-Role"
    DEFAULT_BUCKET = "ml-sagemaker-pipeline-demo"

    pipeline = get_pipeline(
        region=REGION,
        role_arn=ROLE_ARN,
        default_bucket=DEFAULT_BUCKET,
    )

    pipeline.upsert(role_arn=ROLE_ARN)
    print("✅ Pipeline deployed successfully:", pipeline.name)