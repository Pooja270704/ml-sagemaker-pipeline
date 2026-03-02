import boto3
import sagemaker

from sagemaker.session import Session
from sagemaker.processing import ScriptProcessor, ProcessingOutput, ProcessingInput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel

from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.model_metrics import ModelMetrics, MetricsSource


def get_pipeline(
    region: str,
    role_arn: str,
    default_bucket: str,
    pipeline_name: str = "ClinicalSeverityTrainingPipeline",
    boto_sess=None,
) -> Pipeline:

    if boto_sess is None:
        boto_sess = boto3.Session(region_name=region)

    sm_session = Session(default_bucket=default_bucket, boto_session=boto_sess)

    # ================================
    # FIXED INFRA CONFIG
    # ================================
    DEFAULT_DB_SECRET = "abalone-db-app-secret"
    PSA_TRIALS_TABLE = "PSA_clinical_trials"
    PSA_SAFETY_TABLE = "PSA_clinical_safety_events"
    MODEL_PACKAGE_GROUP = "ClinicalSeverityModelPackageGroup"

    # ================================
    # PIPELINE PARAMETERS
    # ================================
    min_accuracy = ParameterFloat(name="MinAccuracy", default_value=0.85)

    # ================================
    # SKLEARN IMAGE
    # ================================
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

    # ================================
    # STEP 1 — BUILD TRAINING DATASET
    # ================================
    step_build = ProcessingStep(
        name="BuildTrainingDataset",
        processor=processing_processor,
        code="src/processing/build_training_dataset.py",
        job_arguments=[
            "--db_secret_name", DEFAULT_DB_SECRET,
            "--region", region,
            "--psa_trials_table", PSA_TRIALS_TABLE,
            "--psa_safety_table", PSA_SAFETY_TABLE,
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
                        "training-processed",
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                    ],
                ),
            )
        ],
    )

    # ================================
    # STEP 2 — TRAIN MODEL
    # ================================
    estimator = SKLearn(
        entry_point="train.py",
        source_dir="src/models",
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
            "train": Join(
                on="/",
                values=[
                    step_build.properties.ProcessingOutputConfig.Outputs[
                        "processed"
                    ].S3Output.S3Uri,
                    "train.csv",
                ],
            )
        },
    )

    # ================================
    # STEP 3 — EVALUATE
    # ================================
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
                source=Join(
                    on="/",
                    values=[
                        step_build.properties.ProcessingOutputConfig.Outputs[
                            "processed"
                        ].S3Output.S3Uri,
                        "test.csv",
                    ],
                ),
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

    # ================================
    # STEP 4 — REGISTER MODEL
    # ================================
    sklearn_model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src/models",
        framework_version="1.2-1",
        sagemaker_session=sm_session,
    )

    step_register = RegisterModel(
        name="RegisterSeverityModel",
        model=sklearn_model,
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=MODEL_PACKAGE_GROUP,
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # ================================
    # STEP 5 — CONDITION GATE
    # ================================
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

    # ================================
    # FINAL PIPELINE
    # ================================
    return Pipeline(
        name=pipeline_name,
        parameters=[min_accuracy],
        steps=[step_build, step_train, step_eval, step_condition],
        sagemaker_session=sm_session,
    )


# ================================
# LOCAL UPSERT
# ================================
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
    print("✅ Training Pipeline deployed successfully:", pipeline.name)