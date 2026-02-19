import sagemaker
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.session import Session
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

REGION = sagemaker.Session().boto_region_name
ROLE_ARN = "arn:aws:iam::604860203124:role/SageMaker-Execution-Role"
DEFAULT_BUCKET = "ml-sagemaker-pipeline-demo"


def get_pipeline() -> Pipeline:
    sm_session = Session(default_bucket=DEFAULT_BUCKET)

    # --- Parameters (override at execution time)
    db_secret_arn = ParameterString(name="DbSecretArn", default_value="abalone-db-app-secret")

    raw_bucket = ParameterString(name="RawBucket", default_value=DEFAULT_BUCKET)
    trials_csv_key = ParameterString(name="TrialsCsvKey", default_value="data/raw/clinical_trials.csv")
    safety_json_key = ParameterString(name="SafetyJsonKey", default_value="data/raw/clinical_safety_events.json")

    stg_trials_table = ParameterString(name="StgTrialsTable", default_value="STG_CLINICAL_TRIALS")
    stg_safety_table = ParameterString(name="StgSafetyTable", default_value="STG_CLINICAL_SAFETY")

    psa_trials_table = ParameterString(name="PsaTrialsTable", default_value="PSA_CLINICAL_TRIAL")
    psa_safety_table = ParameterString(name="PsaSafetyTable", default_value="PSA_CLINICAL_SAFETY")

    psa_view_name = ParameterString(name="PsaViewName", default_value="PSA_VIEW_TRAINING")

    min_r2 = ParameterFloat(name="MinR2", default_value=0.10)

    # --- Common processor
    processing_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=REGION,
            version="1.2-1",
        ),
        command=["python3"],
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sm_session,
    )

    # 1) RAW → STG
    step_load_stg = ProcessingStep(
        name="LoadRawToSTG",
        processor=processing_processor,
        code="src/processing/load_raw_to_stg.py",
        job_arguments=[
            "--db_secret_arn", db_secret_arn,
            "--region", REGION,
            "--raw_bucket", raw_bucket,
            "--trials_csv_key", trials_csv_key,
            "--safety_json_key", safety_json_key,
            "--stg_trials_table", stg_trials_table,
            "--stg_safety_table", stg_safety_table,
        ],
    )

    # 2) STG → PSA
    step_stg_to_psa = ProcessingStep(
        name="STGToPSA",
        depends_on=[step_load_stg.name],
        processor=processing_processor,
        code="src/processing/stg_to_psa.py",
        job_arguments=[
            "--db_secret_arn", db_secret_arn,
            "--region", REGION,
            "--stg_trials_table", stg_trials_table,
            "--stg_safety_table", stg_safety_table,
            "--psa_trials_table", psa_trials_table,
            "--psa_safety_table", psa_safety_table,
        ],
    )

    # 3) PSA tables → PSA VIEW
    step_create_view = ProcessingStep(
        name="CreatePSAView",
        depends_on=[step_stg_to_psa.name],
        processor=processing_processor,
        code="src/processing/create_psa_view.py",
        job_arguments=[
            "--db_secret_arn", db_secret_arn,
            "--region", REGION,
            "--psa_trials_table", psa_trials_table,
            "--psa_safety_table", psa_safety_table,
            "--psa_view_name", psa_view_name,
        ],
    )

    # 4) Export VIEW → S3 for training
    step_process = ProcessingStep(
        name="ExportTrainingDataFromView",
        depends_on=[step_create_view.name],
        processor=processing_processor,
        code="src/processing/sm_processing_from_db.py",
        job_arguments=[
            "--db_secret_arn", db_secret_arn,
            "--region", REGION,
            "--db_table", psa_view_name,  # reads the VIEW
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed",
                source="/opt/ml/processing/output",
                destination=f"s3://{DEFAULT_BUCKET}/clinical/processed/",
            )
        ],
    )

    # --- TRAINING
    estimator = SKLearn(
        entry_point="src/training/train.py",
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        instance_count=1,
        framework_version="1.2-1",
        sagemaker_session=sm_session,
    )

    step_train = TrainingStep(
        name="TrainClinicalModel",
        depends_on=[step_process.name],
        estimator=estimator,
        inputs={
            "train": step_process.properties.ProcessingOutputConfig.Outputs["processed"].S3Output.S3Uri
        },
    )

    # --- EVALUATION
    evaluation_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(
            framework="sklearn",
            region=REGION,
            version="1.2-1",
        ),
        command=["python3"],
        role=ROLE_ARN,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sm_session,
    )

    evaluation_report = PropertyFile(
        name="ClinicalEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateClinicalModel",
        depends_on=[step_train.name],
        processor=evaluation_processor,
        code="src/evaluation/evaluate.py",
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["processed"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{DEFAULT_BUCKET}/clinical/evaluation/",
            )
        ],
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json",
                ],
            ),
            content_type="application/json",
        )
    )

    # --- REGISTER
    step_register = RegisterModel(
        name="RegisterClinicalModel",
        estimator=estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        model_package_group_name="ClinicalTrialsModelPackageGroup",
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.large"],
        approval_status="PendingManualApproval",
        model_metrics=model_metrics,
    )

    # --- CONDITION
    step_condition = ConditionStep(
        name="RegisterIfR2Passes",
        depends_on=[step_evaluate.name],
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=step_evaluate.name,
                    property_file=evaluation_report,
                    json_path="regression_metrics.r2.value",
                ),
                right=min_r2,
            )
        ],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name="ClinicalTrialsPipeline",
        parameters=[
            db_secret_arn,
            raw_bucket, trials_csv_key, safety_json_key,
            stg_trials_table, stg_safety_table,
            psa_trials_table, psa_safety_table,
            psa_view_name,
            min_r2,
        ],
        steps=[
            step_load_stg,
            step_stg_to_psa,
            step_create_view,
            step_process,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sm_session,
    )