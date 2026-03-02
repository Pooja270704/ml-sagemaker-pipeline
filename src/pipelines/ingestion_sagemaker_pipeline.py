import boto3
import sagemaker
from sagemaker.session import Session
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep


def get_pipeline(
    region,
    role_arn,
    default_bucket,
    pipeline_name="ClinicalIngestionPipeline",
):

    boto_sess = boto3.Session(region_name=region)
    sm_session = Session(default_bucket=default_bucket, boto_session=boto_sess)

    # ======================
    # ONLY BUSINESS PARAMETERS
    # ======================
    source_param = ParameterString(name="Source", default_value="DB")
    format_param = ParameterString(name="FileFormat", default_value="csv")

    # ======================
    # PROCESSOR
    # ======================
    sklearn_image = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
    )

    processor = ScriptProcessor(
        image_uri=sklearn_image,
        command=["python3"],
        role=role_arn,
        instance_type="ml.m5.large",
        instance_count=1,
        sagemaker_session=sm_session,
    )

    step_ingestion = ProcessingStep(
        name="IngestionStep",
        processor=processor,
        code="src/pipelines/ingestion_pipeline.py",
        job_arguments=[
            "--source", source_param,
            "--file_format", format_param,
        ],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[source_param, format_param],
        steps=[step_ingestion],
        sagemaker_session=sm_session,
    )


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
    print("Ingestion Pipeline created successfully.")