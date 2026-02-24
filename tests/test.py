import boto3

sm = boto3.client("sagemaker", region_name="ap-south-1")

response = sm.describe_endpoint(
    EndpointName="clinical-severity-dev"
)

print(response["ProductionVariants"])