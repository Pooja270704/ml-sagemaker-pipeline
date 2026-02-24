import boto3
import json

# Use same region as your endpoint
region = "ap-south-1"

# Create runtime client
runtime = boto3.client("sagemaker-runtime", region_name=region)

# Your endpoint name
endpoint_name = "clinical-severity-dev"

# Change input format if needed
payload = {
    "text": "Patient experienced severe headache and nausea"
}

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload),
    InferenceComponentName="variant-1"
)

result = response["Body"].read().decode()

print("Prediction Response:")
print(result)