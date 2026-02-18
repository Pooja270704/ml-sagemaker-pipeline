import boto3
import json

client = boto3.client("secretsmanager", region_name="ap-south-1")

response = client.get_secret_value(
    SecretId="abalone-db-app-secret"
)

secret = json.loads(response["SecretString"])

print(secret)