import boto3
import os

# ---- CONFIGURATION ----
bucket_name = "ml-sagemaker-pipeline-demo"  # pip bucket name
file_path = "data/raw/clinical_safety_events.json"                    # local file path relative to project root
object_key = "data/raw/clinical_safety_events.json"                   # destination path in S3

# ---- AWS CLIENT SETUP ----
# Use environment credentials 
s3 = boto3.client("s3")

# ---- UPLOAD ----
try:
    s3.upload_file(file_path, bucket_name, object_key)
    print(f"✅ Uploaded {file_path} → s3://{bucket_name}/{object_key}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
