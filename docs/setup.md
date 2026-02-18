# Week 1 – Setup & Infrastructure Details

## Objective
Prepare architecture, environment, and repository structure for the SageMaker MLOps project.

## AWS Resources
| Resource | Name | Security Features | Status |
|-----------|------|------------------|--------|
| S3 Bucket | ml-sagemaker-pipeline-demo | Versioning + SSE-S3 + Block Public Access | ✅ Created |
| IAM Role  | Sagemaker-Execution-Role | Least privilege trust policy | ✅ Created |
| IAM Role  | GithubActionsSagemakerRole | Least privilege trust policy | ✅ Created |

## Verification
- Test S3 upload successful (`tests/test_s3_upload.py`)
- Verified IAM role permissions
- Architecture diagram added

## Security Considerations
- No hardcoded credentials  
- Using AWS-managed encryption keys  
- IAM scoped to SageMaker  
- CI/CD to use GitHub Secrets for AWS access
