import boto3
import json
import pandas as pd
from sqlalchemy import create_engine, text

# -----------------------------
# 1. Get Secret
# -----------------------------
client = boto3.client("secretsmanager", region_name="ap-south-1")
response = client.get_secret_value(SecretId="abalone-db-app-secret")
secret = json.loads(response["SecretString"])

engine = create_engine(
    f"mysql+pymysql://{secret['username']}:{secret['password']}"
    f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
)

# -----------------------------
# 2. TRUNCATE TABLE
# -----------------------------
with engine.connect() as conn:
    conn.execute(text("TRUNCATE TABLE clinical_trials"))
    conn.commit()

print("Table truncated successfully")

# -----------------------------
# 3. Load CSV
# -----------------------------
df = pd.read_csv("data/raw/clinical_trials.csv")

df.to_sql(
    name="clinical_trials",
    con=engine,
    if_exists="append",   # IMPORTANT: use append now
    index=False
)

print("Clinical trials data reloaded successfully!")