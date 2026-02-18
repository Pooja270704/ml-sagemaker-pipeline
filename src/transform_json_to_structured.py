import json
import boto3
import pandas as pd
from sqlalchemy import create_engine

# -------------------------
# 1️⃣ Get DB credentials
# -------------------------
client = boto3.client("secretsmanager", region_name="ap-south-1")

response = client.get_secret_value(
    SecretId="abalone-db-app-secret"
)

secret = json.loads(response["SecretString"])

engine = create_engine(
    f"mysql+pymysql://{secret['username']}:{secret['password']}"
    f"@{secret['host']}:{secret['port']}/{secret['dbname']}"
)

# -------------------------
# 2️⃣ Read raw JSON table
# -------------------------
raw_df = pd.read_sql(
    "SELECT raw_json FROM clinical_safety_events_raw",
    engine
)

print("Raw JSON rows:", len(raw_df))

# -------------------------
# 3️⃣ Normalize JSON
# -------------------------

# Convert JSON column properly
parsed_json = raw_df["raw_json"].apply(
    lambda x: json.loads(x) if isinstance(x, str) else x
)

json_expanded = pd.json_normalize(parsed_json)

print("Structured rows:", len(json_expanded))
print("Columns:", json_expanded.columns)
# -------------------------
# 4️⃣ Save to structured table
# -------------------------
json_expanded.to_sql(
    "clinical_safety_events",
    con=engine,
    if_exists="replace",
    index=False
)

print("JSON transformed and loaded successfully!")