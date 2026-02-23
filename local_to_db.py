import subprocess
import sys
import os
import json
import pandas as pd
import boto3
from sqlalchemy import create_engine, text, String  # Added String import

# --------------------------------------------------
# 🛠️ SETUP
# --------------------------------------------------
SECRET_ARN = "abalone-db-app-secret"
REGION = "ap-south-1" 

CSV_PATH = "data/raw/clinical_trials.csv"
JSON_PATH = "data/raw/clinical_safety_events.json"

TABLE_TRIALS = "STG_clinical_trials"
TABLE_SAFETY = "STG_clinical_safety_events"

# --------------------------------------------------
# 🔗 DB CONNECTION
# --------------------------------------------------
def get_engine():
    client = boto3.client("secretsmanager", region_name=REGION)
    resp = client.get_secret_value(SecretId=SECRET_ARN)
    creds = json.loads(resp["SecretString"])
    
    conn_str = f"mysql+pymysql://{creds['username']}:{creds['password']}@{creds['host']}:{creds['port']}/{creds['dbname']}"
    return create_engine(conn_str)

# --------------------------------------------------
# 📤 LOAD CSV
# --------------------------------------------------
def load_csv(engine):
    if not os.path.exists(CSV_PATH):
        print(f"❌ File not found: {CSV_PATH}")
        return

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # We force the data into a list of dictionaries
    data = [{"raw_csv": line.strip()} for line in lines[1:] if line.strip()]
    df = pd.DataFrame(data)

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_TRIALS} (
                id INT AUTO_INCREMENT PRIMARY KEY, 
                raw_csv LONGTEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text(f"TRUNCATE TABLE {TABLE_TRIALS}"))
        
        # FIX: Define dtype to prevent Pandas from searching for 'DOUBLE'
        df.to_sql(
            TABLE_TRIALS, 
            con=conn, 
            if_exists="append", 
            index=False,
            dtype={"raw_csv": String}
        )
            
    print(f"✅ Successfully loaded {len(df)} rows into {TABLE_TRIALS}")

# --------------------------------------------------
# 📤 LOAD JSON
# --------------------------------------------------
def load_json(engine):
    if not os.path.exists(JSON_PATH):
        print(f"❌ File not found: {JSON_PATH}")
        return

    with open(JSON_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            records = [{"raw_json": json.dumps(item)} for item in parsed]
        else:
            records = [{"raw_json": json.dumps(parsed)}]
    except json.JSONDecodeError:
        records = [{"raw_json": line.strip()} for line in content.strip().split("\n") if line.strip()]

    df = pd.DataFrame(records)

    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_SAFETY} (
                id INT AUTO_INCREMENT PRIMARY KEY, 
                raw_json LONGTEXT, 
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text(f"TRUNCATE TABLE {TABLE_SAFETY}"))
        
        # FIX: Define dtype here too
        df.to_sql(
            TABLE_SAFETY, 
            con=conn, 
            if_exists="append", 
            index=False,
            dtype={"raw_json": String}
        )
            
    print(f"✅ Successfully loaded {len(df)} rows into {TABLE_SAFETY}")

# --------------------------------------------------
# EXECUTE
# --------------------------------------------------
if __name__ == "__main__":
    try:
        print(f"⏳ Connecting using SQLAlchemy 2.0...")
        engine = get_engine()
        print("🚀 Uploading data...")
        load_csv(engine)
        load_json(engine)
        print("✨ Finished! Data is in the DB.")
    except Exception as e:
        print(f"🚨 An error occurred: {e}")