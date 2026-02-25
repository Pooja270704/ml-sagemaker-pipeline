import os
import json
import joblib
import pandas as pd


# =========================
# Model Loading
# =========================

def model_fn(model_dir):
    print("🔄 Loading model...")
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print("✅ Model loaded successfully.")
    return model


# =========================
# Input Processing
# =========================

def input_fn(request_body, request_content_type):

    print("📥 Content type received:", request_content_type)

    # 🔥 Prevent ping crash
    if request_content_type is None:
        return pd.DataFrame()

    if request_content_type == "application/json":

        data = json.loads(request_body)

        # Allow single record
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        print("📊 Incoming columns:", df.columns.tolist())

        # ------------------------------
        # 🔥 SCHEMA ALIGNMENT FIX
        # ------------------------------

        # Training dataset contains 'EnrollmentDays'
        # Inference payload contains 'EventDurationDays'
        if "EventDurationDays" in df.columns and "EnrollmentDays" not in df.columns:
            df["EnrollmentDays"] = df["EventDurationDays"]

        # Drop raw column if it doesn't exist in training
        if "EventDurationDays" in df.columns:
            df = df.drop(columns=["EventDurationDays"])

        print("📊 Final columns after alignment:", df.columns.tolist())

        return df

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# =========================
# Prediction
# =========================

def predict_fn(input_data, model):

    print("🔮 Running prediction...")

    predictions = model.predict(input_data)

    return predictions


# =========================
# Output Formatting
# =========================

def output_fn(prediction, accept):

    print("📤 Formatting output...")

    if accept == "application/json":

        result = {
            "predictions": prediction.tolist()
        }

        return json.dumps(result), accept

    else:
        raise ValueError(f"Unsupported accept type: {accept}")