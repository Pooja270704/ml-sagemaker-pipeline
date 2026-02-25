import os
import json
import joblib
import pandas as pd


# ======================================================
# 1️⃣ Load Model (Executed once at container startup)
# ======================================================

def model_fn(model_dir):
    print("🔄 Loading model from:", model_dir)

    model_path = os.path.join(model_dir, "model.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = joblib.load(model_path)

    print("✅ Model loaded successfully.")
    return model


# ======================================================
# 2️⃣ Input Processing
# ======================================================

def input_fn(request_body, request_content_type):

    print("📥 Content type received:", request_content_type)

    # 🔥 VERY IMPORTANT — prevent ping crash
    if request_content_type is None:
        return pd.DataFrame()

    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Expect list of records
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        return df

    elif request_content_type == "text/csv":
        from io import StringIO
        return pd.read_csv(StringIO(request_body))

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# ======================================================
# 3️⃣ Prediction
# ======================================================

def predict_fn(input_data, model):

    print("🔮 Running prediction...")

    if input_data.empty:
        return []

    predictions = model.predict(input_data)

    return predictions


# ======================================================
# 4️⃣ Output Formatting
# ======================================================

def output_fn(prediction, accept):

    print("📤 Formatting output...")

    if accept is None:
        accept = "application/json"

    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), accept

    elif accept == "text/csv":
        return ",".join(map(str, prediction)), accept

    else:
        raise ValueError(f"Unsupported accept type: {accept}")