import json
import joblib
import pandas as pd
import os


# ============================================
# 1️⃣ Load Model (Called Once When Container Starts)
# ============================================

def model_fn(model_dir):
    print("🔄 Loading model from:", model_dir)

    model_path = os.path.join(model_dir, "model.joblib")

    model = joblib.load(model_path)

    print("✅ Model loaded successfully.")
    return model


# ============================================
# 2️⃣ Input Processing (Request → DataFrame)
# ============================================

def input_fn(request_body, request_content_type):

    print("📥 Received request with content type:", request_content_type)

    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Expect list of records
        df = pd.DataFrame(data)

        print("Input converted to DataFrame. Shape:", df.shape)
        return df

    elif request_content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
        return df

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# ============================================
# 3️⃣ Prediction Logic
# ============================================

def predict_fn(input_data, model):

    print("🔮 Making predictions...")

    predictions = model.predict(input_data)

    return predictions


# ============================================
# 4️⃣ Output Formatting
# ============================================

def output_fn(prediction, accept):

    print("📤 Formatting output...")

    if accept == "application/json":
        return json.dumps({"predictions": prediction.tolist()}), accept

    elif accept == "text/csv":
        return ",".join(map(str, prediction)), accept

    else:
        raise ValueError(f"Unsupported accept type: {accept}")