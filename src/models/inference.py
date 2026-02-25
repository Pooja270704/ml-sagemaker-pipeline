import os
import json
import joblib
import pandas as pd


# ===============================
# 1️⃣ Load Model
# ===============================
def model_fn(model_dir):
    print("Loading model from:", model_dir)
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    return model


# ===============================
# 2️⃣ Input Processing
# ===============================
def input_fn(request_body, request_content_type):

    print("Content type received:", request_content_type)

    # Prevent ping crash
    if request_content_type is None:
        return pd.DataFrame()

    if request_content_type == "application/json":

        data = json.loads(request_body)

        # If single record → convert to list
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)
        print("JSON converted to DataFrame. Shape:", df.shape)
        return df

    elif request_content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
        print("CSV converted to DataFrame. Shape:", df.shape)
        return df

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# ===============================
# 3️⃣ Prediction Logic
# ===============================
def predict_fn(input_data, model):

    print("Incoming columns:", input_data.columns.tolist())

    # Get expected feature columns from trained model
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        # Fallback (should not happen in your case)
        expected_cols = input_data.columns.tolist()

    print("Model expects columns:", expected_cols)

    # Add missing columns safely
    for col in expected_cols:
        if col not in input_data.columns:
            print(f"Adding missing column: {col}")
            input_data[col] = 0

    # Keep only expected columns in correct order
    input_data = input_data[expected_cols]

    print("Final columns used for prediction:", input_data.columns.tolist())

    predictions = model.predict(input_data)

    return predictions


# ===============================
# 4️⃣ Output Formatting
# ===============================
def output_fn(prediction, content_type):

    response = {
        "predictions": prediction.tolist()
    }

    return json.dumps(response), "application/json"