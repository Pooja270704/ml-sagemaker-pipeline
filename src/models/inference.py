import json
import joblib
import os
import pandas as pd


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)

        # Ensure input is list of records
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        return df
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model):
    predictions = model.predict(input_data)
    return predictions


def output_fn(prediction, content_type):
    if content_type == "application/json":
        return json.dumps({
            "predictions": prediction.tolist()
        }), content_type
    else:
        raise ValueError("Unsupported content type: {}".format(content_type))