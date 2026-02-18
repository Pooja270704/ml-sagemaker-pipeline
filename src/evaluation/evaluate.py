import argparse
import json
import os
import tarfile

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "ResponseScore"
REQUIRED_COLUMNS = [
    "StudyID",
    "PatientID",
    "DrugName",
    "Age",
    "Gender",
    "WeightKg",
    "HeightCm",
    "BMI",
    "DosageMg",
    "BiomarkerLevel",
    "AdverseEventFlag",
    "EnrolledDate",
    TARGET_COLUMN,
]
FEATURE_COLUMNS = [
    "StudyID",
    "DrugName",
    "Age",
    "Gender",
    "WeightKg",
    "HeightCm",
    "BMI",
    "DosageMg",
    "BiomarkerLevel",
    "AdverseEventFlag",
    "EnrollmentDays",
]


def _prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    clean = df[REQUIRED_COLUMNS].copy()
    for column in [
        "Age",
        "WeightKg",
        "HeightCm",
        "BMI",
        "DosageMg",
        "BiomarkerLevel",
        "AdverseEventFlag",
        TARGET_COLUMN,
    ]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean["EnrolledDate"] = pd.to_datetime(clean["EnrolledDate"], errors="coerce")
    min_date = clean["EnrolledDate"].dropna().min()
    clean["EnrollmentDays"] = (clean["EnrolledDate"] - min_date).dt.days

    clean = clean.dropna(subset=[TARGET_COLUMN])
    clean = clean.drop_duplicates(subset=["StudyID", "PatientID"], keep="first")
    return clean


def _find_csv(input_dir: str) -> str:
    csv_files = [
        file_name
        for file_name in os.listdir(input_dir)
        if file_name.endswith(".csv") and os.path.isfile(os.path.join(input_dir, file_name))
    ]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {input_dir}")
    csv_files.sort()
    return os.path.join(input_dir, csv_files[0])


def _load_model(model_dir: str):
    direct_model = os.path.join(model_dir, "model.joblib")
    if os.path.exists(direct_model):
        return joblib.load(direct_model)

    model_tar = None
    for file_name in os.listdir(model_dir):
        if file_name.endswith(".tar.gz") or file_name.endswith(".tar"):
            model_tar = os.path.join(model_dir, file_name)
            break
    if not model_tar:
        raise RuntimeError(f"No model artifact found in {model_dir}")

    extracted_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extracted_dir, exist_ok=True)
    with tarfile.open(model_tar) as tar:
        tar.extractall(path=extracted_dir)

    extracted_model = os.path.join(extracted_dir, "model.joblib")
    if not os.path.exists(extracted_model):
        raise RuntimeError("model.joblib not found in extracted model artifact")

    return joblib.load(extracted_model)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--input-data", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-data", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    model = _load_model(args.model_path)
    csv_path = _find_csv(args.input_data)

    raw_df = pd.read_csv(csv_path)
    df = _prepare_dataframe(raw_df)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    predictions = model.predict(X_test)

    metrics = {
        "regression_metrics": {
            "r2": {"value": float(r2_score(y_test, predictions))},
            "rmse": {"value": float(mean_squared_error(y_test, predictions, squared=False))},
            "mae": {"value": float(mean_absolute_error(y_test, predictions))},
        }
    }

    os.makedirs(args.output_data, exist_ok=True)
    evaluation_path = os.path.join(args.output_data, "evaluation.json")
    with open(evaluation_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file)

    print(f"Evaluation report written to: {evaluation_path}")
    print(f"R2: {metrics['regression_metrics']['r2']['value']:.4f}")
    print(f"RMSE: {metrics['regression_metrics']['rmse']['value']:.4f}")
    print(f"MAE: {metrics['regression_metrics']['mae']['value']:.4f}")


if __name__ == "__main__":
    main()
