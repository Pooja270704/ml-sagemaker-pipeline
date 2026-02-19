import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    "EventDate",
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
CATEGORICAL_COLUMNS = ["StudyID", "DrugName", "Gender"]
NUMERIC_COLUMNS = [
    "Age",
    "WeightKg",
    "HeightCm",
    "BMI",
    "DosageMg",
    "BiomarkerLevel",
    "AdverseEventFlag",
    "EnrollmentDays",
]


def _find_training_csv(train_dir: str) -> str:
    if not os.path.isdir(train_dir):
        raise RuntimeError(f"Training data directory does not exist: {train_dir}")

    csv_files = [
        file_name
        for file_name in os.listdir(train_dir)
        if file_name.endswith(".csv") and os.path.isfile(os.path.join(train_dir, file_name))
    ]
    if not csv_files:
        raise RuntimeError(f"No CSV files found in {train_dir}")

    csv_files.sort()
    return os.path.join(train_dir, csv_files[0])


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

    clean["EventDate"] = pd.to_datetime(clean["EventDate"], errors="coerce")
    min_date = clean["EventDate"].dropna().min()
    clean["EnrollmentDays"] = (clean["EventDate"] - min_date).dt.days

    clean = clean.dropna(subset=[TARGET_COLUMN])
    clean = clean.drop_duplicates(subset=["StudyID", "PatientID"], keep="first")
    return clean


def _build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_COLUMNS,
            ),
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLUMNS,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="/opt/ml/input/data/train")
    args = parser.parse_args()

    data_path = _find_training_csv(args.train)
    print(f"Loading training data from: {data_path}")
    raw_df = pd.read_csv(data_path)
    df = _prepare_dataframe(raw_df)

    print(f"Prepared training shape: {df.shape}")
    print(f"Prepared columns: {df.columns.tolist()}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_pipeline = _build_model()
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)

    metrics = {
        "regression_metrics": {
            "r2": {"value": float(r2_score(y_test, predictions))},
            "rmse": {"value": float(mean_squared_error(y_test, predictions, squared=False))},
            "mae": {"value": float(mean_absolute_error(y_test, predictions))},
        }
    }

    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model_pipeline, os.path.join(model_dir, "model.joblib"))

    metrics_dir = "/opt/ml/output/metrics"
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, "train_metrics.json"), "w", encoding="utf-8") as file:
        json.dump(metrics, file)

    print("Training complete.")
    print(f"R2: {metrics['regression_metrics']['r2']['value']:.4f}")
    print(f"RMSE: {metrics['regression_metrics']['rmse']['value']:.4f}")
    print(f"MAE: {metrics['regression_metrics']['mae']['value']:.4f}")


if __name__ == "__main__":
    main()
