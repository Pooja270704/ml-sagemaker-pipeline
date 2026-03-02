import argparse
import json
import os
import tarfile
import glob
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


# ======================================================
# Utility: Find CSV file inside input directory
# ======================================================
def find_csv(input_dir: str) -> str:
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV found under: {input_dir}")
    files.sort()
    return files[0]


# ======================================================
# Utility: Load model artifact safely
# ======================================================
def load_model(model_dir: str):
    """
    TrainingStep outputs model.tar.gz
    ProcessingStep receives it under:
    /opt/ml/processing/model/

    We must extract model.joblib
    """

    # Case 1: Direct model.joblib exists
    direct = os.path.join(model_dir, "model.joblib")
    if os.path.exists(direct):
        print("✅ Found direct model.joblib")
        return joblib.load(direct)

    # Case 2: model.tar.gz
    tar_files = glob.glob(os.path.join(model_dir, "*.tar.gz"))
    if not tar_files:
        raise FileNotFoundError(f"No model artifact found in {model_dir}")

    tar_path = tar_files[0]
    extract_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"📦 Extracting model artifact: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    extracted_model = os.path.join(extract_dir, "model.joblib")

    if not os.path.exists(extracted_model):
        raise FileNotFoundError("model.joblib not found inside extracted model.tar.gz")

    print("✅ Model loaded successfully from artifact.")
    return joblib.load(extracted_model)


# ======================================================
# MAIN
# ======================================================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    print("🔎 Loading model...")
    model = load_model(args.model_dir)

    print("📥 Loading evaluation dataset...")
    csv_path = find_csv(args.input_dir)
    print("Eval CSV Path:", csv_path)

    df = pd.read_csv(csv_path)
    print("Eval dataset shape:", df.shape)

    if "Severity" not in df.columns:
        raise ValueError("❌ Severity column missing in evaluation dataset")

    # Drop rows without target
    df = df.dropna(subset=["Severity"])

    y_true = df["Severity"]
    X = df.drop(columns=["Severity"])

    print("🧠 Running predictions...")
    preds = model.predict(X)

    # ======================================================
    # Metrics
    # ======================================================
    acc = accuracy_score(y_true, preds)

    report = classification_report(
        y_true,
        preds,
        output_dict=True,
        zero_division=0  # prevents crash if a class missing
    )

    metrics = {
        "classification_metrics": {
            "accuracy": {
                "value": float(acc)
            },
            "classification_report": report
        }
    }

    # ======================================================
    # Save evaluation.json
    # ======================================================
    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join(args.output_dir, "evaluation.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print("======================================")
    print("✅ Evaluation completed successfully")
    print(f"Accuracy: {acc:.4f}")
    print(f"Saved to: {output_path}")
    print("======================================")


if __name__ == "__main__":
    main()