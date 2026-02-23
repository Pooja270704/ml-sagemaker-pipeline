import argparse
import json
import os
import tarfile
import glob
import joblib
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report


def find_csv(input_dir: str) -> str:
    files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not files:
        raise RuntimeError(f"No CSV found under: {input_dir}")
    files.sort()
    return files[0]


def load_model(model_dir: str):
    """
    In SageMaker, TrainingStep outputs model.tar.gz.
    ProcessingStep receives that at /opt/ml/processing/model/model.tar.gz
    We must extract it to find model.joblib.
    """
    direct = os.path.join(model_dir, "model.joblib")
    if os.path.exists(direct):
        print("✅ Found direct model.joblib")
        return joblib.load(direct)

    tar_path = os.path.join(model_dir, "model.tar.gz")
    if not os.path.exists(tar_path):
        # sometimes named differently; search
        tgz = glob.glob(os.path.join(model_dir, "*.tar.gz"))
        if tgz:
            tar_path = tgz[0]

    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"No model.tar.gz found in {model_dir}")

    extract_dir = os.path.join(model_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)

    print(f"📦 Extracting model artifact: {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    extracted_model = os.path.join(extract_dir, "model.joblib")
    if not os.path.exists(extracted_model):
        raise FileNotFoundError("model.joblib not found inside extracted model.tar.gz")

    print("✅ Loaded model from extracted artifact.")
    return joblib.load(extracted_model)


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
        raise ValueError("Severity column missing in evaluation dataset")

    df = df.dropna(subset=["Severity"])

    y_true = df["Severity"]
    X = df.drop(columns=["Severity"])

    preds = model.predict(X)
    acc = accuracy_score(y_true, preds)

    report = classification_report(y_true, preds, output_dict=True)

    metrics = {
        "classification_metrics": {
            "accuracy": {"value": float(acc)},
            "report": report
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)

    print(f"✅ Evaluation written to: {out_path}")
    print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()