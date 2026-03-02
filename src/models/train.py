import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


TARGET = "Severity"


def train():

    print("📥 Loading training dataset...")

    input_path = "/opt/ml/input/data/train"
    files = os.listdir(input_path)
    file_path = os.path.join(input_path, files[0])

    df = pd.read_csv(file_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    if TARGET not in df.columns:
        raise ValueError("Target column 'Severity' not found!")

    df = df.dropna(subset=[TARGET])

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    print("Target distribution:")
    print(y.value_counts())

    # Detect categorical & numeric columns
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", numeric_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numeric_cols,
            )
        ]
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(kernel="rbf", probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    best_model = None
    best_model_name = None

    print("🏋️ Training models...")

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X, y)

        print(f"{name} trained successfully.")

        # For simplicity: choose first model as best
        # (Evaluation step will decide actual performance)
        if best_model is None:
            best_model = pipeline
            best_model_name = name

    print("🏆 Selected Model:", best_model_name)

    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))
    print("✅ Model saved successfully.")


if __name__ == "__main__":
    train()