import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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

    # Automatically detect features
    feature_columns = [col for col in df.columns if col != TARGET]

    X = df[feature_columns]
    y = df[TARGET]

    # Auto-detect categorical vs numeric
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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1500),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    best_model = None
    best_accuracy = 0
    best_model_name = None

    print("🏋️ Training models...")

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        preds = pipeline.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = pipeline
            best_model_name = name

    print("🏆 Best Model:", best_model_name)
    print("🏆 Best Accuracy:", best_accuracy)

    model_dir = "/opt/ml/model"
    os.makedirs(model_dir, exist_ok=True)

    joblib.dump(best_model, os.path.join(model_dir, "model.joblib"))
    print("✅ Model saved successfully.")


if __name__ == "__main__":
    train()