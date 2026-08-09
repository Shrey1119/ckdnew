import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "kidney_multimodal_dataset_FIXED.csv"
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "phase1_knn_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical_cols = [col for col in X.columns if X[col].dtype == "object"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", KNeighborsClassifier()),
        ]
    )

    return pipeline


def train() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    df = df.drop(columns=["id", "image_path"], errors="ignore")

    if "classification" not in df.columns:
        raise ValueError("classification column not found")

    y = df["classification"].astype(int)
    X = df.drop(columns=["classification"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(X)

    grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11, 13, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "best_params": search.best_params_,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
    }

    joblib.dump(best_model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)

    with FEATURES_PATH.open("w", encoding="utf-8") as fp:
        json.dump(list(X.columns), fp, indent=2)

    print("Phase 1 training complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train()
