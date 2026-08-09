import json
import sys
from pathlib import Path

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"
MODEL_PATH = ARTIFACTS_DIR / "phase1_knn_model.joblib"
FEATURES_PATH = ARTIFACTS_DIR / "features.json"


def load_features() -> list[str]:
    with FEATURES_PATH.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def to_float_if_possible(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except Exception:
        return value


def main():
    if len(sys.argv) < 2:
        raise ValueError("Missing payload argument")

    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run train_phase1_knn.py first")

    payload = json.loads(sys.argv[1])
    features = load_features()

    row = {}
    for feature in features:
        row[feature] = to_float_if_possible(payload.get(feature))

    model = joblib.load(MODEL_PATH)
    data = pd.DataFrame([row], columns=features)

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    if probability >= 0.8:
        confidence = "high"
    elif probability >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"

    result = {
        "prediction": prediction,
        "probability": round(probability, 4),
        "label": "ckd" if prediction == 1 else "not_ckd",
        "confidence": confidence,
    }

    print(json.dumps(result))


if __name__ == "__main__":
    main()
