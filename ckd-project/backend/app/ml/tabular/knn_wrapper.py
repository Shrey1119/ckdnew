import json
import logging
import joblib
import pandas as pd
from pathlib import Path
from app.config.config import settings

logger = logging.getLogger(__name__)

class KNNModelWrapper:
    def __init__(self):
        self.model_path = Path(settings.EXISTING_KNN_MODEL_PATH)
        self.features_path = Path(settings.EXISTING_KNN_FEATURES_PATH)
        self.model = None
        self.features = []
        self.load_model()

    def load_model(self):
        try:
            if not self.model_path.exists():
                # Check alternative locations up directory hierarchy
                base_dir = Path(__file__).resolve().parent
                candidates = [
                    base_dir.parents[4] / "model_artifacts" / "phase1_knn_model.joblib", # CKD_main
                    base_dir.parents[3] / "model_artifacts" / "phase1_knn_model.joblib",
                    Path("../model_artifacts/phase1_knn_model.joblib"),
                    Path("../../model_artifacts/phase1_knn_model.joblib")
                ]
                for cand in candidates:
                    if cand.exists():
                        self.model_path = cand
                        self.features_path = cand.parent / "features.json"
                        break
                else:
                    raise FileNotFoundError(f"KNN model artifacts not found at {self.model_path}")
            
            logger.info(f"Loading KNN model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            with open(self.features_path, "r", encoding="utf-8") as f:
                self.features = json.load(f)
            logger.info("KNN model and feature schema loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading KNN model: {e}")
            raise e

    def preprocess_input(self, payload: dict) -> pd.DataFrame:
        row = {}
        for feature in self.features:
            val = payload.get(feature)
            # Try converting numeric cols to float, keep string objects as is
            if val is not None and val != "":
                try:
                    # Clean numerical formatting issues if any
                    val = float(val)
                except ValueError:
                    val = str(val).strip()
            else:
                val = None
            row[feature] = val
        
        # Build DataFrame with features in exact order
        return pd.DataFrame([row], columns=self.features)

    def predict(self, payload: dict) -> dict:
        df = self.preprocess_input(payload)
        
        # Get raw prediction and probability of CKD (class 1)
        prediction = int(self.model.predict(df)[0])
        probability = float(self.model.predict_proba(df)[0][1])
        
        # Calculate confidence score relative to predicted class
        pred_prob = probability if prediction == 1 else (1.0 - probability)
        
        if pred_prob >= 0.8:
            confidence = "high"
        elif pred_prob >= 0.6:
            confidence = "medium"
        else:
            confidence = "low"
            
        # Determine risk level based on probability of CKD
        if probability >= 0.7:
            risk_level = "high"
        elif probability >= 0.3:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "label": "ckd" if prediction == 1 else "not_ckd",
            "confidence": confidence,
            "risk_level": risk_level,
            "df_row": df.to_dict(orient="records")[0]  # Useful for explainability
        }

# Global singleton instance
knn_wrapper = KNNModelWrapper()
