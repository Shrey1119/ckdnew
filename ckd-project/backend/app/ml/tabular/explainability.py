import logging
import pandas as pd
import numpy as np
import shap
from pathlib import Path
from app.config.config import settings
from app.ml.tabular.knn_wrapper import knn_wrapper

logger = logging.getLogger(__name__)

class TabularExplainer:
    def __init__(self):
        self.explainer = None
        self.background_df = None
        self.initialized = False

    def initialize(self):
        if self.initialized:
            return
        try:
            csv_path = Path(settings.DATASET_CSV_PATH)
            if not csv_path.exists():
                # Fallback to parent workspace
                alt_path = Path(__file__).resolve().parents[4] / "kidney_multimodal_dataset_FIXED.csv"
                if alt_path.exists():
                    csv_path = alt_path
                else:
                    logger.warning("Dataset CSV not found. SHAP will operate on dummy background.")
                    # Build dummy background from features
                    dummy_data = {feat: [0.0]*10 for feat in knn_wrapper.features}
                    self.background_df = pd.DataFrame(dummy_data)
                    self.initialized = True
                    return

            df = pd.read_csv(csv_path)
            # Remove metadata and targets
            df = df.drop(columns=["id", "image_path", "classification"], errors="ignore")
            
            # Align features
            df = df[knn_wrapper.features]
            
            # Subsample to speed up KernelExplainer (50 samples is optimal)
            self.background_df = shap.sample(df, 50, random_state=42)
            
            # Define predictor function targeting CKD probability (class 1)
            def predict_fn(x):
                # Convert back to DataFrame
                x_df = pd.DataFrame(x, columns=knn_wrapper.features)
                # Map data types if needed
                for col in x_df.columns:
                    if x_df[col].dtype == object:
                        # Clean values
                        x_df[col] = x_df[col].astype(str).str.strip()
                # Run model prediction probabilities
                return knn_wrapper.model.predict_proba(x_df)[:, 1]
            
            logger.info("Initializing SHAP KernelExplainer...")
            self.explainer = shap.KernelExplainer(predict_fn, self.background_df)
            self.initialized = True
            logger.info("SHAP Explainer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP Explainer: {e}")
            self.explainer = None

    def explain(self, patient_dict: dict) -> dict:
        """
        Compute SHAP values for a single patient record.
        """
        if not self.initialized:
            self.initialize()
            
        if not self.explainer:
            return {"status": "error", "message": "SHAP Explainer not initialized"}
            
        try:
            # Preprocess the payload using the wrapper
            df_row = knn_wrapper.preprocess_input(patient_dict)
            
            # Compute shap values
            shap_values = self.explainer.shap_values(df_row)
            
            # Handle list output for shap values
            if isinstance(shap_values, list):
                # KernelExplainer on single output returns list of arrays
                # We target class 1 probability, which is index 0 in lists if 1D output
                shap_arr = shap_values[0]
            else:
                shap_arr = shap_values
                
            # If 2D (batch size 1, num features), flatten
            if len(shap_arr.shape) == 2:
                shap_arr = shap_arr[0]
                
            base_value = float(self.explainer.expected_value)
            
            # Map features to values
            contributions = {}
            for feat, val, shap_val in zip(knn_wrapper.features, df_row.iloc[0], shap_arr):
                contributions[feat] = {
                    "feature_value": val if pd.notna(val) else "Missing",
                    "shap_value": float(shap_val)
                }
            
            # Sort contributions by absolute impact
            sorted_impact = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]["shap_value"]),
                reverse=True
            )
            
            return {
                "status": "success",
                "base_value": base_value,
                "prediction_impact": [
                    {
                        "feature": item[0],
                        "value": item[1]["feature_value"],
                        "shap": round(item[1]["shap_value"], 5)
                    } for item in sorted_impact
                ]
            }
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return {"status": "error", "message": str(e)}

# Global explainer instance
tabular_explainer = TabularExplainer()
