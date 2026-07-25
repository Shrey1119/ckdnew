import logging
from pathlib import Path
from sqlalchemy.orm import Session

from app.config.config import settings
from app.ml.tabular.knn_wrapper import knn_wrapper
from app.ml.tabular.explainability import tabular_explainer
from app.ml.image_model_loader import image_model_loader
from app.ml.image.explainability import generate_gradcam_image, CLASS_NAMES
from app.ml.fusion.late_fusion import predict_late_fusion, compute_image_ckd_prob
from app.ml.fusion.early_fusion import early_fusion_manager
from app.services import db_service

logger = logging.getLogger(__name__)

def get_image_prediction_probabilities(image_path: str) -> dict:
    """
    Run the ResNet18 model via image_model_loader and return probabilities for the 4 categories.
    """
    return image_model_loader.predict_image_probs(image_path)

def run_tabular_prediction(db: Session, payload: dict, user_id: int = None) -> dict:
    # 1. Run predictions
    res = knn_wrapper.predict(payload)
    
    # 2. Add SHAP explainability
    shap_res = tabular_explainer.explain(payload)
    res["explainability"] = shap_res if shap_res.get("status") == "success" else None
    
    # 3. Save to database
    db_patient = db_service.create_patient_record(db, res["df_row"], classification=res["prediction"])
    db_service.save_prediction(
        db,
        patient_id=db_patient.id,
        prediction_label=res["label"],
        confidence=res["confidence"],
        risk_level=res["risk_level"],
        tabular_prob=res["probability"],
        explainability_data=res["explainability"],
        created_by=user_id
    )
    
    return res

def run_image_prediction(db: Session, image_path: str, user_id: int = None) -> dict:
    # 1. Run inference
    probs = get_image_prediction_probabilities(image_path)
    pred_class = max(probs, key=probs.get)
    pred_prob = probs[pred_class]
    
    # Compute confidence tag
    if pred_prob >= 0.8:
        confidence = "high"
    elif pred_prob >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"
        
    # 2. Add Grad-CAM
    cam_res = generate_gradcam_image(image_path)
    gradcam_url = cam_res.get("gradcam_url") if cam_res.get("status") == "success" else None
    
    # 3. Create dummy clinical inputs for DB record mapping
    dummy_payload = {"sc": 4.0 if pred_class == "Tumor" else 1.2}
    # Map to classification label
    classification = 1 if pred_class in ["Tumor", "Stone", "Cyst"] else 0
    
    db_patient = db_service.create_patient_record(db, dummy_payload, image_path=image_path, classification=classification)
    
    img_ckd_prob = compute_image_ckd_prob(probs)
    
    db_service.save_prediction(
        db,
        patient_id=db_patient.id,
        prediction_label="ckd" if classification == 1 else "not_ckd",
        confidence=confidence,
        risk_level="high" if pred_class == "Tumor" else "medium" if pred_class in ["Stone", "Cyst"] else "low",
        image_prob=img_ckd_prob,
        image_path=image_path,
        explainability_data={"gradcam_url": gradcam_url, "probs": probs},
        created_by=user_id
    )
    
    # Register upload attachment
    db_service.save_upload_record(db, Path(image_path).name, image_path, patient_id=db_patient.id)
    
    return {
        "predicted_class": pred_class,
        "confidence_probs": probs,
        "confidence": confidence,
        "gradcam_url": gradcam_url
    }

def run_fusion_prediction(
    db: Session, 
    tabular_payload: dict, 
    image_path: str, 
    fusion_type: str = "late", 
    tabular_weight: float = None, 
    image_weight: float = None, 
    user_id: int = None
) -> dict:
    """
    Combines clinical values and images into multimodal predictions.
    """
    # 1. Run individual predictions
    tab_res = knn_wrapper.predict(tabular_payload)
    img_probs = get_image_prediction_probabilities(image_path)
    img_ckd_prob = compute_image_ckd_prob(img_probs)
    pred_class = max(img_probs, key=img_probs.get)
    
    # 2. Run Fusion logic
    if fusion_type == "early":
        # MLP Fusion
        fusion_res = early_fusion_manager.predict(tabular_payload, image_path)
        if fusion_res.get("status") == "error":
            logger.warning(f"Early fusion failed: {fusion_res.get('message')}. Falling back to late fusion.")
            # Fallback
            fusion_res = predict_late_fusion(tab_res["probability"], img_probs, tabular_weight, image_weight)
            fusion_res["fusion_type"] = "late_fallback"
        else:
            fusion_res["fusion_type"] = "early"
            fusion_res["tabular_contrib"] = 0.5 # Dummy representational contribs
            fusion_res["image_contrib"] = 0.5
    else:
        # Late Fusion
        fusion_res = predict_late_fusion(tab_res["probability"], img_probs, tabular_weight, image_weight)
        fusion_res["fusion_type"] = "late"
        
    # 3. Add explainability metrics
    shap_res = tabular_explainer.explain(tabular_payload)
    cam_res = generate_gradcam_image(image_path)
    
    gradcam_url = cam_res.get("gradcam_url") if cam_res.get("status") == "success" else None
    
    explainability_payload = {
        "shap": shap_res if shap_res.get("status") == "success" else None,
        "gradcam_url": gradcam_url,
        "image_probs": img_probs
    }
    
    # 4. Save to Database
    db_patient = db_service.create_patient_record(
        db, 
        tab_res["df_row"], 
        image_path=image_path, 
        classification=fusion_res["prediction"]
    )
    
    db_service.save_prediction(
        db,
        patient_id=db_patient.id,
        prediction_label=fusion_res["label"],
        confidence=fusion_res["confidence"],
        risk_level=fusion_res["risk_level"],
        tabular_prob=tab_res["probability"],
        image_prob=img_ckd_prob,
        fusion_prob=fusion_res["probability"],
        image_path=image_path,
        explainability_data=explainability_payload,
        created_by=user_id
    )
    
    # Register upload attachment
    db_service.save_upload_record(db, Path(image_path).name, image_path, patient_id=db_patient.id)
    
    return {
        "prediction": fusion_res["prediction"],
        "probability": fusion_res["probability"],
        "label": fusion_res["label"],
        "confidence": fusion_res["confidence"],
        "risk_level": fusion_res["risk_level"],
        "tabular_prob": tab_res["probability"],
        "image_prob": img_ckd_prob,
        "tabular_weight": tabular_weight or settings.DEFAULT_TABULAR_WEIGHT,
        "image_weight": image_weight or settings.DEFAULT_IMAGE_WEIGHT,
        "fusion_type": fusion_res.get("fusion_type", fusion_type),
        "image_class": pred_class,
        "gradcam_url": gradcam_url,
        "explainability": explainability_payload
    }
