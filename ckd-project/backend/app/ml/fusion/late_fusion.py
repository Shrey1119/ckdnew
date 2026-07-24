import logging
from app.config.config import settings

logger = logging.getLogger(__name__)

def compute_image_ckd_prob(image_probs: dict) -> float:
    """
    Map 4-class image probabilities to a single CKD risk probability.
    In our dataset schema:
      - Normal = Healthy (0% CKD baseline risk)
      - Cyst = Mild disease (30% CKD risk)
      - Stone = Moderate disease (60% CKD risk)
      - Tumor = Severe disease (95% CKD risk)
    Alternatively, a binary categorization:
      - P(CKD) = P(Cyst) + P(Stone) + P(Tumor) = 1.0 - P(Normal)
    We will use a weighted risk representation which is clinically realistic:
    """
    p_normal = image_probs.get("Normal", 0.0)
    p_cyst = image_probs.get("Cyst", 0.0)
    p_stone = image_probs.get("Stone", 0.0)
    p_tumor = image_probs.get("Tumor", 0.0)
    
    # Weighted risk mapping
    ckd_prob_image = (p_cyst * 0.3) + (p_stone * 0.6) + (p_tumor * 0.95)
    return min(max(ckd_prob_image, 0.0), 1.0)

def predict_late_fusion(
    tabular_prob: float, 
    image_probs: dict, 
    tabular_weight: float = None, 
    image_weight: float = None
) -> dict:
    """
    Perform late fusion prediction.
    """
    if tabular_weight is None:
        tabular_weight = settings.DEFAULT_TABULAR_WEIGHT
    if image_weight is None:
        image_weight = settings.DEFAULT_IMAGE_WEIGHT
        
    # Ensure weights sum to 1.0
    total_w = tabular_weight + image_weight
    w_tab = tabular_weight / total_w
    w_img = image_weight / total_w
    
    image_ckd_prob = compute_image_ckd_prob(image_probs)
    
    # Calculate fused probability
    fused_prob = (w_tab * tabular_prob) + (w_img * image_ckd_prob)
    
    # Final label decision (threshold = 0.5)
    prediction = 1 if fused_prob >= 0.5 else 0
    confidence_val = fused_prob if prediction == 1 else (1.0 - fused_prob)
    
    if confidence_val >= 0.8:
        confidence = "high"
    elif confidence_val >= 0.6:
        confidence = "medium"
    else:
        confidence = "low"
        
    if fused_prob >= 0.7:
        risk_level = "high"
    elif fused_prob >= 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"
        
    return {
        "prediction": prediction,
        "probability": round(fused_prob, 4),
        "label": "ckd" if prediction == 1 else "not_ckd",
        "confidence": confidence,
        "risk_level": risk_level,
        "tabular_contrib": round(w_tab * tabular_prob, 4),
        "image_contrib": round(w_img * image_ckd_prob, 4)
    }
