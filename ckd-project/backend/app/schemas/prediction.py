from datetime import datetime
from pydantic import BaseModel, Field

class TabularPredictRequest(BaseModel):
    # Age and Blood Pressure
    age: float | None = Field(default=None, description="Age in years")
    bp: float | None = Field(default=None, description="Blood pressure in mm/Hg")
    
    # Urine chemistry
    sg: float | None = Field(default=None, description="Specific gravity (1.005 - 1.025)")
    al: float | None = Field(default=None, description="Albumin (0 - 5)")
    su: float | None = Field(default=None, description="Sugar (0 - 5)")
    
    # Microscopic urine tests
    rbc: str | None = Field(default=None, description="Red blood cells (normal, abnormal)")
    pc: str | None = Field(default=None, description="Pus cell (normal, abnormal)")
    pcc: str | None = Field(default=None, description="Pus cell clumps (present, notpresent)")
    ba: str | None = Field(default=None, description="Bacteria (present, notpresent)")
    
    # Blood chemistry
    bgr: float | None = Field(default=None, description="Blood glucose random (mgs/dl)")
    bu: float | None = Field(default=None, description="Blood urea (mgs/dl)")
    sc: float | None = Field(default=None, description="Serum creatinine (mgs/dl)")
    sod: float | None = Field(default=None, description="Sodium (mEq/L)")
    pot: float | None = Field(default=None, description="Potassium (mEq/L)")
    hemo: float | None = Field(default=None, description="Hemoglobin (gms)")
    pcv: float | None = Field(default=None, description="Packed cell volume")
    wc: float | None = Field(default=None, description="White blood cell count (cells/cumm)")
    rc: float | None = Field(default=None, description="Red blood cell count (millions/cmm)")
    
    # Clinical history & metadata
    htn: str | None = Field(default=None, description="Hypertension (yes, no)")
    dm: str | None = Field(default=None, description="Diabetes mellitus (yes, no)")
    cad: str | None = Field(default=None, description="Coronary artery disease (yes, no)")
    appet: str | None = Field(default=None, description="Appetite (good, poor)")
    pe: str | None = Field(default=None, description="Pedal edema (yes, no)")
    ane: str | None = Field(default=None, description="Anemia (yes, no)")

class TabularPredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    confidence: str
    risk_level: str
    explainability: dict | None = None

class ImagePredictResponse(BaseModel):
    predicted_class: str
    confidence_probs: dict # Likelihood for Normal, Cyst, Tumor, Stone
    confidence: str
    gradcam_url: str | None = None

class FusionPredictResponse(BaseModel):
    prediction: int
    probability: float
    label: str
    confidence: str
    risk_level: str
    tabular_prob: float
    image_prob: float
    tabular_weight: float | None = 0.5
    image_weight: float | None = 0.5
    fusion_type: str = "cross_attention"
    image_class: str | None = None
    ct_confidence: float | None = None
    ct_confidence_probs: dict | None = None
    gradcam_url: str | None = None
    explainability: dict | None = None
    meta: dict | None = None

class ModelBenchmarkItem(BaseModel):
    modality: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: list[list[int]] | None = None

class ModelComparisonResponse(BaseModel):
    best_fusion_model: str
    timestamp: str | None = None
    benchmarks: dict[str, ModelBenchmarkItem]
    comparison_chart_url: str | None = None

