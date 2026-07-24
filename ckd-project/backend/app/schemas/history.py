from datetime import datetime
from pydantic import BaseModel

class PatientHistorySchema(BaseModel):
    id: int
    age: float | None = None
    bp: float | None = None
    sc: float | None = None
    hemo: float | None = None
    classification: int | None = None
    image_path: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True

class PredictionHistorySchema(BaseModel):
    id: int
    patient_id: int
    tabular_prob: float | None = None
    image_prob: float | None = None
    fusion_prob: float | None = None
    prediction_label: str
    confidence: str
    risk_level: str
    image_path: str | None = None
    created_at: datetime
    patient: PatientHistorySchema | None = None

    class Config:
        from_attributes = True

class HistoryListResponse(BaseModel):
    total: int
    page: int
    size: int
    items: list[PredictionHistorySchema]
