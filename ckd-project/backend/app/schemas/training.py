from datetime import datetime
from pydantic import BaseModel

class TrainingLogSchema(BaseModel):
    id: int
    epoch: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float
    created_at: datetime

    class Config:
        from_attributes = True

class ModelStatusSchema(BaseModel):
    id: int
    name: str
    version: str
    accuracy: float | None = None
    f1: float | None = None
    is_active: bool
    updated_at: datetime

    class Config:
        from_attributes = True

class TrainingStatusResponse(BaseModel):
    is_training: bool
    current_epoch: int
    total_epochs: int
    logs: list[TrainingLogSchema]
