from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import User, Prediction
from app.schemas.history import HistoryListResponse
from app.services import auth_service, db_service

router = APIRouter(tags=["Patient History"])

# Access checkers
any_role = auth_service.RoleChecker(["doctor", "researcher", "admin"])

@router.get("/history", response_model=HistoryListResponse)
def get_history(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=10, ge=1, le=100),
    search: str = Query(default=None),
    label: str = Query(default=None, description="ckd or not_ckd"),
    risk: str = Query(default=None, description="high, medium, low"),
    db: Session = Depends(get_db),
    current_user: User = Depends(any_role)
):
    items, total = db_service.get_prediction_history(
        db,
        page=page,
        size=size,
        search=search,
        label_filter=label,
        risk_filter=risk
    )
    
    return {
        "total": total,
        "page": page,
        "size": size,
        "items": items
    }

@router.get("/predictions/meta")
def get_predictions_meta(
    db: Session = Depends(get_db),
    current_user: User = Depends(any_role)
):
    total = db.query(Prediction).count()
    ckd = db.query(Prediction).filter(Prediction.prediction_label == "ckd").count()
    not_ckd = total - ckd
    
    return {
        "total_predictions": total,
        "ckd_cases": ckd,
        "healthy_cases": not_ckd
    }
