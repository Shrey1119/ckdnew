import os
import shutil
import json
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.config.config import settings
from app.database.connection import get_db
from app.database.models import User
from app.schemas.prediction import (
    TabularPredictRequest, 
    TabularPredictResponse, 
    ImagePredictResponse, 
    FusionPredictResponse,
    ModelComparisonResponse
)
from app.services import auth_service, predict_service

router = APIRouter(tags=["Predictions"])
logger = logging.getLogger(__name__)

# Access checkers
doctor_or_admin = auth_service.RoleChecker(["doctor", "admin"])
any_role = auth_service.RoleChecker(["doctor", "researcher", "admin"])

@router.post("/predict/tabular", response_model=TabularPredictResponse)
@router.post("/predictions/predict-tabular", response_model=TabularPredictResponse)
def predict_tabular(
    request: TabularPredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(doctor_or_admin)
):
    payload = request.model_dump(exclude_none=True)
    try:
        result = predict_service.run_tabular_prediction(db, payload, user_id=current_user.id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tabular prediction failed: {str(e)}"
        )

@router.post("/predict/image", response_model=ImagePredictResponse)
@router.post("/predictions/predict-image", response_model=ImagePredictResponse)
async def predict_image(
    file: UploadFile = File(None),
    image_path: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(doctor_or_admin)
):
    # Handle image source
    local_path = None
    if file:
        uploads_dir = Path(settings.UPLOAD_DIR)
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = uploads_dir / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        local_path = str(file_path)
    elif image_path:
        p = Path(image_path)
        if not p.exists():
            p = Path(settings.DATASET_CSV_PATH).parent / image_path
            
        if p.exists():
            local_path = str(p)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference image path {image_path} not found on server disk"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either file upload or image_path"
        )
        
    try:
        result = predict_service.run_image_prediction(db, local_path, user_id=current_user.id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image prediction failed: {str(e)}"
        )

@router.post("/predict/fusion", response_model=FusionPredictResponse)
@router.post("/predict/multimodal", response_model=FusionPredictResponse)
@router.post("/predictions/predict-multimodal", response_model=FusionPredictResponse)
async def predict_multimodal(
    patient_data: str = Form(...), 
    file: UploadFile = File(None),
    image_path: str = Form(None),
    fusion_type: str = Form("cross_attention"),
    tabular_weight: float = Form(None),
    image_weight: float = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(doctor_or_admin)
):
    try:
        payload = json.loads(patient_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="patient_data parameter must be a valid JSON string"
        )
        
    local_path = None
    if file:
        uploads_dir = Path(settings.UPLOAD_DIR)
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = uploads_dir / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        local_path = str(file_path)
    elif image_path:
        p = Path(image_path)
        if not p.exists():
            p = Path(settings.DATASET_CSV_PATH).parent / image_path
            
        if p.exists():
            local_path = str(p)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference image path {image_path} not found on server disk"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either file upload or image_path"
        )
        
    try:
        result = predict_service.run_multimodal_prediction(
            db,
            payload,
            local_path,
            fusion_type=fusion_type,
            tabular_weight=tabular_weight,
            image_weight=image_weight,
            user_id=current_user.id
        )
        return result
    except Exception as e:
        logger.error(f"Multimodal fusion prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multimodal prediction failed: {str(e)}"
        )

@router.get("/predictions/compare", response_model=ModelComparisonResponse)
@router.get("/predict/compare", response_model=ModelComparisonResponse)
def get_model_comparison(
    current_user: User = Depends(any_role)
):
    """
    Returns comparative evaluation metrics (accuracy, precision, recall, f1, roc_auc)
    across Phase 1 KNN, Phase 2 ResNet-18, and Phase 3 Fusion architectures.
    """
    try:
        return predict_service.get_model_comparison_metrics()
    except Exception as e:
        logger.error(f"Failed to get model comparison metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model comparison: {str(e)}"
        )

@router.post("/predictions/predict-optimized", response_model=FusionPredictResponse)
@router.post("/predict/optimized", response_model=FusionPredictResponse)
async def predict_optimized(
    patient_data: str = Form(...), 
    file: UploadFile = File(None),
    image_path: str = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(doctor_or_admin)
):
    """
    Phase 4 Preview: Multimodal prediction with genetic-algorithm optimized weights.
    """
    try:
        payload = json.loads(patient_data)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="patient_data parameter must be a valid JSON string"
        )
        
    local_path = None
    if file:
        uploads_dir = Path(settings.UPLOAD_DIR)
        os.makedirs(uploads_dir, exist_ok=True)
        file_path = uploads_dir / file.filename
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        local_path = str(file_path)
    elif image_path:
        p = Path(image_path)
        if not p.exists():
            p = Path(settings.DATASET_CSV_PATH).parent / image_path
            
        if p.exists():
            local_path = str(p)
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference image path {image_path} not found on server disk"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Must supply either file upload or image_path"
        )

    try:
        return predict_service.run_optimized_prediction(
            db,
            payload,
            local_path,
            user_id=current_user.id
        )
    except Exception as e:
        logger.error(f"Optimized prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimized prediction failed: {str(e)}"
        )

