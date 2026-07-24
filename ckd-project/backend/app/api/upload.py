import os
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session

from app.database.connection import get_db
from app.database.models import User
from app.services import auth_service, db_service
from app.config.config import settings

router = APIRouter(prefix="/upload", tags=["Uploads"])

doctor_or_above = auth_service.RoleChecker(["doctor", "admin"])

@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_file(
    file: UploadFile = File(...),
    patient_id: int = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(doctor_or_above)
):
    # Setup dir
    uploads_dir = Path(settings.UPLOAD_DIR)
    os.makedirs(uploads_dir, exist_ok=True)
    
    file_path = uploads_dir / file.filename
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Log to db
        db_upload = db_service.save_upload_record(
            db, 
            filename=file.filename, 
            filepath=str(file_path), 
            patient_id=patient_id
        )
        
        return {
            "status": "success",
            "upload_id": db_upload.id,
            "filename": file.filename,
            "filepath": str(file_path)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )
