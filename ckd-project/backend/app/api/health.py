import time
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.database.connection import get_db

router = APIRouter(prefix="/health", tags=["System Health"])

start_time = time.time()

@router.get("")
def health_check(db: Session = Depends(get_db)):
    db_connected = False
    try:
        db.execute(text("SELECT 1"))
        db_connected = True
    except Exception:
        pass
            
    return {
        "status": "healthy",
        "uptime_seconds": round(time.time() - start_time, 2),
        "database_connected": db_connected
    }
