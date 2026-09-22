import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config.config import settings
from app.database.connection import engine, Base, SessionLocal
from app.database.models import User, ModelStatus
from app.services import auth_service
from app.api import auth, predict, train, history, upload, health

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.makedirs("logs", exist_ok=True) or "logs", "api.log"))
    ]
)
logger = logging.getLogger(__name__)

# Initialize database tables
logger.info("Initializing database tables...")
Base.metadata.create_all(bind=engine)

# Seed default users if table is empty
db = SessionLocal()
try:
    if db.query(User).count() == 0:
        logger.info("Seeding default accounts (admin, doctor, researcher)...")
        users = [
            User(username="admin", hashed_password=auth_service.get_password_hash("adminpassword"), role="admin"),
            User(username="doctor", hashed_password=auth_service.get_password_hash("doctorpassword"), role="doctor"),
            User(username="researcher", hashed_password=auth_service.get_password_hash("researcherpassword"), role="researcher")
        ]
        db.add_all(users)
        db.commit()
        logger.info("Default accounts seeded successfully.")
except Exception as e:
    logger.error(f"Error seeding database: {e}")
    db.rollback()
finally:
    db.close()

# Initialize FastAPI App
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multimodal predict backend using tabular clinical data and kidney scan images.",
    version="1.0.0"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static uploads directory for serving patient images and Grad-CAM results
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.UPLOAD_DIR, "gradcam"), exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Include Routers
app.include_router(auth.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(train.router, prefix=settings.API_V1_STR)
app.include_router(history.router, prefix=settings.API_V1_STR)
app.include_router(upload.router, prefix=settings.API_V1_STR)
app.include_router(health.router, prefix=settings.API_V1_STR)

@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.PROJECT_NAME}",
        "docs_url": "/docs",
        "api_v1_prefix": settings.API_V1_STR
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["app"])

