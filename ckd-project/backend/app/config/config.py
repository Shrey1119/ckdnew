import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "CKD Multimodal Prediction System"
    API_V1_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = Field(default="supersecretkeyforckdpredictionapp12345!@#", validation_alias="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"
    
    # Database
    DATABASE_URL: str = Field(default="sqlite:///./ckd.db", validation_alias="DATABASE_URL")
    
    # File Uploads
    UPLOAD_DIR: str = Field(default="./uploads", validation_alias="UPLOAD_DIR")
    
    # Model Paths
    SAVED_MODELS_DIR: str = Field(default="./saved_models", validation_alias="SAVED_MODELS_DIR")
    # Existing model paths relative to backend root
    EXISTING_KNN_MODEL_PATH: str = Field(default="../model_artifacts/phase1_knn_model.joblib", validation_alias="EXISTING_KNN_MODEL_PATH")
    EXISTING_KNN_FEATURES_PATH: str = Field(default="../model_artifacts/features.json", validation_alias="EXISTING_KNN_FEATURES_PATH")
    EXISTING_RESNET_MODEL_PATH: str = Field(default="../model_artifacts/phase2_resnet.pth", validation_alias="EXISTING_RESNET_MODEL_PATH")
    
    # Dataset Paths
    DATASET_CSV_PATH: str = Field(default="../kidney_multimodal_dataset_FIXED.csv", validation_alias="DATASET_CSV_PATH")
    DATASET_IMAGES_DIR: str = Field(default="../kidney_images", validation_alias="DATASET_IMAGES_DIR")
    
    # Fusion Weights
    DEFAULT_TABULAR_WEIGHT: float = 0.6
    DEFAULT_IMAGE_WEIGHT: float = 0.4
    
    # Model configs
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 16
    EPOCHS: int = 10
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
# Ensure directory setup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
