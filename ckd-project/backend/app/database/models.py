import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.database.connection import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="doctor") # admin, doctor, researcher
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="creator")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    
    # Tabular Clinical Fields
    age = Column(Float, nullable=True)
    bp = Column(Float, nullable=True)
    sg = Column(Float, nullable=True)
    al = Column(Float, nullable=True)
    su = Column(Float, nullable=True)
    rbc = Column(String, nullable=True)
    pc = Column(String, nullable=True)
    pcc = Column(String, nullable=True)
    ba = Column(String, nullable=True)
    bgr = Column(Float, nullable=True)
    bu = Column(Float, nullable=True)
    sc = Column(Float, nullable=True)
    sod = Column(Float, nullable=True)
    pot = Column(Float, nullable=True)
    hemo = Column(Float, nullable=True)
    pcv = Column(Float, nullable=True)
    wc = Column(Float, nullable=True)
    rc = Column(Float, nullable=True)
    htn = Column(String, nullable=True)
    dm = Column(String, nullable=True)
    cad = Column(String, nullable=True)
    appet = Column(String, nullable=True)
    pe = Column(String, nullable=True)
    ane = Column(String, nullable=True)
    
    # Targets/Meta
    classification = Column(Integer, nullable=True) # 0 = notckd, 1 = ckd
    image_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    predictions = relationship("Prediction", back_populates="patient", cascade="all, delete-orphan")
    uploads = relationship("Upload", back_populates="patient", cascade="all, delete-orphan")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    
    # Diagnostics Metrics
    tabular_prob = Column(Float, nullable=True)
    image_prob = Column(Float, nullable=True)
    fusion_prob = Column(Float, nullable=True)
    prediction_label = Column(String, nullable=False) # "ckd" or "not_ckd"
    confidence = Column(String, nullable=False) # "high", "medium", "low"
    risk_level = Column(String, nullable=False) # "high", "medium", "low"
    image_path = Column(String, nullable=True)
    explainability_data = Column(JSON, nullable=True) # Holds SHAP & Grad-CAM outputs
    
    # Audit Meta
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="predictions")
    creator = relationship("User", back_populates="predictions")


class Upload(Base):
    __tablename__ = "uploads"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    patient = relationship("Patient", back_populates="uploads")


class ModelStatus(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False) # "KNN", "ResNet18", "Fusion"
    version = Column(String, nullable=False)
    accuracy = Column(Float, nullable=True)
    f1 = Column(Float, nullable=True)
    is_active = Column(Boolean, default=True)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, index=True)
    epoch = Column(Integer, nullable=False)
    loss = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    val_loss = Column(Float, nullable=False)
    val_accuracy = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
