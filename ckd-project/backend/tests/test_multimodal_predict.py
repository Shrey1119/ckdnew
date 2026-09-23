import os
import sys
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
client = TestClient(app)
from app.ml.fusion_model_loader import fusion_model_loader
from app.services import auth_service
from app.database.connection import SessionLocal
from app.database.models import User

def get_auth_headers(role: str = "doctor"):
    """Helper to get JWT token headers for authenticated routes."""
    db = SessionLocal()
    user = db.query(User).filter(User.username == role).first()
    if not user:
        user = User(
            username=role,
            hashed_password=auth_service.get_password_hash(f"{role}password"),
            role=role
        )
        db.add(user)
        db.commit()
    db.close()
    token = auth_service.create_access_token(data={"sub": role, "role": role})
    return {"Authorization": f"Bearer {token}"}

def test_fusion_model_resolution():
    """Verify Phase 3 fusion model weights resolve and can be loaded."""
    path = fusion_model_loader.resolve_fusion_model_path()
    assert path is not None, "Phase 3 fusion model path could not be resolved"
    assert path.exists(), f"Model file does not exist at {path}"
    model = fusion_model_loader.get_model()
    assert model is not None, "Failed to load fusion model instance"

def test_model_comparison_endpoint():
    """Verify GET /api/predictions/compare returns benchmark metrics for all models."""
    headers = get_auth_headers("doctor")
    response = client.get("/api/predictions/compare", headers=headers)
    assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
    data = response.json()
    assert "best_fusion_model" in data
    assert "benchmarks" in data
    benchmarks = data["benchmarks"]
    assert "Phase 1 KNN (Tabular)" in benchmarks
    assert "Phase 2 ResNet (Image)" in benchmarks
    assert "Cross-Attention Fusion" in benchmarks
    assert benchmarks["Cross-Attention Fusion"]["accuracy"] >= 0.9

def test_model_comparison_alias():
    """Verify alias GET /api/predict/compare works identically."""
    headers = get_auth_headers("researcher")
    response = client.get("/api/predict/compare", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "benchmarks" in data

def test_multimodal_predict_with_file_upload():
    """Test POST /api/predictions/predict-multimodal with multipart clinical data + CT scan."""
    # Find sample image
    dataset_dir = Path(backend_dir).parents[1] / "kidney_images"
    sample_images = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
    assert len(sample_images) > 0, "No sample CT scan found for test"
    sample_img = sample_images[0]

    patient_payload = {
        "age": 55.0,
        "bp": 80.0,
        "sg": 1.015,
        "al": 2.0,
        "sc": 3.4,
        "bgr": 180.0,
        "hemo": 10.5
    }

    headers = get_auth_headers("doctor")
    with open(sample_img, "rb") as f:
        response = client.post(
            "/api/predictions/predict-multimodal",
            data={
                "patient_data": json.dumps(patient_payload),
                "fusion_type": "cross_attention"
            },
            files={"file": (sample_img.name, f, "image/jpeg")},
            headers=headers
        )

    assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in [0, 1]
    assert "probability" in data
    assert "confidence" in data
    assert "risk_level" in data
    assert "image_class" in data
    assert data["image_class"] in ["Normal", "Cyst", "Tumor", "Stone"]

def test_multimodal_predict_with_image_path():
    """Test POST /api/predict/multimodal using referenced server image_path."""
    dataset_dir = Path(backend_dir).parents[1] / "kidney_images"
    sample_images = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
    sample_img = sample_images[0]

    patient_payload = {
        "age": 45.0,
        "bp": 70.0,
        "sg": 1.025,
        "al": 0.0,
        "sc": 0.8,
        "bgr": 95.0,
        "hemo": 15.5
    }

    headers = get_auth_headers("doctor")
    response = client.post(
        "/api/predict/multimodal",
        data={
            "patient_data": json.dumps(patient_payload),
            "image_path": str(sample_img),
            "fusion_type": "cross_attention"
        },
        headers=headers
    )

    assert response.status_code == 200, f"Error {response.status_code}: {response.text}"
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

def test_predictions_history_alias():
    """Test GET /api/predictions/history returns paginated database records."""
    headers = get_auth_headers("doctor")
    response = client.get("/api/predictions/history?page=1&size=5", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "total" in data
    assert "items" in data
    assert isinstance(data["items"], list)

def test_predict_optimized_endpoint():
    """Test POST /api/predictions/predict-optimized (Phase 4 preview)."""
    dataset_dir = Path(backend_dir).parents[1] / "kidney_images"
    sample_images = list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.png"))
    sample_img = sample_images[0]

    patient_payload = {
        "age": 60.0,
        "bp": 90.0,
        "sc": 4.5,
        "bgr": 210.0
    }

    headers = get_auth_headers("doctor")
    response = client.post(
        "/api/predictions/predict-optimized",
        data={
            "patient_data": json.dumps(patient_payload),
            "image_path": str(sample_img)
        },
        headers=headers
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
