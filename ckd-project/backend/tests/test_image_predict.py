import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from main import app
client = TestClient(app)
from app.ml.image_model_loader import image_model_loader
from app.services import auth_service

from app.database.connection import SessionLocal
from app.database.models import User

def get_auth_headers():
    """Helper to get JWT token headers for doctor role"""
    db = SessionLocal()
    user = db.query(User).filter(User.username == "doctor").first()
    if not user:
        user = User(username="doctor", hashed_password=auth_service.get_password_hash("doctorpassword"), role="doctor")
        db.add(user)
        db.commit()
    db.close()
    token = auth_service.create_access_token(data={"sub": "doctor", "role": "doctor"})
    return {"Authorization": f"Bearer {token}"}

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_loader_resolution():
    model_path = image_model_loader.resolve_model_path()
    assert model_path is not None
    assert model_path.exists()
    assert "phase2_resnet.pth" in str(model_path) or "phase2_resnet18.pth" in str(model_path)

def test_image_predict_endpoint_with_sample_image():
    # Find a sample CT scan from kidney_images directory
    dataset_dir = Path(backend_dir).parents[1] / "kidney_images"
    sample_images = list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpg"))
    
    if not sample_images:
        print("Skipping image test: No sample images found in kidney_images directory")
        return

    sample_img = sample_images[0]
    headers = get_auth_headers()

    with open(sample_img, "rb") as f:
        response = client.post(
            "/api/predict/image",
            files={"file": (sample_img.name, f, "image/png")},
            headers=headers
        )

    if response.status_code != 200:
        print(f"Error status {response.status_code}: {response.text}")
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Normal", "Cyst", "Tumor", "Stone"]
    assert "confidence_probs" in data
    assert len(data["confidence_probs"]) == 4
    assert "confidence" in data

def test_image_predict_alias_endpoint():
    dataset_dir = Path(backend_dir).parents[1] / "kidney_images"
    sample_images = list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpg"))
    
    if not sample_images:
        print("Skipping image test: No sample images found in kidney_images directory")
        return

    sample_img = sample_images[0]
    headers = get_auth_headers()

    with open(sample_img, "rb") as f:
        response = client.post(
            "/api/predictions/predict-image",
            files={"file": (sample_img.name, f, "image/png")},
            headers=headers
        )

    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data
    assert "confidence_probs" in data

if __name__ == "__main__":
    print("Running Member C Sprint 1 Backend Tests...")
    test_health_check()
    print("[OK] Health check passed")
    test_model_loader_resolution()
    print("[OK] Model loader resolution passed")
    test_image_predict_endpoint_with_sample_image()
    print("[OK] Image predict endpoint passed")
    test_image_predict_alias_endpoint()
    print("[OK] Image predict alias endpoint passed")
    print("\nALL MEMBER C SPRINT 1 TESTS PASSED SUCCESSFULLY!")
