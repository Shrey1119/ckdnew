from pydantic import BaseModel, Field

class ImagePredictRequest(BaseModel):
    image_path: str | None = Field(default=None, description="Path to image on server disk")

class ImagePredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Predicted scan class (Normal, Cyst, Tumor, Stone)")
    confidence_probs: dict[str, float] = Field(..., description="Probability score for each of the 4 classes")
    confidence: str = Field(..., description="Confidence tier: high, medium, low")
    gradcam_url: str | None = Field(default=None, description="URL of generated Grad-CAM heatmap visualization")

class ImageMetricsResponse(BaseModel):
    accuracy: float
    classes: list[str]
    model_name: str = "ResNet-18"
