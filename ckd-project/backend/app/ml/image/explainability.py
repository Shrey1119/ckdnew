import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pathlib import Path
from app.config.config import settings
from app.ml.image.resnet_model import ResNet18KidneyClassifier

CLASS_NAMES = ["Normal", "Cyst", "Tumor", "Stone"]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        outputs = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
            
        self.model.zero_grad()
        # Target probability output
        score = outputs[0, class_idx]
        score.backward()
        
        # Calculate pooled gradients
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pool gradients
        weights = torch.mean(gradients, dim=(1, 2))
        
        # Linear combination
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU to retain positive influences
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam, class_idx

def generate_gradcam_image(image_path: str, save_filename: str = None) -> dict:
    """
    Load ResNet18, run Grad-CAM on the image, and save the overlay image.
    """
    model_path = Path(settings.SAVED_MODELS_DIR) / "phase2_resnet18.pth"
    if not model_path.exists():
        return {"status": "error", "message": "ResNet18 weights not found. Run training first."}
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = ResNet18KidneyClassifier(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            return {"status": "error", "message": f"Could not read image from {image_path}"}
            
        # Keep copy of size
        orig_height, orig_width = orig_img.shape[:2]
        
        # Preprocess PIL image for PyTorch
        pil_img = Image.fromarray(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        # Perform Grad-CAM on layer4 (specifically the last bottleneck/residual block)
        target_layer = model.resnet.layer4[-1]
        gradcam = GradCAM(model, target_layer)
        
        heatmap_mask, predicted_class = gradcam.generate_heatmap(input_tensor)
        
        # Resize heatmap mask to original dimensions
        heatmap_resized = cv2.resize(heatmap_mask, (orig_width, orig_height))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        
        # Apply colormap Jet
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # Superimpose original and heatmap (0.6 original + 0.4 heatmap)
        overlay = cv2.addWeighted(orig_img, 0.6, color_heatmap, 0.4, 0)
        
        # Save output
        gradcam_dir = Path(settings.UPLOAD_DIR) / "gradcam"
        os.makedirs(gradcam_dir, exist_ok=True)
        
        if not save_filename:
            save_filename = f"gradcam_{Path(image_path).stem}.png"
            
        output_filepath = gradcam_dir / save_filename
        cv2.imwrite(str(output_filepath), overlay)
        
        # Return path relative to server root
        relative_url = f"/uploads/gradcam/{save_filename}"
        
        return {
            "status": "success",
            "predicted_class": CLASS_NAMES[predicted_class],
            "class_idx": predicted_class,
            "gradcam_url": relative_url,
            "saved_path": str(output_filepath)
        }
    except Exception as e:
        return {"status": "error", "message": f"Grad-CAM generation failed: {str(e)}"}
