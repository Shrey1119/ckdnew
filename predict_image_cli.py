"""Standalone CLI tool for CT Kidney Image Classification (Phase 2).

Usage:
  python predict_image_cli.py --image path/to/scan.png
  python predict_image_cli.py --dir path/to/scans_dir/
"""

import os
import sys
import json
import argparse
import glob
from typing import Dict, Any, Tuple

from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models

MODEL_PATH = os.path.join("model_artifacts", "phase2_resnet.pth")
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]

def get_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path: str, device: str) -> Tuple[nn.Module, list]:
    if not os.path.exists(model_path):
        print(f"❌ Error: Model checkpoint '{model_path}' not found!")
        print("Please train the Phase 2 model first by running: python phase2_image_baseline.py")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', CLASSES)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, classes

def predict_single_image(image_path: str, model: nn.Module, classes: list, device: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image {image_path}: {e}"}

    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    top_idx = int(probabilities.argmax())
    predicted_class = classes[top_idx]
    confidence = float(probabilities[top_idx])

    prob_dict = {classes[i]: round(float(probabilities[i]), 4) for i in range(len(classes))}

    return {
        "image_path": image_path,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": prob_dict
    }

def main():
    parser = argparse.ArgumentParser(description="CKD CT Scan Image Classification CLI")
    parser.add_argument("--image", type=str, help="Path to a single CT scan image file")
    parser.add_argument("--dir", type=str, help="Path to a directory containing CT scan images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Loading Phase 2 ResNet-18 model on device: {device}...")
    model, classes = load_model(MODEL_PATH, device)

    if args.image:
        result = predict_single_image(args.image, model, classes, device)
        print("\n" + "="*50)
        print("[+] CT SCAN CLASSIFICATION RESULT")
        print("="*50)
        if "error" in result:
            print(f"[!] {result['error']}")
        else:
            print(f"[-] Image: {result['image_path']}")
            print(f"[-] Prediction: {result['prediction']}")
            print(f"[-] Confidence: {result['confidence_percentage']}")
            print("\n[*] Class Probabilities:")
            for cls, prob in result['probabilities'].items():
                bar = "#" * int(prob * 20)
                print(f"   - {cls:8s}: {prob*100:6.2f}% {bar}")
        print("="*50)

    elif args.dir:
        image_files = glob.glob(os.path.join(args.dir, "*.[jJ][pP][gG]")) + \
                      glob.glob(os.path.join(args.dir, "*.[pP][nN][gG]"))
        if not image_files:
            print(f"❌ No JPG/PNG image files found in '{args.dir}'")
            sys.exit(1)

        print(f"\n📂 Processing {len(image_files)} images from '{args.dir}'...\n")
        print(f"{'Image File':<35} | {'Prediction':<10} | {'Confidence':<10}")
        print("-" * 62)

        for img_path in image_files:
            res = predict_single_image(img_path, model, classes, device)
            fname = os.path.basename(img_path)
            if len(fname) > 33:
                fname = fname[:30] + "..."
            if "error" in res:
                print(f"{fname:<35} | ERROR      | N/A")
            else:
                print(f"{fname:<35} | {res['prediction']:<10} | {res['confidence_percentage']:<10}")

    else:
        print("ℹ️ Please provide either --image <path> or --dir <path>")
        print("Example: python predict_image_cli.py --image sample_scan.png")

if __name__ == "__main__":
    main()
