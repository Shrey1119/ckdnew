"""Standalone CLI for Multimodal CKD Prediction (Phase 3).

Allows evaluating combined clinical tabular data and kidney CT scan images
using the best trained Phase 3 Multimodal Fusion model.

Usage Examples:
  1. Test by dataset row index:
     python predict_multimodal_cli.py --csv-row 0

  2. Test with explicit image and clinical features:
     python predict_multimodal_cli.py --image "kidney_images/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (55).jpg" --sc 3.8 --al 3.0 --bgr 210 --bp 85

  3. Output in raw JSON:
     python predict_multimodal_cli.py --csv-row 15 --json
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

# Set UTF-8 output encoding for Windows terminals
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Import models from phase3 pipeline
from phase3_multimodal_fusion import (
    ConcatenationFusionModel,
    WeightedAverageFusionModel,
    CrossModalAttentionFusionModel
)

DEFAULT_FUSION_MODEL_PATH = os.path.join("model_artifacts", "phase3_fusion_model.pth")
DEFAULT_RESNET_PATH = os.path.join("model_artifacts", "phase2_resnet.pth")
DEFAULT_DATASET_CSV = "kidney_multimodal_dataset_FIXED.csv"
CT_CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]


def get_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_image_feature_extractor(resnet_path: str, device: str) -> Tuple[nn.Module, nn.Module]:
    """Loads ResNet-18 to get both 4-class classification and 512-dim features."""
    resnet = models.resnet18(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 4)

    if os.path.exists(resnet_path):
        ckpt = torch.load(resnet_path, map_location=device, weights_only=False)
        resnet.load_state_dict(ckpt["model_state_dict"])
    
    resnet.to(device)
    resnet.eval()

    # Penultimate layer feature extractor
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    return resnet, feature_extractor


def load_fusion_model(model_path: str, device: str) -> Tuple[nn.Module, Any, List[str], Dict[str, Any]]:
    """Loads the trained best fusion model, preprocessor, and metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fusion model checkpoint '{model_path}' not found! "
            "Please train Phase 3 first: python phase3_multimodal_fusion.py"
        )

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    arch_name = ckpt.get("model_architecture", "Cross-Attention Fusion")
    tabular_dim = ckpt["tabular_dim"]
    feature_columns = ckpt["feature_columns"]
    preprocessor = ckpt["preprocessor"]
    config = ckpt.get("config", {})

    if arch_name == "Concatenation Fusion":
        model = ConcatenationFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)
    elif arch_name == "Weighted Fusion":
        model = WeightedAverageFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)
    else:
        model = CrossModalAttentionFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, preprocessor, feature_columns, ckpt


def predict_multimodal(
    tabular_dict: Dict[str, Any],
    image_path: str,
    device: str = "cpu"
) -> Dict[str, Any]:
    """Runs complete end-to-end multimodal inference."""
    # 1. Load models
    fusion_model, preprocessor, feature_cols, ckpt = load_fusion_model(DEFAULT_FUSION_MODEL_PATH, device)
    resnet_classifier, resnet_extractor = load_image_feature_extractor(DEFAULT_RESNET_PATH, device)

    # 2. Process image
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    img = Image.open(image_path).convert("RGB")
    img_tensor = get_image_transform()(img).unsqueeze(0).to(device)

    with torch.no_grad():
        # Get CT scan class probabilities
        ct_logits = resnet_classifier(img_tensor)
        ct_probs = F.softmax(ct_logits, dim=1)[0].cpu().numpy()
        ct_pred_idx = int(ct_probs.argmax())
        ct_pred_class = CT_CLASSES[ct_pred_idx]
        ct_confidence = float(ct_probs[ct_pred_idx])

        # Get 512-dim feature embedding
        img_feat = resnet_extractor(img_tensor).flatten(1)

    # 3. Process Tabular Data
    row_df = pd.DataFrame([tabular_dict])
    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = np.nan
    row_df = row_df[feature_cols]

    tab_proc = preprocessor.transform(row_df)
    tab_tensor = torch.tensor(tab_proc, dtype=torch.float32).to(device)

    # 4. Predict via Fusion Model
    with torch.no_grad():
        logits, meta = fusion_model(tab_tensor, img_feat)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    p_ckd = float(probs[1])
    pred_class = 1 if p_ckd >= 0.5 else 0
    confidence_val = p_ckd if pred_class == 1 else (1.0 - p_ckd)

    if confidence_val >= 0.85:
        conf_label = "HIGH"
    elif confidence_val >= 0.65:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW"

    if p_ckd >= 0.70:
        risk_label = "HIGH RISK"
    elif p_ckd >= 0.35:
        risk_label = "MODERATE RISK"
    else:
        risk_label = "LOW RISK"

    return {
        "status": "success",
        "fusion_architecture": ckpt.get("model_architecture", "Cross-Attention Fusion"),
        "prediction": pred_class,
        "prediction_label": "CKD Detected" if pred_class == 1 else "Normal / Not CKD",
        "ckd_probability": round(p_ckd, 4),
        "confidence_score": round(confidence_val, 4),
        "confidence_level": conf_label,
        "risk_level": risk_label,
        "image_analysis": {
            "image_path": image_path,
            "predicted_ct_finding": ct_pred_class,
            "ct_confidence": round(ct_confidence, 4),
            "probabilities": {CT_CLASSES[i]: round(float(ct_probs[i]), 4) for i in range(4)}
        },
        "modality_metadata": meta
    }


def main():
    parser = argparse.ArgumentParser(description="Multimodal CKD Prediction Tool (Phase 3)")
    parser.add_argument("--csv-row", type=int, default=None, help="Index of row in dataset CSV to evaluate")
    parser.add_argument("--image", type=str, default=None, help="Path to CT scan image")
    parser.add_argument("--tabular-json", type=str, default=None, help="Path to JSON file or raw JSON string of features")
    parser.add_argument("--json", action="store_true", help="Output results in pure JSON format")
    
    # Clinical flags
    parser.add_argument("--age", type=float, default=None)
    parser.add_argument("--bp", type=float, default=None)
    parser.add_argument("--sg", type=float, default=None)
    parser.add_argument("--al", type=float, default=None)
    parser.add_argument("--sc", type=float, default=None)
    parser.add_argument("--bgr", type=float, default=None)
    parser.add_argument("--hemo", type=float, default=None)

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tab_data = {}
    img_path = args.image

    # Mode A: Load by CSV row
    if args.csv_row is not None:
        if not os.path.exists(DEFAULT_DATASET_CSV):
            print(f"Error: Dataset CSV '{DEFAULT_DATASET_CSV}' not found.")
            sys.exit(1)
        df = pd.read_csv(DEFAULT_DATASET_CSV)
        if args.csv_row < 0 or args.csv_row >= len(df):
            print(f"Error: csv-row index {args.csv_row} out of range [0, {len(df)-1}].")
            sys.exit(1)
        row = df.iloc[args.csv_row]
        img_path = row["image_path"]
        tab_data = row.drop(["id", "image_path", "classification"], errors="ignore").to_dict()
        true_label = int(row["classification"])
    else:
        true_label = None
        if args.tabular_json:
            if os.path.exists(args.tabular_json):
                with open(args.tabular_json, "r") as f:
                    tab_data = json.load(f)
            else:
                tab_data = json.loads(args.tabular_json)

        # CLI overrides
        for k in ["age", "bp", "sg", "al", "sc", "bgr", "hemo"]:
            val = getattr(args, k)
            if val is not None:
                tab_data[k] = val

        if not img_path:
            # Fallback to first image from dataset if none supplied
            if os.path.exists(DEFAULT_DATASET_CSV):
                df = pd.read_csv(DEFAULT_DATASET_CSV)
                img_path = df.iloc[0]["image_path"]
            else:
                print("Error: No image provided and dataset not found.")
                sys.exit(1)

    try:
        res = predict_multimodal(tab_data, img_path, device=device)
        if true_label is not None:
            res["ground_truth"] = true_label
            res["ground_truth_label"] = "CKD Detected" if true_label == 1 else "Normal / Not CKD"

        if args.json:
            print(json.dumps(res, indent=2))
        else:
            print("\n" + "=" * 65)
            print("🔬 KIDNEY AI — MULTIMODAL CKD DIAGNOSTIC REPORT (PHASE 3)")
            print("=" * 65)
            print(f"Fusion Architecture : {res['fusion_architecture']}")
            print(f"Input Image         : {res['image_analysis']['image_path']}")
            print(f"CT Scan Finding     : {res['image_analysis']['predicted_ct_finding']} ({res['image_analysis']['ct_confidence']*100:.1f}% confidence)")
            print("-" * 65)
            print(f"FINAL PREDICTION    : {res['prediction_label'].upper()}")
            if "ground_truth_label" in res:
                match_str = "MATCH" if res["prediction"] == res["ground_truth"] else "MISMATCH"
                print(f"Ground Truth        : {res['ground_truth_label'].upper()} [{match_str}]")
            print(f"CKD Probability     : {res['ckd_probability']*100:.2f}%")
            print(f"Confidence Level    : {res['confidence_level']} ({res['confidence_score']*100:.1f}%)")
            print(f"Clinical Risk Tier  : {res['risk_level']}")
            print("-" * 65)
            print("Modality Breakdown:")
            for cls_name, prob in res['image_analysis']['probabilities'].items():
                print(f"  - CT {cls_name:<7}: {prob*100:>5.1f}%")
            print("=" * 65 + "\n")

    except Exception as e:
        print(f"Error during multimodal prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
