import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from app.config.config import settings
from app.ml.image_model_loader import image_model_loader
from app.ml.image.explainability import CLASS_NAMES as CT_CLASSES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neural Architectures (Mirroring Phase 3 Trained Checkpoint)
# ---------------------------------------------------------------------------
class TabularEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ImageProjector(nn.Module):
    def __init__(self, input_dim: int = 512, embed_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossModalAttentionFusionModel(nn.Module):
    def __init__(self, tabular_dim: int, image_dim: int = 512, embed_dim: int = 128, num_classes: int = 2, num_heads: int = 4):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, embed_dim)
        self.image_projector = ImageProjector(image_dim, embed_dim)
        
        self.cross_tab_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.cross_img_tab = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.ln_tab = nn.LayerNorm(embed_dim)
        self.ln_img = nn.LayerNorm(embed_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, tab_x: torch.Tensor, img_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z_tab = self.tabular_encoder(tab_x).unsqueeze(1)
        z_img = self.image_projector(img_feat).unsqueeze(1)
        
        attn_tab_out, attn_weights_tab = self.cross_tab_img(query=z_tab, key=z_img, value=z_img)
        attn_img_out, attn_weights_img = self.cross_img_tab(query=z_img, key=z_tab, value=z_tab)
        
        z_tab_fused = self.ln_tab(z_tab + attn_tab_out).squeeze(1)
        z_img_fused = self.ln_img(z_img + attn_img_out).squeeze(1)
        
        fused = torch.cat([z_tab_fused, z_img_fused], dim=1)
        logits = self.classifier(fused)
        
        meta = {
            "attn_tab_weight": float(attn_weights_tab.mean().item()),
            "attn_img_weight": float(attn_weights_img.mean().item()),
            "fusion_type": "cross_attention"
        }
        return logits, meta


class ConcatenationFusionModel(nn.Module):
    def __init__(self, tabular_dim: int, image_dim: int = 512, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim + image_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, tab_x: torch.Tensor, img_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z_tab = self.tabular_encoder(tab_x)
        z_fused = torch.cat([z_tab, img_feat], dim=1)
        logits = self.classifier(z_fused)
        meta = {"fusion_type": "concatenation"}
        return logits, meta


class WeightedAverageFusionModel(nn.Module):
    def __init__(self, tabular_dim: int, image_dim: int = 512, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, embed_dim)
        self.image_projector = ImageProjector(image_dim, embed_dim)
        self.tab_head = nn.Linear(embed_dim, num_classes)
        self.img_head = nn.Linear(embed_dim, num_classes)
        self.gate_param = nn.Parameter(torch.tensor([0.0]))

    def forward(self, tab_x: torch.Tensor, img_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z_tab = self.tabular_encoder(tab_x)
        z_img = self.image_projector(img_feat)
        logits_tab = self.tab_head(z_tab)
        logits_img = self.img_head(z_img)
        alpha = torch.sigmoid(self.gate_param)
        alpha = 0.05 + 0.90 * alpha
        fused_logits = (alpha * logits_tab) + ((1.0 - alpha) * logits_img)
        meta = {
            "tabular_weight": float(alpha.item()),
            "image_weight": float((1.0 - alpha).item()),
            "fusion_type": "weighted_average"
        }
        return fused_logits, meta


# ---------------------------------------------------------------------------
# Fusion Model Loader Manager
# ---------------------------------------------------------------------------
class FusionModelLoader:
    def __init__(self):
        self._model = None
        self._preprocessor = None
        self._feature_columns = None
        self._checkpoint_meta = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def resolve_fusion_model_path(self) -> Optional[Path]:
        """Locates trained Phase 3 fusion model checkpoint."""
        base_dir = Path(__file__).resolve().parent
        candidates = [
            Path(settings.SAVED_MODELS_DIR) / "phase3_fusion_model.pth",
            base_dir.parents[3] / "model_artifacts" / "phase3_fusion_model.pth",
            Path("../../model_artifacts/phase3_fusion_model.pth"),
            Path("../model_artifacts/phase3_fusion_model.pth"),
            Path("model_artifacts/phase3_fusion_model.pth")
        ]
        for p in candidates:
            if p.exists() and p.is_file():
                return p
        return None

    def get_model(self) -> Optional[nn.Module]:
        """Lazy loads the multimodal fusion model."""
        if self._model is not None:
            return self._model

        model_path = self.resolve_fusion_model_path()
        if not model_path:
            logger.warning("Phase 3 fusion model checkpoint not found on disk.")
            return None

        try:
            logger.info(f"Loading Phase 3 Fusion model from {model_path}...")
            ckpt = torch.load(model_path, map_location=self._device, weights_only=False)
            arch_name = ckpt.get("model_architecture", "Cross-Attention Fusion")
            tabular_dim = ckpt["tabular_dim"]
            self._preprocessor = ckpt["preprocessor"]
            self._feature_columns = ckpt["feature_columns"]
            self._checkpoint_meta = ckpt

            if arch_name == "Concatenation Fusion":
                model = ConcatenationFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)
            elif arch_name == "Weighted Fusion":
                model = WeightedAverageFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)
            else:
                model = CrossModalAttentionFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)

            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(self._device)
            model.eval()
            self._model = model
            logger.info(f"Phase 3 Fusion Model ({arch_name}) loaded successfully.")
            return self._model
        except Exception as e:
            logger.error(f"Failed to load Fusion model from {model_path}: {e}")
            return None

    def extract_image_features(self, image_path: str) -> Tuple[torch.Tensor, dict]:
        """Extracts ResNet-18 512-dim penultimate layer features and class probabilities."""
        resnet_model = image_model_loader.get_model()
        if resnet_model is None or not os.path.exists(image_path):
            return torch.zeros((1, 512)).to(self._device), {"Normal": 0.25, "Cyst": 0.25, "Tumor": 0.25, "Stone": 0.25}

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_tensor = self.transform(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            img_feat = resnet_model.extract_features(img_tensor)
            logits = resnet_model.resnet.fc(img_feat)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        probs_dict = {CT_CLASSES[i]: round(float(probs[i]), 4) for i in range(len(CT_CLASSES))}
        return img_feat, probs_dict

    def predict(self, tabular_dict: dict, image_path: str, fusion_type: str = "attention", tabular_weight: float = None, image_weight: float = None) -> dict:
        """Runs end-to-end multimodal prediction."""
        model = self.get_model()
        img_feat, ct_probs = self.extract_image_features(image_path)
        pred_ct_class = max(ct_probs, key=ct_probs.get)
        ct_confidence = ct_probs[pred_ct_class]

        if model is None or self._preprocessor is None:
            # Fallback to heuristic late fusion if weights missing
            logger.warning("Fusion model not ready, using fallback estimation.")
            p_img_ckd = float(1.0 - ct_probs.get("Normal", 0.7))
            return {
                "prediction": 1 if p_img_ckd >= 0.5 else 0,
                "probability": round(p_img_ckd, 4),
                "label": "ckd" if p_img_ckd >= 0.5 else "not_ckd",
                "confidence": "medium",
                "risk_level": "medium",
                "tabular_prob": 0.5,
                "image_prob": round(p_img_ckd, 4),
                "tabular_weight": 0.6,
                "image_weight": 0.4,
                "fusion_type": "fallback",
                "image_class": pred_ct_class,
                "ct_confidence_probs": ct_probs,
                "meta": {}
            }

        # 1. Transform tabular features
        row_df = pd.DataFrame([tabular_dict])
        for col in self._feature_columns:
            if col not in row_df.columns:
                row_df[col] = np.nan
        row_df = row_df[self._feature_columns]

        tab_proc = self._preprocessor.transform(row_df)
        tab_tensor = torch.tensor(tab_proc, dtype=torch.float32).to(self._device)

        # 2. Run Forward Pass
        with torch.no_grad():
            logits, meta = model(tab_tensor, img_feat)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        p_ckd = float(probs[1])
        prediction = 1 if p_ckd >= 0.5 else 0
        confidence_score = p_ckd if prediction == 1 else (1.0 - p_ckd)

        if confidence_score >= 0.85:
            confidence = "high"
        elif confidence_score >= 0.65:
            confidence = "medium"
        else:
            confidence = "low"

        if p_ckd >= 0.70:
            risk_level = "high"
        elif p_ckd >= 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Image CKD probability proxy
        img_ckd_prob = round(float(1.0 - ct_probs.get("Normal", 0.5)), 4)

        return {
            "prediction": prediction,
            "probability": round(p_ckd, 4),
            "label": "ckd" if prediction == 1 else "not_ckd",
            "confidence": confidence,
            "risk_level": risk_level,
            "tabular_prob": round(p_ckd, 4),
            "image_prob": img_ckd_prob,
            "tabular_weight": meta.get("tabular_weight", 0.5),
            "image_weight": meta.get("image_weight", 0.5),
            "fusion_type": meta.get("fusion_type", fusion_type),
            "image_class": pred_ct_class,
            "ct_confidence": round(float(ct_confidence), 4),
            "ct_confidence_probs": ct_probs,
            "meta": meta
        }

fusion_model_loader = FusionModelLoader()
