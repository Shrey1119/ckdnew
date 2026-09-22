"""Phase 3 Multimodal Fusion Pipeline.

Combines Tabular Clinical Data (UCI CKD features) with CT Scan Kidney Images (ResNet-18)
using three advanced fusion architectures:
1. Concatenation Fusion (Feature Concatenation + Deep MLP Classifier)
2. Weighted Average Fusion (Learnable Modality Gating & Decision Calibration)
3. Cross-Attention Fusion (Bi-directional Cross-Modal Attention between Clinical & Imaging Tokens)

Outputs:
- model_artifacts/phase3_fusion_model.pth (Trained best fusion model)
- model_artifacts/phase3_metrics.json (Comprehensive benchmarks & comparative metrics)
- model_artifacts/phase3_comparison.png (Multi-panel visual benchmark charts)
"""

import os
import sys
import json
import time
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Phase3MultimodalFusion")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Phase3Config:
    data_csv_path: str = "kidney_multimodal_dataset_FIXED.csv"
    resnet_path: str = os.path.join("model_artifacts", "phase2_resnet.pth")
    knn_model_path: str = os.path.join("model_artifacts", "phase1_knn_model.joblib")
    artifacts_dir: str = "model_artifacts"
    best_model_path: str = os.path.join("model_artifacts", "phase3_fusion_model.pth")
    metrics_path: str = os.path.join("model_artifacts", "phase3_metrics.json")
    comparison_plot_path: str = os.path.join("model_artifacts", "phase3_comparison.png")
    
    # Model dimensions
    tabular_embed_dim: int = 128
    image_embed_dim: int = 512
    projected_dim: int = 128
    num_classes: int = 2  # 0 = Not CKD, 1 = CKD
    
    # Training hyperparams
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 6
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tabular Preprocessing Pipeline
# ---------------------------------------------------------------------------
def build_tabular_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
    """Builds and fits Scikit-Learn ColumnTransformer for tabular clinical features."""
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    
    cat_transformer_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]

    from sklearn.pipeline import Pipeline
    num_pipe = Pipeline(numeric_transformer)
    cat_pipe = Pipeline(cat_transformer_steps)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ]
    )
    preprocessor.fit(X)
    return preprocessor, list(X.columns)


# ---------------------------------------------------------------------------
# Neural Architectures
# ---------------------------------------------------------------------------
class TabularEncoder(nn.Module):
    """Dense encoder that maps preprocessed tabular clinical features to a latent vector."""
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
    """Projects 512-dim ResNet image embeddings to a common latent space."""
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


class ConcatenationFusionModel(nn.Module):
    """Strategy 1: Early Feature Concatenation Fusion."""
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
        meta = {
            "tabular_norm": float(torch.norm(z_tab, dim=1).mean().item()),
            "image_norm": float(torch.norm(img_feat, dim=1).mean().item()),
            "fusion_type": "concatenation"
        }
        return logits, meta


class WeightedAverageFusionModel(nn.Module):
    """Strategy 2: Modality-Gated Weighted Fusion with Learnable Balance."""
    def __init__(self, tabular_dim: int, image_dim: int = 512, embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, embed_dim)
        self.image_projector = ImageProjector(image_dim, embed_dim)
        
        # Modality classification branches
        self.tab_head = nn.Linear(embed_dim, num_classes)
        self.img_head = nn.Linear(embed_dim, num_classes)
        
        # Learnable gating parameter initialized at 0.5 (logit 0.0)
        self.gate_param = nn.Parameter(torch.tensor([0.0]))

    def forward(self, tab_x: torch.Tensor, img_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        z_tab = self.tabular_encoder(tab_x)
        z_img = self.image_projector(img_feat)
        
        logits_tab = self.tab_head(z_tab)
        logits_img = self.img_head(z_img)
        
        # Clamped gating weight alpha in [0.05, 0.95] for numerical stability
        alpha = torch.sigmoid(self.gate_param)
        alpha = 0.05 + 0.90 * alpha  # bounded range
        
        fused_logits = (alpha * logits_tab) + ((1.0 - alpha) * logits_img)
        
        meta = {
            "tabular_weight": float(alpha.item()),
            "image_weight": float((1.0 - alpha).item()),
            "tabular_logits": logits_tab.detach().cpu().tolist(),
            "image_logits": logits_img.detach().cpu().tolist(),
            "fusion_type": "weighted_average"
        }
        return fused_logits, meta


class CrossModalAttentionFusionModel(nn.Module):
    """Strategy 3: Cross-Modal Attention Fusion.
    Allows clinical tabular tokens and imaging visual features to dynamically attend to each other.
    """
    def __init__(self, tabular_dim: int, image_dim: int = 512, embed_dim: int = 128, num_classes: int = 2, num_heads: int = 4):
        super().__init__()
        self.tabular_encoder = TabularEncoder(tabular_dim, embed_dim)
        self.image_projector = ImageProjector(image_dim, embed_dim)
        
        # Multihead Cross Attention: Tabular attends to Image
        self.cross_tab_img = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        # Multihead Cross Attention: Image attends to Tabular
        self.cross_img_tab = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        self.ln_tab = nn.LayerNorm(embed_dim)
        self.ln_img = nn.LayerNorm(embed_dim)
        
        # Fusion classification head
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
        z_tab = self.tabular_encoder(tab_x).unsqueeze(1)    # (B, 1, 128)
        z_img = self.image_projector(img_feat).unsqueeze(1) # (B, 1, 128)
        
        # Cross Attention passes
        attn_tab_out, attn_weights_tab = self.cross_tab_img(query=z_tab, key=z_img, value=z_img)
        attn_img_out, attn_weights_img = self.cross_img_tab(query=z_img, key=z_tab, value=z_tab)
        
        # Residual connections + LayerNorm
        z_tab_fused = self.ln_tab(z_tab + attn_tab_out).squeeze(1) # (B, 128)
        z_img_fused = self.ln_img(z_img + attn_img_out).squeeze(1) # (B, 128)
        
        fused = torch.cat([z_tab_fused, z_img_fused], dim=1) # (B, 256)
        logits = self.classifier(fused)
        
        meta = {
            "attn_tab_weight": float(attn_weights_tab.mean().item()),
            "attn_img_weight": float(attn_weights_img.mean().item()),
            "fusion_type": "cross_attention"
        }
        return logits, meta


# ---------------------------------------------------------------------------
# Dataset & Image Feature Extraction
# ---------------------------------------------------------------------------
class PrecomputedMultimodalDataset(Dataset):
    """Dataset holding preprocessed tabular arrays, precomputed ResNet embeddings, and labels."""
    def __init__(self, tab_data: np.ndarray, img_feats: np.ndarray, labels: np.ndarray):
        self.tab_data = torch.tensor(tab_data, dtype=torch.float32)
        self.img_feats = torch.tensor(img_feats, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.tab_data[idx], self.img_feats[idx], self.labels[idx]


def extract_resnet_embeddings(image_paths: List[str], resnet_checkpoint_path: str, device: str) -> np.ndarray:
    """Extracts 512-dim penultimate layer feature embeddings using Member A's ResNet-18."""
    logger.info(f"Extracting image embeddings using ResNet checkpoint: {resnet_checkpoint_path}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load ResNet-18
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    
    if os.path.exists(resnet_checkpoint_path):
        checkpoint = torch.load(resnet_checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("  Loaded weights from Phase 2 checkpoint successfully.")
    else:
        logger.warning(f"  Checkpoint not found at {resnet_checkpoint_path}, using untrained ResNet18!")

    # Cut off final classification layer to extract 512-dim features
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    feature_extractor.to(device)
    feature_extractor.eval()

    embeddings = []
    batch_tensors = []
    batch_size = 64

    with torch.no_grad():
        for i, img_path in enumerate(image_paths):
            try:
                img = Image.open(img_path).convert("RGB")
                t = transform(img)
            except Exception as e:
                logger.warning(f"Could not load image at {img_path}: {e}. Using zero tensor.")
                t = torch.zeros(3, 224, 224)

            batch_tensors.append(t)

            if len(batch_tensors) == batch_size or i == len(image_paths) - 1:
                batch = torch.stack(batch_tensors).to(device)
                feats = feature_extractor(batch).flatten(1)
                embeddings.append(feats.cpu().numpy())
                batch_tensors = []

    embeddings_arr = np.concatenate(embeddings, axis=0)
    logger.info(f"Extracted {len(embeddings_arr)} image embeddings of shape {embeddings_arr.shape}")
    return embeddings_arr


# ---------------------------------------------------------------------------
# Training & Evaluation Engine
# ---------------------------------------------------------------------------
def train_fusion_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Phase3Config,
    model_name: str
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Trains a given fusion architecture with AdamW, lr scheduler, and early stopping."""
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for tab_b, img_b, y_b in train_loader:
            tab_b = tab_b.to(config.device)
            img_b = img_b.to(config.device)
            y_b = y_b.to(config.device)

            optimizer.zero_grad()
            logits, _ = model(tab_b, img_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(y_b)
            total_samples += len(y_b)

        epoch_train_loss = running_loss / total_samples

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for tab_b, img_b, y_b in val_loader:
                tab_b = tab_b.to(config.device)
                img_b = img_b.to(config.device)
                y_b = y_b.to(config.device)

                logits, _ = model(tab_b, img_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * len(y_b)

                preds = logits.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[{model_name}] Epoch {epoch:02d}/{config.epochs:02d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc * 100:.2f}%")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"[{model_name}] Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    return model, history


def evaluate_fusion_model(model: nn.Module, test_loader: DataLoader, device: str) -> Dict[str, Any]:
    """Computes full evaluation metrics on test dataset."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for tab_b, img_b, y_b in test_loader:
            tab_b = tab_b.to(device)
            img_b = img_b.to(device)

            logits, _ = model(tab_b, img_b)
            probs = F.softmax(logits, dim=1)[:, 1] # Probability of CKD
            preds = logits.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob
    }


# ---------------------------------------------------------------------------
# Baselines Evaluation (KNN and ResNet-18 Alone)
# ---------------------------------------------------------------------------
def evaluate_knn_baseline(X_test_df: pd.DataFrame, y_test: np.ndarray, knn_path: str) -> Dict[str, Any]:
    """Evaluates Phase 1 KNN baseline on test split."""
    if not os.path.exists(knn_path):
        logger.warning("KNN model not found, skipping baseline.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "roc_auc": 0.0, "confusion_matrix": []}

    knn_model = joblib.load(knn_path)
    y_pred = knn_model.predict(X_test_df)
    try:
        y_prob = knn_model.predict_proba(X_test_df)[:, 1]
    except Exception:
        y_prob = y_pred

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }


def evaluate_resnet_baseline(test_img_paths: List[str], y_test: np.ndarray, resnet_path: str, device: str) -> Dict[str, Any]:
    """Evaluates Phase 2 ResNet-18 alone mapped to CKD risk on test split."""
    if not os.path.exists(resnet_path):
        logger.warning("ResNet model not found, skipping baseline.")
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "roc_auc": 0.0, "confusion_matrix": []}

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    checkpoint = torch.load(resnet_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # In our dataset schema:
    # 0 = Normal -> CKD 0
    # 1 = Cyst, 2 = Tumor, 3 = Stone -> Non-normal pathology
    y_probs = []
    y_preds = []

    with torch.no_grad():
        for path in test_img_paths:
            try:
                img = Image.open(path).convert("RGB")
                t = transform(img).unsqueeze(0).to(device)
                logits = model(t)
                probs = F.softmax(logits, dim=1)[0].cpu().numpy()
                # Probability of CKD / pathology (1.0 - P(Normal))
                p_ckd = float(1.0 - probs[0])
            except Exception:
                p_ckd = 0.5

            y_probs.append(p_ckd)
            y_preds.append(1 if p_ckd >= 0.5 else 0)

    y_true = y_test
    y_pred = np.array(y_preds)
    y_prob = np.array(y_probs)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }


# ---------------------------------------------------------------------------
# Visualizations & Plot Generation
# ---------------------------------------------------------------------------
def generate_comparison_plots(
    benchmarks: Dict[str, Dict[str, Any]],
    best_cm: List[List[int]],
    best_model_name: str,
    histories: Dict[str, Dict[str, List[float]]],
    output_path: str
) -> None:
    """Generates comprehensive high-resolution 4-panel comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    # Panel 1: Multi-metric Bar Chart
    models_list = list(benchmarks.keys())
    metrics_names = ["accuracy", "f1_score", "roc_auc", "precision", "recall"]
    display_names = ["Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall"]
    
    x = np.arange(len(models_list))
    width = 0.15
    palette = ["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    for i, (m_key, m_disp) in enumerate(zip(metrics_names, display_names)):
        vals = [benchmarks[m][m_key] for m in models_list]
        axes[0, 0].bar(x + (i - 2) * width, vals, width, label=m_disp, color=palette[i])

    axes[0, 0].set_title("Model Benchmark Comparison across Modalities", fontsize=13, fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_list, rotation=15, ha='right', fontsize=10)
    axes[0, 0].set_ylim([0.0, 1.05])
    axes[0, 0].set_ylabel("Score", fontsize=11)
    axes[0, 0].legend(loc="lower right", framealpha=0.9)
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.5)

    # Panel 2: Best Model Confusion Matrix Heatmap
    sns.heatmap(
        best_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=["Not CKD (0)", "CKD (1)"],
        yticklabels=["Not CKD (0)", "CKD (1)"],
        ax=axes[0, 1],
        annot_kws={"size": 14, "weight": "bold"}
    )
    axes[0, 1].set_title(f"Best Fusion Model Confusion Matrix\n({best_model_name})", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel("Predicted Label", fontsize=11)
    axes[0, 1].set_ylabel("True Ground Truth", fontsize=11)

    # Panel 3: Validation Loss Across Epochs
    loss_colors = {"Concatenation Fusion": "#3b82f6", "Weighted Fusion": "#10b981", "Cross-Attention Fusion": "#8b5cf6"}
    for name, hist in histories.items():
        val_losses = hist.get("val_loss", [])
        if val_losses:
            axes[1, 0].plot(range(1, len(val_losses) + 1), val_losses, label=name, color=loss_colors.get(name, "#64748b"), linewidth=2.2)

    axes[1, 0].set_title("Validation Loss Progression Across Epochs", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("Validation Cross-Entropy Loss", fontsize=11)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel 4: Validation Accuracy Convergence
    for name, hist in histories.items():
        val_accs = [acc * 100 for acc in hist.get("val_acc", [])]
        if val_accs:
            axes[1, 1].plot(range(1, len(val_accs) + 1), val_accs, label=name, color=loss_colors.get(name, "#64748b"), linewidth=2.2)

    axes[1, 1].set_title("Validation Accuracy Convergence (%)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Validation Accuracy (%)", fontsize=11)
    axes[1, 1].set_ylim([70, 102])
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    logger.info(f"Saved benchmark comparison visualizations to {output_path}")


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_phase3_multimodal_fusion() -> None:
    """End-to-end execution of Sprint 2 Phase 3 Multimodal Fusion."""
    config = Phase3Config()
    os.makedirs(config.artifacts_dir, exist_ok=True)
    
    # Also ensure backend saved_models directory exists
    backend_saved_dir = os.path.join("ckd-project", "backend", "saved_models")
    os.makedirs(backend_saved_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("🚀 STARTING PHASE 3: MULTIMODAL FUSION PIPELINE (MEMBER B)")
    logger.info(f"Target Device: {config.device.upper()}")
    logger.info("=" * 70)

    # 1. Load Dataset
    if not os.path.exists(config.data_csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at '{config.data_csv_path}'")

    df = pd.read_csv(config.data_csv_path)
    logger.info(f"Loaded multimodal dataset with {len(df)} samples and {df.shape[1]} columns.")

    # Validate classification and image path
    y_raw = df["classification"].astype(int).values
    image_paths = df["image_path"].tolist()
    
    # Feature columns
    drop_cols = ["id", "image_path", "classification"]
    X_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    
    logger.info(f"Identified {X_df.shape[1]} clinical tabular features.")

    # 2. Extract Image Embeddings from Phase 2 ResNet
    img_embeddings = extract_resnet_embeddings(image_paths, config.resnet_path, config.device)

    # 3. Train/Val/Test Split (70% Train, 15% Val, 15% Test, Stratified)
    indices = np.arange(len(df))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        indices, y_raw, test_size=0.30, random_state=config.seed, stratify=y_raw
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.50, random_state=config.seed, stratify=y_temp
    )

    logger.info(f"Data Split -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # 4. Fit Tabular Preprocessor strictly on Train Split
    X_train_df = X_df.iloc[train_idx]
    X_val_df = X_df.iloc[val_idx]
    X_test_df = X_df.iloc[test_idx]

    preprocessor, feature_columns = build_tabular_preprocessor(X_train_df)
    X_train_proc = preprocessor.transform(X_train_df)
    X_val_proc = preprocessor.transform(X_val_df)
    X_test_proc = preprocessor.transform(X_test_df)

    tabular_dim = X_train_proc.shape[1]
    logger.info(f"Preprocessed tabular dimension: {tabular_dim} (Standardized + One-Hot Encoded)")

    # 5. Build PyTorch DataLoaders
    train_dataset = PrecomputedMultimodalDataset(X_train_proc, img_embeddings[train_idx], y_train)
    val_dataset = PrecomputedMultimodalDataset(X_val_proc, img_embeddings[val_idx], y_val)
    test_dataset = PrecomputedMultimodalDataset(X_test_proc, img_embeddings[test_idx], y_test)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 6. Instantiate and Train 3 Fusion Strategies
    models_to_train = {
        "Concatenation Fusion": ConcatenationFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2),
        "Weighted Fusion": WeightedAverageFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2),
        "Cross-Attention Fusion": CrossModalAttentionFusionModel(tabular_dim=tabular_dim, image_dim=512, embed_dim=128, num_classes=2)
    }

    trained_models = {}
    training_histories = {}
    evaluation_results = {}

    for name, model_instance in models_to_train.items():
        logger.info(f"\n--- Training Strategy: {name} ---")
        t0 = time.time()
        m, h = train_fusion_model(model_instance, train_loader, val_loader, config, name)
        elapsed = time.time() - t0
        logger.info(f"Completed {name} training in {elapsed:.2f}s")
        
        trained_models[name] = m
        training_histories[name] = h
        eval_metrics = evaluate_fusion_model(m, test_loader, config.device)
        evaluation_results[name] = eval_metrics
        logger.info(f"Test Results [{name}] -> Acc: {eval_metrics['accuracy']*100:.2f}% | F1: {eval_metrics['f1_score']*100:.2f}% | AUC: {eval_metrics['roc_auc']:.4f}")

    # 7. Evaluate Baselines for Rigorous Empirical Comparison
    logger.info("\n--- Evaluating Baselines on Identical Test Split ---")
    knn_metrics = evaluate_knn_baseline(X_test_df, y_test, config.knn_model_path)
    logger.info(f"Phase 1 KNN Baseline   -> Acc: {knn_metrics['accuracy']*100:.2f}% | F1: {knn_metrics['f1_score']*100:.2f}% | AUC: {knn_metrics['roc_auc']:.4f}")

    test_image_paths = [image_paths[i] for i in test_idx]
    resnet_metrics = evaluate_resnet_baseline(test_image_paths, y_test, config.resnet_path, config.device)
    logger.info(f"Phase 2 ResNet Baseline-> Acc: {resnet_metrics['accuracy']*100:.2f}% | F1: {resnet_metrics['f1_score']*100:.2f}% | AUC: {resnet_metrics['roc_auc']:.4f}")

    # 8. Assemble Full Benchmark Suite
    benchmarks_summary = {
        "Phase 1 KNN (Tabular)": {
            "modality": "Tabular Clinical",
            "accuracy": knn_metrics["accuracy"],
            "precision": knn_metrics["precision"],
            "recall": knn_metrics["recall"],
            "f1_score": knn_metrics["f1_score"],
            "roc_auc": knn_metrics["roc_auc"],
            "confusion_matrix": knn_metrics["confusion_matrix"]
        },
        "Phase 2 ResNet (Image)": {
            "modality": "CT Scan Image",
            "accuracy": resnet_metrics["accuracy"],
            "precision": resnet_metrics["precision"],
            "recall": resnet_metrics["recall"],
            "f1_score": resnet_metrics["f1_score"],
            "roc_auc": resnet_metrics["roc_auc"],
            "confusion_matrix": resnet_metrics["confusion_matrix"]
        },
        "Concatenation Fusion": {
            "modality": "Multimodal (Concat)",
            "accuracy": evaluation_results["Concatenation Fusion"]["accuracy"],
            "precision": evaluation_results["Concatenation Fusion"]["precision"],
            "recall": evaluation_results["Concatenation Fusion"]["recall"],
            "f1_score": evaluation_results["Concatenation Fusion"]["f1_score"],
            "roc_auc": evaluation_results["Concatenation Fusion"]["roc_auc"],
            "confusion_matrix": evaluation_results["Concatenation Fusion"]["confusion_matrix"]
        },
        "Weighted Fusion": {
            "modality": "Multimodal (Gated)",
            "accuracy": evaluation_results["Weighted Fusion"]["accuracy"],
            "precision": evaluation_results["Weighted Fusion"]["precision"],
            "recall": evaluation_results["Weighted Fusion"]["recall"],
            "f1_score": evaluation_results["Weighted Fusion"]["f1_score"],
            "roc_auc": evaluation_results["Weighted Fusion"]["roc_auc"],
            "confusion_matrix": evaluation_results["Weighted Fusion"]["confusion_matrix"]
        },
        "Cross-Attention Fusion": {
            "modality": "Multimodal (Attention)",
            "accuracy": evaluation_results["Cross-Attention Fusion"]["accuracy"],
            "precision": evaluation_results["Cross-Attention Fusion"]["precision"],
            "recall": evaluation_results["Cross-Attention Fusion"]["recall"],
            "f1_score": evaluation_results["Cross-Attention Fusion"]["f1_score"],
            "roc_auc": evaluation_results["Cross-Attention Fusion"]["roc_auc"],
            "confusion_matrix": evaluation_results["Cross-Attention Fusion"]["confusion_matrix"]
        }
    }

    # 9. Determine Best Fusion Architecture
    fusion_candidates = ["Cross-Attention Fusion", "Concatenation Fusion", "Weighted Fusion"]
    # Selection criteria: highest F1-score, tiebreaker ROC-AUC, then Accuracy
    best_fusion_name = max(
        fusion_candidates,
        key=lambda k: (benchmarks_summary[k]["f1_score"], benchmarks_summary[k]["roc_auc"], benchmarks_summary[k]["accuracy"])
    )
    best_model = trained_models[best_fusion_name]
    logger.info(f"\n🏆 Best Multimodal Fusion Architecture: {best_fusion_name.upper()}")

    # 10. Checkpoint Best Model & Pipeline Metadata
    checkpoint_payload = {
        "model_architecture": best_fusion_name,
        "model_state_dict": best_model.state_dict(),
        "tabular_dim": tabular_dim,
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "classes": ["Not CKD", "CKD"],
        "config": {
            "tabular_embed_dim": config.tabular_embed_dim,
            "image_embed_dim": config.image_embed_dim,
            "projected_dim": config.projected_dim,
            "num_classes": config.num_classes,
        },
        "benchmark_metrics": benchmarks_summary[best_fusion_name]
    }

    torch.save(checkpoint_payload, config.best_model_path)
    logger.info(f"Saved primary fusion model artifact to {config.best_model_path}")
    
    # Also save to backend saved_models for seamless Member C / service layer pickup
    backend_fusion_path = os.path.join(backend_saved_dir, "phase3_fusion_model.pth")
    torch.save(checkpoint_payload, backend_fusion_path)
    logger.info(f"Copied fusion model artifact to backend: {backend_fusion_path}")

    # 11. Save Metrics JSON
    with open(config.metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "best_fusion_model": best_fusion_name,
            "benchmarks": benchmarks_summary
        }, f, indent=2)
    logger.info(f"Exported metrics report to {config.metrics_path}")

    # 12. Generate Visual Comparison Plots
    best_cm = benchmarks_summary[best_fusion_name]["confusion_matrix"]
    generate_comparison_plots(
        benchmarks=benchmarks_summary,
        best_cm=best_cm,
        best_model_name=best_fusion_name,
        histories=training_histories,
        output_path=config.comparison_plot_path
    )

    logger.info("=" * 70)
    logger.info("✅ PHASE 3 MULTIMODAL FUSION TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_phase3_multimodal_fusion()
