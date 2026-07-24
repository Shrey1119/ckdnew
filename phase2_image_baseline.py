"""Phase 2 Image Classification Pipeline (ResNet-18 for CT Kidney Classification).

Classifies CT scan kidney images into 4 categories:
- Normal (0)
- Cyst (1)
- Tumor (2)
- Stone (3)
"""

import os
import json
import time
import glob
import logging
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Class Mapping
CLASSES = ["Normal", "Cyst", "Tumor", "Stone"]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

MODEL_DIR = "model_artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "phase2_resnet.pth")
METRICS_PATH = os.path.join(MODEL_DIR, "phase2_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "phase2_confusion_matrix.png")
CLASS_MAPPING_PATH = os.path.join(MODEL_DIR, "phase2_class_mapping.json")

@dataclass
class Phase2Config:
    model_name: str = "resnet18"
    image_size: int = 224
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_classes: int = 4
    data_dir: str = "kidney_images"
    test_size: float = 0.15
    val_size: float = 0.15
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def create_synthetic_images_if_needed(data_dir: str, num_samples_per_class: int = 40) -> List[Tuple[str, int]]:
    """Creates synthetic CT kidney image files if real images are not yet present."""
    os.makedirs(data_dir, exist_ok=True)
    image_paths_labels = []

    # Check if real images exist
    all_found = glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)

    found_by_class = {cls: [] for cls in CLASSES}
    for filepath in all_found:
        filepath_lower = filepath.lower()
        for cls in CLASSES:
            if cls.lower() in filepath_lower:
                found_by_class[cls].append((filepath, CLASS_TO_IDX[cls]))
                break

    total_real = sum(len(v) for v in found_by_class.values())
    if total_real >= 20:
        logger.info(f"✅ Found {total_real} real images across classes in '{data_dir}'.")
        for cls in CLASSES:
            image_paths_labels.extend(found_by_class[cls])
        return image_paths_labels

    logger.info(f"⚠️ Real dataset not found in '{data_dir}' (found {total_real} images). Generating synthetic CT scans for demonstration...")
    synth_dir = os.path.join(data_dir, "synthetic_dataset")
    os.makedirs(synth_dir, exist_ok=True)

    np.random.seed(42)
    for cls in CLASSES:
        cls_dir = os.path.join(synth_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)
        label = CLASS_TO_IDX[cls]

        for i in range(num_samples_per_class):
            img_path = os.path.join(cls_dir, f"synth_{cls.lower()}_{i:03d}.png")
            # Create synthetic grayscale CT scan (224x224)
            base_noise = np.random.normal(60, 15, (224, 224)).astype(np.uint8)
            img = Image.fromarray(base_noise).convert("RGB")
            draw = ImageDraw.Draw(img)

            # Draw kidney outline
            draw.ellipse([30, 40, 190, 180], fill=(90, 90, 90), outline=(140, 140, 140))
            draw.ellipse([50, 60, 170, 160], fill=(70, 70, 70))

            # Class specific features
            if cls == "Cyst":
                draw.ellipse([90, 90, 130, 130], fill=(20, 20, 20), outline=(40, 40, 40))
            elif cls == "Tumor":
                draw.ellipse([80, 80, 140, 140], fill=(180, 170, 160), outline=(220, 210, 200))
            elif cls == "Stone":
                draw.ellipse([100, 100, 115, 115], fill=(255, 255, 255), outline=(255, 255, 255))
            # Normal remains clean

            img.save(img_path)
            image_paths_labels.append((img_path, label))

    logger.info(f"✅ Created {len(image_paths_labels)} synthetic images across 4 classes in '{synth_dir}'.")
    return image_paths_labels


class CTKidneyDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            image = Image.new("RGB", (224, 224), color=0)

        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def build_resnet18_model(num_classes: int = 4) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        logger.info("Loading ResNet18 without pre-trained weights (offline fallback)...")
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_epoch(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: str) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate_model(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, device: str) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total = len(all_labels)
    eval_loss = running_loss / total
    eval_acc = accuracy_score(all_labels, all_preds)
    return eval_loss, eval_acc, np.array(all_labels), np.array(all_preds)


def train_pipeline(config: Phase2Config) -> Dict[str, Any]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"🚀 Starting Phase 2 ResNet-18 Training Pipeline on device: {config.device}")

    # 1. Prepare Data
    samples = create_synthetic_images_if_needed(config.data_dir)
    labels = [s[1] for s in samples]

    train_samples, test_val_samples = train_test_split(
        samples, test_size=(config.val_size + config.test_size), random_state=config.seed, stratify=labels
    )
    val_ratio = config.val_size / (config.val_size + config.test_size)
    test_val_labels = [s[1] for s in test_val_samples]
    val_samples, test_samples = train_test_split(
        test_val_samples, test_size=(1.0 - val_ratio), random_state=config.seed, stratify=test_val_labels
    )

    logger.info(f"📊 Dataset Split -> Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # 2. Data Loaders
    train_tf, val_test_tf = get_transforms(config.image_size)
    train_dataset = CTKidneyDataset(train_samples, transform=train_tf)
    val_dataset = CTKidneyDataset(val_samples, transform=val_test_tf)
    test_dataset = CTKidneyDataset(test_samples, transform=val_test_tf)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 3. Model, Loss, Optimizer, Scheduler
    model = build_resnet18_model(num_classes=config.num_classes).to(config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # 4. Training Loop with Checkpointing
    best_val_loss = float('inf')
    patience = 4
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.device)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, config.device)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        logger.info(f"Epoch [{epoch:02d}/{config.epochs:02d}] -> Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'classes': CLASSES,
                'class_to_idx': CLASS_TO_IDX
            }, MODEL_PATH)
            logger.info(f"  💾 Saved best model checkpoint to '{MODEL_PATH}'")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                break

    elapsed_time = time.time() - start_time
    logger.info(f"⏱️ Training completed in {elapsed_time:.2f} seconds.")

    # 5. Evaluate Best Model on Test Set
    checkpoint = torch.load(MODEL_PATH, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, config.device)
    logger.info(f"🎯 Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")

    report_dict = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=CLASSES)
    cm = confusion_matrix(y_true, y_pred)

    logger.info("\n" + report_text)

    # 6. Save Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('Phase 2 ResNet-18 CT Scan Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    logger.info(f"📊 Saved confusion matrix plot to '{CONFUSION_MATRIX_PATH}'")

    # 7. Save Metrics JSON & Class Mapping
    metrics_summary = {
        "model_name": config.model_name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classes": CLASSES,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "training_history": history,
        "training_time_seconds": round(elapsed_time, 2)
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"📄 Saved metrics summary to '{METRICS_PATH}'")

    with open(CLASS_MAPPING_PATH, "w") as f:
        json.dump({"classes": CLASSES, "class_to_idx": CLASS_TO_IDX, "idx_to_class": IDX_TO_CLASS}, f, indent=2)

    return metrics_summary


def main() -> None:
    config = Phase2Config()
    train_pipeline(config)

if __name__ == "__main__":
    main()
