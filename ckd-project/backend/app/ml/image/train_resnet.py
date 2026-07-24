import os
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from app.config.config import settings
from app.ml.image.resnet_model import ResNet18KidneyClassifier

logger = logging.getLogger(__name__)

LABEL_MAP = {"Normal": 0, "Cyst": 1, "Tumor": 2, "Stone": 3}
CLASS_NAMES = ["Normal", "Cyst", "Tumor", "Stone"]

class KidneyDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.csv_dir = Path(csv_file).resolve().parent
        self.transform = transform
        
        self.samples = []
        for idx, row in self.df.iterrows():
            img_rel_path = row["image_path"]
            img_path = self.csv_dir / img_rel_path
            
            # Extract category from path
            img_path_str = str(img_rel_path).replace("\\", "/")
            category = None
            for cat in LABEL_MAP.keys():
                if cat in img_path_str:
                    category = cat
                    break
            
            if category is not None:
                self.samples.append({
                    "path": img_path,
                    "label": LABEL_MAP[category],
                    "patient_id": row.get("id")
                })
            else:
                # Default fallback
                self.samples.append({
                    "path": img_path,
                    "label": 0,
                    "patient_id": row.get("id")
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["path"]
        label = sample["label"]
        
        try:
            # Open image as RGB
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Fallback to black image if corrupt/missing during dynamic training
            logger.warning(f"Failed to load image {img_path}, using dummy placeholder. Error: {e}")
            img = Image.new("RGB", (settings.IMAGE_SIZE, settings.IMAGE_SIZE), color=0)
            
        if self.transform:
            img = self.transform(img)
            
        return img, label

class TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a transform independently so train/val don't share the same transform."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path = self.subset.dataset.samples[self.subset.indices[idx]]["path"]
        label = self.subset.dataset.samples[self.subset.indices[idx]]["label"]
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}, using dummy placeholder. Error: {e}")
            img = Image.new("RGB", (settings.IMAGE_SIZE, settings.IMAGE_SIZE), color=0)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_data_loaders(csv_file, batch_size=16, val_split=0.2):
    train_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load base dataset without transform (TransformSubset applies it independently)
    dataset = KidneyDataset(csv_file, transform=None)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = random_split(dataset, [train_size, val_size], generator=generator)
    
    # Wrap subsets with their own independent transforms
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader

def train_image_model(db_session=None, on_epoch_completed=None):
    """
    Run PyTorch training for ResNet18 and log metrics.
    """
    # Create save dir
    os.makedirs(settings.SAVED_MODELS_DIR, exist_ok=True)
    model_save_path = Path(settings.SAVED_MODELS_DIR) / "phase2_resnet18.pth"
    
    # TensorBoard setup
    tb_log_dir = Path("logs/tensorboard")
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    
    csv_path = Path(settings.DATASET_CSV_PATH)
    if not csv_path.exists():
        # Fallback to parent path
        alt_path = Path(__file__).resolve().parents[4] / "kidney_multimodal_dataset_FIXED.csv"
        if alt_path.exists():
            csv_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")
            
    train_loader, val_loader = get_data_loaders(str(csv_path), batch_size=settings.BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training Image AI on device: {device}")
    
    model = ResNet18KidneyClassifier(num_classes=4, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1, factor=0.1)
    
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    
    for epoch in range(1, settings.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        # Log to TensorBoard
        writer.add_scalar("Loss/Train", epoch_loss, epoch)
        writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", epoch_acc, epoch)
        writer.add_scalar("Accuracy/Val", epoch_val_acc, epoch)
        
        # Log to Database
        if db_session:
            try:
                from app.database.models import TrainingLog
                log_entry = TrainingLog(
                    epoch=epoch,
                    loss=epoch_loss,
                    accuracy=epoch_acc,
                    val_loss=epoch_val_loss,
                    val_accuracy=epoch_val_acc
                )
                db_session.add(log_entry)
                db_session.commit()
            except Exception as e:
                logger.error(f"Error saving training log to DB: {e}")
                db_session.rollback()
                
        if on_epoch_completed:
            on_epoch_completed(epoch, epoch_loss, epoch_acc, epoch_val_loss, epoch_val_acc)
            
        logger.info(f"Epoch {epoch}/{settings.EPOCHS} | Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
        
        # LR Scheduler
        scheduler.step(epoch_val_loss)
        
        # Save best model checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"--> Saved new best model checkpoint to {model_save_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("--> Early stopping triggered.")
                break
                
    writer.close()
    
    # Calculate final evaluation metrics on validation set
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)
    
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    logger.info(f"Training completed. Final metrics: {metrics}")
    
    # Save model status in DB
    if db_session:
        try:
            from app.database.models import ModelStatus
            status = db_session.query(ModelStatus).filter_by(name="ResNet18").first()
            if not status:
                status = ModelStatus(name="ResNet18", version="1.0.0")
                db_session.add(status)
            status.accuracy = metrics["accuracy"]
            status.f1 = metrics["f1"]
            status.is_active = True
            db_session.commit()
        except Exception as e:
            logger.error(f"Failed to update model status: {e}")
            db_session.rollback()
            
    return metrics
