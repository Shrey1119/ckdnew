"""Phase 2 Incremental Chunk-Wise Training for ResNet-18 CT Kidney Classification.

Divides the full dataset into chunks of N images (default: 100).
Trains the model on each chunk sequentially, saving a checkpoint after each chunk.
Supports stop & resume: re-running the script picks up from where it left off.

Usage:
    python phase2_incremental_train.py                  # Run all remaining chunks
    python phase2_incremental_train.py --chunks 5       # Run only next 5 chunks
    python phase2_incremental_train.py --chunk-size 200 # Use 200 images per chunk
    python phase2_incremental_train.py --reset          # Start fresh from chunk 0
"""

import os
import json
import time
import glob
import logging
import argparse
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CLASSES        = ["Normal", "Cyst", "Tumor", "Stone"]
CLASS_TO_IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS   = {i: c for i, c in enumerate(CLASSES)}

MODEL_DIR       = "model_artifacts"
LATEST_CKPT     = os.path.join(MODEL_DIR, "phase2_resnet_latest.pth")
FINAL_MODEL     = os.path.join(MODEL_DIR, "phase2_resnet.pth")
PROGRESS_FILE   = os.path.join(MODEL_DIR, "phase2_incremental_progress.json")

DATA_ROOT = os.path.join(
    "kidney_images",
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
    "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
)

IMAGE_SIZE       = 224
EPOCHS_PER_CHUNK = 3
BATCH_SIZE       = 16
LR               = 1e-4
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
SEED             = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────

def collect_all_samples(data_root: str) -> List[Tuple[str, int]]:
    """Walk all 4 class folders and return (path, label) list."""
    samples = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_root, cls)
        if not os.path.isdir(cls_dir):
            logger.warning(f"Warning: Class folder not found: {cls_dir}")
            continue
        files = sorted(
            glob.glob(os.path.join(cls_dir, "*.jpg")) +
            glob.glob(os.path.join(cls_dir, "*.jpeg")) +
            glob.glob(os.path.join(cls_dir, "*.png"))
        )
        label = CLASS_TO_IDX[cls]
        for f in files:
            samples.append((f, label))
        logger.info(f"  {cls:8s}: {len(files):5d} images")
    return samples


def make_stratified_chunks(samples: List[Tuple[str, int]], chunk_size: int) -> List[List[Tuple[str, int]]]:
    """Split samples into stratified chunks so every chunk has all 4 classes proportionally."""
    np.random.seed(SEED)

    by_class = {i: [] for i in range(len(CLASSES))}
    for path, label in samples:
        by_class[label].append((path, label))

    for label in by_class:
        np.random.shuffle(by_class[label])

    total = len(samples)
    class_fractions = {label: len(by_class[label]) / total for label in by_class}
    class_per_chunk  = {label: max(1, round(class_fractions[label] * chunk_size)) for label in by_class}

    class_pointers = {label: 0 for label in by_class}
    chunks = []
    while True:
        chunk = []
        for label in range(len(CLASSES)):
            start = class_pointers[label]
            end   = start + class_per_chunk[label]
            take  = by_class[label][start:end]
            if not take:
                continue
            chunk.extend(take)
            class_pointers[label] += len(take)

        if not chunk:
            break
        np.random.shuffle(chunk)
        chunks.append(chunk)

        if all(class_pointers[label] >= len(by_class[label]) for label in by_class):
            break

    return chunks


# ─────────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────────

class ChunkDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            logger.error(f"Bad image {path}: {e}")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=0)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────

def build_model(num_classes: int = 4) -> nn.Module:
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except Exception:
        model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer) -> dict:
    """Load latest checkpoint if it exists; returns metadata dict."""
    if os.path.exists(LATEST_CKPT):
        logger.info(f"Loading checkpoint from '{LATEST_CKPT}' ...")
        ckpt = torch.load(LATEST_CKPT, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info(f"   Resumed from chunk {ckpt['meta']['last_chunk_done'] + 1} "
                    f"(chunks done: {ckpt['meta']['chunks_done']})")
        return ckpt["meta"]
    logger.info("No checkpoint found — starting fresh training.")
    return {"last_chunk_done": -1, "chunks_done": 0, "history": []}


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, meta: dict):
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "meta":                 meta,
        "classes":              CLASSES,
        "class_to_idx":         CLASS_TO_IDX,
    }, LATEST_CKPT)
    # Also overwrite the main model file so the backend can use it immediately
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes":          CLASSES,
        "class_to_idx":     CLASS_TO_IDX,
    }, FINAL_MODEL)


# ─────────────────────────────────────────────
# Training / Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out  = model(images)
            loss = criterion(out, labels)
            total_loss += loss.item() * images.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    return total_loss / n, accuracy_score(all_labels, all_preds)


# ─────────────────────────────────────────────
# Main incremental training loop
# ─────────────────────────────────────────────

def run_incremental_training(chunk_size: int = 100, max_chunks: int = None, reset: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  Phase 2 - Incremental Chunk-Wise ResNet-18 Training")
    logger.info(f"  Device      : {DEVICE}")
    logger.info(f"  Chunk size  : {chunk_size} images")
    logger.info(f"  Epochs/chunk: {EPOCHS_PER_CHUNK}")
    logger.info("=" * 60)

    # 1. Collect & chunk the full dataset
    logger.info(f"\nScanning dataset at: {DATA_ROOT}")
    all_samples = collect_all_samples(DATA_ROOT)
    logger.info(f"   Total images found: {len(all_samples)}")

    chunks = make_stratified_chunks(all_samples, chunk_size)
    logger.info(f"   Total chunks: {len(chunks)} (chunk_size={chunk_size})\n")

    # Save manifest
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "total_chunks":  len(chunks),
            "chunk_size":    chunk_size,
            "total_images":  len(all_samples),
        }, f, indent=2)

    # 2. Build model & optimizer
    model     = build_model().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 3. Load or reset checkpoint
    if reset and os.path.exists(LATEST_CKPT):
        os.remove(LATEST_CKPT)
        logger.info("Reset: removed old checkpoint.")

    meta        = load_checkpoint(model, optimizer)
    start_chunk = meta.get("last_chunk_done", -1) + 1
    history     = meta.get("history", [])

    # 4. Determine range for this session
    end_chunk = len(chunks)
    if max_chunks is not None:
        end_chunk = min(start_chunk + max_chunks, len(chunks))

    if start_chunk >= len(chunks):
        logger.info("All chunks have already been trained! Model is fully trained.")
        logger.info(f"Final model: {FINAL_MODEL}")
        return

    logger.info(f"Running chunks {start_chunk + 1} to {end_chunk} of {len(chunks)} total\n")
    train_tf, val_tf = get_transforms()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

    session_start = time.time()

    for chunk_idx in range(start_chunk, end_chunk):
        chunk_data = chunks[chunk_idx]
        chunk_num  = chunk_idx + 1  # 1-indexed

        split         = max(1, int(0.8 * len(chunk_data)))
        train_samples = chunk_data[:split]
        val_samples   = chunk_data[split:]

        train_loader = DataLoader(
            ChunkDataset(train_samples, train_tf),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            ChunkDataset(val_samples, val_tf),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        logger.info("-" * 55)
        logger.info(f"  CHUNK {chunk_num:03d}/{len(chunks)}  |  "
                    f"train={len(train_samples)}  val={len(val_samples)}")
        logger.info("-" * 55)

        chunk_start   = time.time()
        best_val_loss = float('inf')

        for epoch in range(1, EPOCHS_PER_CHUNK + 1):
            t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            v_loss, v_acc = evaluate(model, val_loader, criterion)
            scheduler.step(v_loss)
            logger.info(
                f"  Epoch {epoch}/{EPOCHS_PER_CHUNK}  |  "
                f"Train Loss: {t_loss:.4f}  Acc: {t_acc*100:.1f}%  |  "
                f"Val Loss: {v_loss:.4f}  Acc: {v_acc*100:.1f}%"
            )
            if v_loss < best_val_loss:
                best_val_loss = v_loss

        chunk_elapsed = time.time() - chunk_start

        history.append({
            "chunk":          chunk_num,
            "train_images":   len(train_samples),
            "val_images":     len(val_samples),
            "best_val_loss":  round(best_val_loss, 4),
            "time_seconds":   round(chunk_elapsed, 1)
        })

        meta = {
            "last_chunk_done": chunk_idx,
            "chunks_done":     chunk_num,
            "total_chunks":    len(chunks),
            "history":         history
        }
        save_checkpoint(model, optimizer, meta)

        images_seen = chunk_num * chunk_size
        logger.info(f"  Checkpoint saved. Chunk {chunk_num} done in {chunk_elapsed:.1f}s")
        logger.info(f"  Progress: {chunk_num}/{len(chunks)} chunks "
                    f"| ~{min(images_seen, len(all_samples))}/{len(all_samples)} images seen\n")

    total_elapsed = time.time() - session_start
    chunks_this_session = end_chunk - start_chunk

    logger.info("=" * 60)
    logger.info(f"  Session complete!")
    logger.info(f"  Chunks trained this session : {chunks_this_session}")
    logger.info(f"  Total chunks done overall   : {end_chunk} / {len(chunks)}")
    logger.info(f"  Session time                : {total_elapsed:.1f}s")
    logger.info(f"  Checkpoint (resume next time): {LATEST_CKPT}")
    logger.info(f"  Live model (backend uses)   : {FINAL_MODEL}")

    if end_chunk >= len(chunks):
        logger.info("\n  FULL DATASET TRAINED! Model is completely trained.")
    else:
        remaining = len(chunks) - end_chunk
        logger.info(f"\n  {remaining} chunks remaining. Run script again to continue.")
    logger.info("=" * 60)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Incremental Chunk-Wise ResNet-18 Training")
    parser.add_argument("--chunk-size", type=int,  default=100,  help="Images per chunk (default: 100)")
    parser.add_argument("--chunks",     type=int,  default=None, help="Max chunks to run this session (default: all remaining)")
    parser.add_argument("--reset",      action="store_true",     help="Discard existing checkpoint and retrain from scratch")
    args = parser.parse_args()

    run_incremental_training(
        chunk_size=args.chunk_size,
        max_chunks=args.chunks,
        reset=args.reset
    )
