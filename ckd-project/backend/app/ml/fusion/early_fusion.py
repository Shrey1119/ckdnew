import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

from app.config.config import settings
from app.ml.tabular.knn_wrapper import knn_wrapper
from app.ml.image.resnet_model import ResNet18KidneyClassifier

logger = logging.getLogger(__name__)

class EarlyFusionMLP(nn.Module):
    def __init__(self, tabular_dim=24, image_dim=512, num_classes=2):
        super(EarlyFusionMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(tabular_dim + image_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, tab_feat, img_feat):
        x = torch.cat([tab_feat, img_feat], dim=1)
        return self.mlp(x)

class EarlyFusionModelManager:
    def __init__(self):
        self.mlp_model = None
        self.tabular_dim = 24  # Standard preprocessed KNN features size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = Path(settings.SAVED_MODELS_DIR) / "early_fusion_mlp.pth"
        
    def load_model(self):
        try:
            # We initialize MLP shape dynamically based on KNN preprocessor output
            dummy_df = knn_wrapper.preprocess_input({})
            preprocessor = knn_wrapper.model.named_steps["preprocessor"]
            dummy_feat = preprocessor.transform(dummy_df)
            self.tabular_dim = dummy_feat.shape[1]
            
            self.mlp_model = EarlyFusionMLP(tabular_dim=self.tabular_dim, image_dim=512)
            if self.model_path.exists():
                logger.info(f"Loading Early Fusion MLP weights from {self.model_path}")
                self.mlp_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.mlp_model = self.mlp_model.to(self.device)
            self.mlp_model.eval()
        except Exception as e:
            logger.error(f"Error loading Early Fusion MLP model: {e}")
            
    def predict(self, tabular_dict: dict, img_path: str) -> dict:
        if self.mlp_model is None:
            self.load_model()
            
        # 1. Preprocess tabular data using KNN pipeline preprocessor
        try:
            df_row = knn_wrapper.preprocess_input(tabular_dict)
            preprocessor = knn_wrapper.model.named_steps["preprocessor"]
            tab_arr = preprocessor.transform(df_row)
            tab_tensor = torch.tensor(tab_arr, dtype=torch.float32).to(self.device)
        except Exception as e:
            logger.error(f"Tabular preprocessing in early fusion failed: {e}")
            return {"status": "error", "message": f"Tabular preprocessor failed: {str(e)}"}
            
        # 2. Extract image features from ResNet18
        img_model_path = Path(settings.SAVED_MODELS_DIR) / "phase2_resnet18.pth"
        if not img_model_path.exists() or not os.path.exists(img_path):
            # If image model or file doesn't exist, we fall back to tabular-only representation
            return {"status": "error", "message": "Image model weights or patient image not found."}
            
        try:
            img_model = ResNet18KidneyClassifier(num_classes=4, pretrained=False)
            img_model.load_state_dict(torch.load(img_model_path, map_location=self.device))
            img_model = img_model.to(self.device)
            img_model.eval()
            
            transform = transforms.Compose([
                transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(self.device)
                
            with torch.no_grad():
                img_feat = img_model.extract_features(img_tensor)
        except Exception as e:
            logger.error(f"Image feature extraction in early fusion failed: {e}")
            return {"status": "error", "message": f"Image feature extractor failed: {str(e)}"}
            
        # 3. Predict using MLP
        try:
            if not self.model_path.exists():
                # Automatically train model if weights do not exist
                logger.info("Early Fusion MLP weights not found. Running auto-train...")
                self.train_mlp()
                
            self.mlp_model.eval()
            with torch.no_grad():
                logits = self.mlp_model(tab_tensor, img_feat)
                probabilities = torch.softmax(logits, dim=1)
                
            prediction = int(probabilities.argmax(dim=1)[0].item())
            probability = float(probabilities[0][1].item()) # probability of class 1 (CKD)
            
            pred_prob = probability if prediction == 1 else (1.0 - probability)
            
            if pred_prob >= 0.8:
                confidence = "high"
            elif pred_prob >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"
                
            if probability >= 0.7:
                risk_level = "high"
            elif probability >= 0.3:
                risk_level = "medium"
            else:
                risk_level = "low"
                
            return {
                "status": "success",
                "prediction": prediction,
                "probability": round(probability, 4),
                "label": "ckd" if prediction == 1 else "not_ckd",
                "confidence": confidence,
                "risk_level": risk_level
            }
        except Exception as e:
            logger.error(f"MLP forward pass in early fusion failed: {e}")
            return {"status": "error", "message": f"MLP prediction failed: {str(e)}"}
            
    def train_mlp(self, epochs=20, lr=1e-3):
        """
        Runs rapid training for the Early Fusion MLP model on the local dataset.
        """
        logger.info("Starting Early Fusion MLP training...")
        
        csv_path = Path(settings.DATASET_CSV_PATH)
        if not csv_path.exists():
            alt_path = Path(__file__).resolve().parents[4] / "kidney_multimodal_dataset_FIXED.csv"
            if alt_path.exists():
                csv_path = alt_path
            else:
                raise FileNotFoundError("Dataset CSV not found")
                
        df = pd.read_csv(csv_path)
        
        # Load image classifier to extract features
        img_model_path = Path(settings.SAVED_MODELS_DIR) / "phase2_resnet18.pth"
        if not img_model_path.exists():
            raise FileNotFoundError("ResNet18 weights not found. Train ResNet18 first.")
            
        img_model = ResNet18KidneyClassifier(num_classes=4, pretrained=False)
        img_model.load_state_dict(torch.load(img_model_path, map_location=self.device))
        img_model = img_model.to(self.device)
        img_model.eval()
        
        # Extract features for all samples
        transform = transforms.Compose([
            transforms.Resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        preprocessor = knn_wrapper.model.named_steps["preprocessor"]
        
        # Prepare tabular features
        tabular_df = df.drop(columns=["id", "image_path", "classification"], errors="ignore")
        tabular_df = tabular_df[knn_wrapper.features]
        tab_processed = preprocessor.transform(tabular_df)
        self.tabular_dim = tab_processed.shape[1]
        
        X_tab = torch.tensor(tab_processed, dtype=torch.float32)
        y = torch.tensor(df["classification"].values, dtype=torch.long)
        
        # Extract image embeddings (CPU or GPU batch-wise to prevent memory overload)
        logger.info("Extracting ResNet18 features for early fusion...")
        X_img_list = []
        csv_dir = Path(csv_path).resolve().parent
        
        for idx, row in df.iterrows():
            img_path = csv_dir / row["image_path"]
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    img_t = transform(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    feat = img_model.extract_features(img_t)
                X_img_list.append(feat.cpu())
            except Exception:
                X_img_list.append(torch.zeros((1, 512)))
                
        X_img = torch.cat(X_img_list, dim=0)
        
        # Initialize MLP
        self.mlp_model = EarlyFusionMLP(tabular_dim=self.tabular_dim, image_dim=512)
        self.mlp_model = self.mlp_model.to(self.device)
        
        optimizer = optim.Adam(self.mlp_model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        dataset_size = len(df)
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        # Train/Val split
        split = int(dataset_size * 0.8)
        train_idx, val_idx = indices[:split], indices[split:]
        
        X_tab_train, X_tab_val = X_tab[train_idx], X_tab[val_idx]
        X_img_train, X_img_val = X_img[train_idx], X_img[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        batch_size = 32
        best_val_loss = float("inf")
        
        for epoch in range(1, epochs + 1):
            self.mlp_model.train()
            running_loss = 0.0
            
            for i in range(0, len(train_idx), batch_size):
                batch_tab = X_tab_train[i:i+batch_size].to(self.device)
                batch_img = X_img_train[i:i+batch_size].to(self.device)
                batch_y = y_train[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                logits = self.mlp_model(batch_tab, batch_img)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch_tab.size(0)
                
            train_loss = running_loss / len(train_idx)
            
            # Val evaluation
            self.mlp_model.eval()
            with torch.no_grad():
                val_logits = self.mlp_model(X_tab_val.to(self.device), X_img_val.to(self.device))
                val_loss = criterion(val_logits, y_val.to(self.device)).item()
                preds = val_logits.argmax(dim=1).cpu()
                val_acc = (preds == y_val).float().mean().item()
                
            logger.info(f"MLP Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.mlp_model.state_dict(), self.model_path)
                logger.info(f"--> Saved best early fusion MLP to {self.model_path}")
                
        # Load best model
        self.load_model()
        
early_fusion_manager = EarlyFusionModelManager()
