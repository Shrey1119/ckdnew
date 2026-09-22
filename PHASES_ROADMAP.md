# Chronic Disease Prediction Phases

## Phase 1 - Accurate Early Prediction (Implemented)
- Model: KNN (Nearest Neighbors) with GridSearchCV tuning
- Input: Tabular clinical values from `kidney_multimodal_dataset_FIXED.csv`
- Output: Binary CKD prediction + probability + confidence
- Scripts:
  - `train_phase1_knn.py`
  - `predict_cli.py`

## Phase 2 - Image Baseline (Implemented & Trained ✅)
- Target: Kidney CT image classification (Normal / Cyst / Tumor / Stone)
- Model: ResNet-18 transfer learning with ImageNet normalization & data augmentation
- Script: `phase2_image_baseline.py`
- Model Artifact: `model_artifacts/phase2_resnet.pth`
- Standalone CLI: `predict_image_cli.py`
- Report: `PHASE2_SPRINT1_MEMBER_A_REPORT.md`

## Phase 3 - Multimodal Fusion (Implemented & Trained ✅)
- Target: Fuse Phase 1 tabular clinical embeddings + Phase 2 ResNet image embeddings
- Architectures Evaluated:
  1. Early Feature Concatenation (`ConcatenationFusionModel`)
  2. Modality-Gated Decision Fusion (`WeightedAverageFusionModel`)
  3. Bi-directional Cross-Modal Attention (`CrossModalAttentionFusionModel` — Best Selected)
- Script: `phase3_multimodal_fusion.py`
- Model Artifacts: `model_artifacts/phase3_fusion_model.pth` & `ckd-project/backend/saved_models/phase3_fusion_model.pth`
- Standalone CLI: `predict_multimodal_cli.py`
- Visual Benchmark: `model_artifacts/phase3_comparison.png`
- Metrics: `model_artifacts/phase3_metrics.json`
- Report: `PHASE3_SPRINT2_MEMBER_B_REPORT.md`

## Phase 4 - Genetic + Generative AI (Upcoming Sprint 3)
- Genetic: optimize feature subsets (DEAP) and fusion weights
- Generative: augment minority classes using SMOTE / tabular GANs
- Deliverable path: `phase4_genetic_generative.py`

## Training and Serving Sequence
1. Run phase 1 training:
   - `python train_phase1_knn.py`
2. Start backend API and call:
   - `GET /api/predictions/meta`
   - `POST /api/predictions/predict`
3. Frontend calls backend via reverse proxy:
   - `/api/backend/predictions/meta`
   - `/api/backend/predictions/predict`
