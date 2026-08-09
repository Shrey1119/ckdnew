# Chronic Disease Prediction Phases

## Phase 1 - Accurate Early Prediction (Implemented)
- Model: KNN (Nearest Neighbors) with GridSearchCV tuning
- Input: Tabular clinical values from `kidney_multimodal_dataset_FIXED.csv`
- Output: Binary CKD prediction + probability + confidence
- Scripts:
  - `train_phase1_knn.py`
  - `predict_cli.py`

## Phase 2 - Image Baseline (Implemented Scaffold)
- Target: Kidney CT image classification (normal/cyst/tumor/stone)
- Recommended model: ResNet18 transfer learning
- Deliverable path: `phase2_image_baseline.py`

## Phase 3 - Multimodal Fusion (Implemented Scaffold)
- Target: Fuse phase 1 tabular embeddings + phase 2 image embeddings
- Recommended approach: Late fusion MLP / weighted averaging
- Deliverable path: `phase3_multimodal_fusion.py`

## Phase 4 - Genetic + Generative AI (Implemented Scaffold)
- Genetic: optimize feature subsets and fusion weights
- Generative: augment minority classes using tabular/image generation
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
