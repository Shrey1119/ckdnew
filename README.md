# Chronic Kidney Disease (CKD) Multimodal Dataset

## 📌 Project Overview
This project builds a **multimodal dataset** for Chronic Kidney Disease detection. It combines:
1.  **Tabular Clinical Data**: Blood tests (Creatinine, Hemoglobin, etc.) and patient metadata.
2.  **Medical Imaging**: CT Scans of kidneys (Normal, Cyst, Tumor, Stone).

The goal is to create a dataset where every patient record (CSV row) is paired with a corresponding medical image, allowing for multi-input machine learning models (e.g., combining MLP for data and CNN for images).

## 📂 File Structure & Functionality

### 1. `img_download.py` (Step 1)
*   **Purpose**: Downloads the CT Kidney dataset from Kaggle.
*   **Action**: Uses the Kaggle API to fetch the dataset, extracts it, and reorganizes the folders into a clean structure (`kidney_images/Tumor`, `kidney_images/Normal`, etc.).

### 2. `ds_generate.py` (Step 2)
*   **Purpose**: Generates the clinical CSV file.
*   **Action**:
    *   Downloads real clinical data from a remote repository.
    *   Cleans the data (handling missing values, fixing types).
    *   **Augments** the dataset to 1500 rows using statistical sampling and noise injection to ensure enough data for training.
    *   Assigns a *theoretical* image path to each row based on medical logic (e.g., High Creatinine -> Tumor image).

### 3. `fix_path.py` (Step 3)
*   **Purpose**: Connects the CSV to the actual downloaded images.
*   **Action**:
    *   Scans the `kidney_images` folder to find all valid `.jpg` files.
    *   Updates the CSV generated in Step 2.
    *   Replaces the theoretical paths with **actual file paths** that exist on your disk, ensuring no "File Not Found" errors during training.

### 4. `audit_data.py` (Step 4)
*   **Purpose**: Quality Assurance.
*   **Action**:
    *   Checks class balance (Sick vs. Healthy).
    *   Verifies that every image path in the CSV actually exists.
    *   Runs a baseline Logistic Regression model on the tabular data to estimate the "information power" of the blood test features.

## 🚀 How to Run
Execute the scripts in this specific order to build the dataset:

1.  **Download Images**:
    ```bash
    python img_download.py
    ```
2.  **Generate Clinical Data**:
    ```bash
    python ds_generate.py
    ```
3.  **Link Images to Data**:
    ```bash
    python fix_path.py
    ```
4.  **Verify Integrity**:
    ```bash
    python audit_data.py
    ```

---

## 📊 Data Audit Report
*(Generated from `audit_data.py`)*

🕵️  STARTING DATA AUDIT...
========================================
✅ CSV Loaded: 1500 rows

📊 CLASS DISTRIBUTION:
   - CKD (Sick):     405 (27.0%)
   - Normal (Healthy): 1095 (73.0%)
   Note: A 'dumb' model guessing the majority class would have 73.0% accuracy.

🔍 DATA QUALITY:
   - Missing Values: 0
   - Duplicate Rows: 0

🖼️  IMAGE INTEGRITY SCAN:
   - Images Found:   1500
   - Images Missing: 0
   - Corrupt Files:  0

========================================
📈 ESTIMATED CLINICAL METRICS (Based on Blood Work)
   (Running rapid Logistic Regression on Tabular Data...)
========================================
✅ Accuracy:    0.9967
🎯 Precision:   1.0000
🔍 Recall:      0.9882
🛡️  Specificity: 1.0000
📉 RMSE:        0.0557
------------------------------
Interpretation:
These numbers represent the 'Information Power' of your dataset.
If a simple statistical test gets high accuracy, your data is very clean.