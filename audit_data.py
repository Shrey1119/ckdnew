import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "kidney_multimodal_dataset_FIXED.csv"
IMG_DIR = "kidney_images"

print(f"🕵️  STARTING DATA AUDIT...")
print("="*40)

# ==========================================
# 1. TABULAR DATA ANALYSIS
# ==========================================
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    print(f"✅ CSV Loaded: {len(df)} rows")
    
    # A. Class Balance
    counts = df['classification'].value_counts()
    sick = counts.get(1, 0)
    healthy = counts.get(0, 0)
    baseline_acc = max(sick, healthy) / len(df)
    
    print(f"\n📊 CLASS DISTRIBUTION:")
    print(f"   - CKD (Sick):     {sick} ({sick/len(df)*100:.1f}%)")
    print(f"   - Normal (Healthy): {healthy} ({healthy/len(df)*100:.1f}%)")
    print(f"   Note: A 'dumb' model guessing the majority class would have {baseline_acc*100:.1f}% accuracy.")

    # B. Missing Values
    missing = df.isnull().sum().sum()
    print(f"\n🔍 DATA QUALITY:")
    print(f"   - Missing Values: {missing}")
    print(f"   - Duplicate Rows: {df.duplicated().sum()}")

else:
    print(f"❌ Error: CSV file '{CSV_FILE}' not found.")
    exit()

# ==========================================
# 2. IMAGE DATA ANALYSIS
# ==========================================
print(f"\n🖼️  IMAGE INTEGRITY SCAN:")
valid_images = 0
missing_images = 0
corrupt_images = 0

# Scan first 100 images to save time (remove [:100] to scan all)
scan_subset = df['image_path']

for img_path in scan_subset:
    if os.path.exists(img_path):
        try:
            with Image.open(img_path) as img:
                img.verify() # Check if file is not broken
            valid_images += 1
        except:
            corrupt_images += 1
    else:
        missing_images += 1

print(f"   - Images Found:   {valid_images}")
print(f"   - Images Missing: {missing_images}")
print(f"   - Corrupt Files:  {corrupt_images}")

if missing_images > 0:
    print("   ⚠️  WARNING: Your CSV points to images that don't exist in the folder.")

# ==========================================
# 3. STATISTICAL BASELINE (The Metrics You Asked For)
# ==========================================
print("\n" + "="*40)
print("📈 ESTIMATED CLINICAL METRICS (Based on Blood Work)")
print("   (Running rapid Logistic Regression on Tabular Data...)")
print("="*40)

# Prepare Data
feature_cols = ['age', 'bp', 'sc', 'hemo', 'wc', 'rc', 'bgr', 'bu']
# Filter only columns that actually exist in your CSV
existing_cols = [c for c in feature_cols if c in df.columns]

X = df[existing_cols]
y = df['classification']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fast Statistical Fit
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# Calculate Metrics
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp) if (tn+fp) > 0 else 0
rmse = np.sqrt(mean_squared_error(y_test, y_prob))

print(f"✅ Accuracy:    {accuracy_score(y_test, y_pred):.4f}")
print(f"🎯 Precision:   {classification_report(y_test, y_pred, output_dict=True)['1']['precision']:.4f}")
print(f"🔍 Recall:      {classification_report(y_test, y_pred, output_dict=True)['1']['recall']:.4f}")
print(f"🛡️  Specificity: {specificity:.4f}")
print(f"📉 RMSE:        {rmse:.4f}")
print("-" * 30)
print("Interpretation:")
print("These numbers represent the 'Information Power' of your dataset.")
print("If a simple statistical test gets high accuracy, your data is very clean.")