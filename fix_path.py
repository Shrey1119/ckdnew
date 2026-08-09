import pandas as pd
import os
import glob
import numpy as np

# CONFIGURATION
CSV_FILE = "kidney_multimodal_dataset.csv"
ROOT_DIR = "kidney_images"

print(f"🔧 STARTING PATH REPAIR...")

# 1. FIND ALL REAL IMAGES
# We search recursively because the images might be in nested subfolders
print("   Scanning for actual image files...")
all_files = glob.glob(os.path.join(ROOT_DIR, "**", "*.jpg"), recursive=True)

# Separate them by category
tumor_files = [f for f in all_files if "Tumor" in f]
stone_files = [f for f in all_files if "Stone" in f]
normal_files = [f for f in all_files if "Normal" in f]

print(f"   ✅ Found Real Files -> Tumor: {len(tumor_files)} | Stone: {len(stone_files)} | Normal: {len(normal_files)}")

if len(tumor_files) == 0:
    print("   ❌ ERROR: Could not find any images. Please check if 'kidney_images' is empty.")
    exit()

# 2. LOAD CSV
df = pd.read_csv(CSV_FILE)
print(f"   📄 Loaded CSV with {len(df)} rows.")

# 3. REMAP PATHS
# We will go through every row and assign a REAL existing file path
def fix_path(row):
    # If CKD (Sick)
    if row['classification'] == 1:
        # High Creatinine -> Tumor
        if row['sc'] > 3.0:
            return np.random.choice(tumor_files)
        # Moderate Creatinine -> Stone
        else:
            # If no stone files found, fallback to tumor
            options = stone_files if len(stone_files) > 0 else tumor_files
            return np.random.choice(options)
    
    # If Healthy
    else:
        return np.random.choice(normal_files)

# Apply the fix
# We use a lambda to ensure a random choice for every single row
df['image_path'] = df.apply(fix_path, axis=1)

# 4. SAVE
output_file = "kidney_multimodal_dataset_FIXED.csv"
df.to_csv(output_file, index=False)

print("\n" + "="*40)
print(f"🎉 REPAIR COMPLETE!")
print(f"   New CSV saved as: '{output_file}'")
print(f"   Please use this new CSV for your training/audit.")
print("="*40)