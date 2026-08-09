import os
import json
import shutil

# ==========================================
# 1. SETUP CREDENTIALS (Auto-Login)
# ==========================================
# ⚠️ REPLACE 'YOUR_KAGGLE_USERNAME_HERE' WITH YOUR ACTUAL USERNAME ⚠️
KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME_HERE" 

# I have extracted the key from the token you provided:
# (I removed the 'KGAT_' prefix assuming it was a label, using the 32-char hex key)
KAGGLE_KEY = "f5a3a79378d3a2490d96d6a5c876a91b"

# Create the kaggle.json file programmatically
kaggle_creds = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}

# Save it to the location where the API looks for it
os.makedirs('/root/.kaggle', exist_ok=True) # For Colab/Linux
with open('kaggle.json', 'w') as f:
    json.dump(kaggle_creds, f)

# Also set environment variables just in case
os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

# Move creds to standard location if on Local Machine (Windows/Mac)
if not os.path.exists(os.path.expanduser('~/.kaggle')):
    os.makedirs(os.path.expanduser('~/.kaggle'))
shutil.copy('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))

print(f"✅ Credentials set for user: {KAGGLE_USERNAME}")

# ==========================================
# 2. DOWNLOAD THE DATASET
# ==========================================
try:
    import opendatasets as od
except ImportError:
    import pip
    pip.main(['install', 'opendatasets'])
    import opendatasets as od

dataset_url = "https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone"

print("⬇️ Starting Download (No typing required)...")
od.download(dataset_url)

# ==========================================
# 3. REORGANIZE FOLDERS
# ==========================================
downloaded_folder = "ct-kidney-dataset-normal-cyst-tumor-and-stone"
target_folder = "kidney_images"

if os.path.exists(downloaded_folder):
    # If target already exists, clean it up
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    
    # Rename
    os.rename(downloaded_folder, target_folder)
    
    # Fix nested folders if necessary
    subfolders = os.listdir(target_folder)
    nested_name = 'CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone'
    
    if nested_name in subfolders:
        print("🔧 Fixing folder structure...")
        nested_path = os.path.join(target_folder, nested_name)
        for item in os.listdir(nested_path):
            shutil.move(os.path.join(nested_path, item), target_folder)
        os.rmdir(nested_path)

    print("\n" + "="*40)
    print("🎉 SUCCESS! Images downloaded and ready.")
    print(f"📂 Location: {os.path.abspath(target_folder)}")
    print("="*40)
else:
    print("❌ Download failed. Please check your Username.")