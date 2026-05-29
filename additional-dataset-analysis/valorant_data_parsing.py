import os
import cv2
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
CSV_PATH = os.path.join(TRAIN_DIR, "_classes.csv")
IMAGE_SIZE = (64, 64)

def build_valorant_dataset():
    df = pd.read_csv(CSV_PATH)
    
    X_data = []
    y_data = []
    
    for index, row in df.iterrows():
        # iloc[0] is the filename, iloc[1] is the first class column (Enemy)
        filename = row.iloc[0] 
        is_enemy = int(row.iloc[1])
        
        img_path = os.path.join(TRAIN_DIR, filename)
        
        if not os.path.exists(img_path):
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Convert to grayscale and shrink to 64x64
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, IMAGE_SIZE)
        
        # Flatten to 1D array (4096 features)
        flattened_features = resized_img.flatten()
        
        X_data.append(flattened_features)
        y_data.append(is_enemy)
        
    X_raw = np.array(X_data)
    y_raw = np.array(y_data)
    
    print(f"\nSuccessfully processed {len(X_raw)} images.")
    print(f"X shape: {X_raw.shape} | y shape: {y_raw.shape}")
    
    enemies_count = np.sum(y_raw == 1)
    no_enemies_count = np.sum(y_raw == 0)
    print(f"Class Distribution -> Enemy: {enemies_count} | No Enemy: {no_enemies_count}")
    
    return X_raw, y_raw

if __name__ == "__main__":
    X, y = build_valorant_dataset()