import os
import cv2
import pandas as pd
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (64, 64) 
SAMPLES_PER_CLASS = 250

def generate_sampled_csv():
    folder_dir = os.path.join(BASE_DIR, "train")
    classes_csv = os.path.join(folder_dir, "_classes.csv")
    output_filename = f"valorant_dataset_{SAMPLES_PER_CLASS * 2}_color.csv"

    if not os.path.exists(classes_csv):
        print("Error: _classes.csv not found in the train folder.")
        return

    df = pd.read_csv(classes_csv)
    
    print("Balancing and sampling dataset...")
    
    df_enemies = df[df.iloc[:, 1] == 1]
    df_no_enemies = df[df.iloc[:, 1] == 0]
    
    df_enemies_sampled = df_enemies.sample(n=SAMPLES_PER_CLASS, random_state=42)
    df_no_enemies_sampled = df_no_enemies.sample(n=SAMPLES_PER_CLASS, random_state=42)
    
    df_sampled = pd.concat([df_enemies_sampled, df_no_enemies_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    all_rows = []
    print(f"Processing {len(df_sampled)} images into a single CSV...")

    for index, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc="Processing Images"):
        filename = row.iloc[0]
        is_enemy = int(row.iloc[1])

        img_path = os.path.join(folder_dir, filename)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        resized_img = cv2.resize(rgb_img, IMAGE_SIZE)
        flattened_features = resized_img.flatten().tolist()

        flattened_features.append(is_enemy)
        all_rows.append(flattened_features)

    total_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3
    feature_cols = [f"pixel_{i}" for i in range(total_pixels)]
    columns = feature_cols + ["label"]

    out_df = pd.DataFrame(all_rows, columns=columns)
    out_path = os.path.join(BASE_DIR, output_filename)
    out_df.to_csv(out_path, index=False)
    
    print(f"\nSUCCESS: Saved {len(all_rows)} images to {output_filename}!")

if __name__ == "__main__":
    generate_sampled_csv()