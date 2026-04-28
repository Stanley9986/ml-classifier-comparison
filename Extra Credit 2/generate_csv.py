import os
import cv2
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_SIZE = (32, 32)

def generate_single_csv():
    folder_dir = os.path.join(BASE_DIR, "train")
    classes_csv = os.path.join(folder_dir, "_classes.csv")
    output_filename = "valorant_dataset.csv"

    if not os.path.exists(classes_csv):
        print("Error: _classes.csv not found in the train folder.")
        return

    df = pd.read_csv(classes_csv)
    
    all_rows = []
    print("Compiling dataset into CSV")

    for index, row in df.iterrows():
        filename = row.iloc[0]
        is_enemy = int(row.iloc[1])

        img_path = os.path.join(folder_dir, filename)

        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(gray_img, IMAGE_SIZE)
        flattened_features = resized_img.flatten().tolist()
        flattened_features.append(is_enemy)
        
        all_rows.append(flattened_features)

    feature_cols = [f"pixel_{i}" for i in range(IMAGE_SIZE[0] * IMAGE_SIZE[1])]
    columns = feature_cols + ["label"]

    out_df = pd.DataFrame(all_rows, columns=columns)
    out_path = os.path.join(BASE_DIR, output_filename)
    out_df.to_csv(out_path, index=False)
    
    print(f"Saved {len(all_rows)} images to {output_filename}")

if __name__ == "__main__":
    generate_single_csv()