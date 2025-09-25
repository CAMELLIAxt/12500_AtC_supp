# generate_s_p.py

import cv2
import os
import json
import numpy as np

IMAGES_ROOT_FOLDER = "dots"  
OUTPUT_FILE = "s_p_vector_averaged.json"

def count_dots_opencv(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[warning] cannot read: {image_path}, skip...")
        return None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return len(contours)

if __name__ == "__main__":
    if not os.path.exists(IMAGES_ROOT_FOLDER):
        print(f"error:cannot find '{IMAGES_ROOT_FOLDER}'")
        exit()

    subfolders = [d for d in os.listdir(IMAGES_ROOT_FOLDER) if os.path.isdir(os.path.join(IMAGES_ROOT_FOLDER, d))]
    if not subfolders:
        print(f"error: '{IMAGES_ROOT_FOLDER}' has no subfolders")
        exit()

    all_predictions = {}

    for folder in subfolders:
        folder_path = os.path.join(IMAGES_ROOT_FOLDER, folder)
        
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        for filename in image_files:
            try:
                ground_truth = int(os.path.splitext(filename)[0])
            except ValueError:
                print(f"  [warning] '{filename}' ignored (not an integer filename)")
                continue

            if ground_truth not in all_predictions:
                all_predictions[ground_truth] = []
            
            full_path = os.path.join(folder_path, filename)
            predicted_count = count_dots_opencv(full_path)
            if predicted_count is not None:
                all_predictions[ground_truth].append(predicted_count)

    s_p_vector = []

    sorted_gts = sorted(all_predictions.keys())
    
    for gt in sorted_gts:
        preds_for_gt = all_predictions[gt]
        if not preds_for_gt:
            print(f"[warning] No predictions for GT={gt}, setting average to 0")
            average_prediction = 0 
        else:
            average_prediction = np.mean(preds_for_gt)
        
        s_p_vector.append(average_prediction)
        
        print(f"GT={gt}: Predictions={preds_for_gt}, Average={average_prediction:.2f}")

    print("s_p_vector (averaged):")
    print([f"{val:.2f}" for val in s_p_vector])
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(s_p_vector, f)
        
    print(f"\nsave to: '{OUTPUT_FILE}'")