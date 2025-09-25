# generate_noisy_sp.py (Gaussian Noise & Blur Version)

import cv2
import os
import json
import numpy as np


# type: 'GAUSSIAN_NOISE' or 'GAUSSIAN_BLUR'
NOISE_TYPE = 'GAUSSIAN_BLUR' 

GAUSSIAN_NOISE_SIGMA = 50

GAUSSIAN_BLUR_KERNEL_SIZE = (9, 9)
GAUSSIAN_BLUR_SIGMA = 5


IMAGES_ROOT_FOLDER = "dots"
NOISY_IMAGES_OUTPUT_FOLDER = "dots_noisy"


def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    image_float = image.astype(np.float32)
    noisy_image = image_float + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_gaussian_blur(image, kernel_size, sigma):
    if not (kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1):
        raise ValueError("Gaussian Blur Kernel Size must be an odd-numbered tuple., e.g., (5, 5)")
    return cv2.GaussianBlur(image, kernel_size, sigma)

def count_dots_opencv(image):
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

if __name__ == "__main__":
    
    if NOISE_TYPE == 'GAUSSIAN_NOISE':
        output_filename = f"s_p_vector_noise_s{GAUSSIAN_NOISE_SIGMA}.json"
        print(f"--- mode: GAUSSIAN_NOISE (Sigma: {GAUSSIAN_NOISE_SIGMA}) ---")
    elif NOISE_TYPE == 'GAUSSIAN_BLUR':
        k = GAUSSIAN_BLUR_KERNEL_SIZE[0]
        s = GAUSSIAN_BLUR_SIGMA
        output_filename = f"s_p_vector_blur_k{k}_s{s}.json"
        print(f"--- mode: GAUSSIAN_BLUR (Kernel: {GAUSSIAN_BLUR_KERNEL_SIZE}, Sigma: {s}) ---")
    else:
        raise ValueError(f"Unsupported NOISE_TYPE: {NOISE_TYPE}")

    subfolders = [d for d in os.listdir(IMAGES_ROOT_FOLDER) if os.path.isdir(os.path.join(IMAGES_ROOT_FOLDER, d))]
    all_predictions = {}

    for folder in subfolders:
        folder_path = os.path.join(IMAGES_ROOT_FOLDER, folder)
        noisy_folder_path = os.path.join(NOISY_IMAGES_OUTPUT_FOLDER, folder)
        os.makedirs(noisy_folder_path, exist_ok=True)

        print(f"\nProcessing folder: '{folder}' -> '{noisy_folder_path}'")
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        
        for filename in image_files:
            try:
                ground_truth = int(os.path.splitext(filename)[0])
            except ValueError:
                continue

            if ground_truth not in all_predictions:
                all_predictions[ground_truth] = []
            
            clean_image_path = os.path.join(folder_path, filename)
            clean_image = cv2.imread(clean_image_path)
            if clean_image is None: continue

            if NOISE_TYPE == 'GAUSSIAN_NOISE':
                noisy_image = add_gaussian_noise(clean_image, GAUSSIAN_NOISE_SIGMA)
            elif NOISE_TYPE == 'GAUSSIAN_BLUR':
                noisy_image = apply_gaussian_blur(clean_image, GAUSSIAN_BLUR_KERNEL_SIZE, GAUSSIAN_BLUR_SIGMA)
            
            cv2.imwrite(os.path.join(noisy_folder_path, filename), noisy_image)
            
            predicted_count = count_dots_opencv(noisy_image)
            if predicted_count is not None:
                all_predictions[ground_truth].append(predicted_count)

    s_p_vector = []
    sorted_gts = sorted(all_predictions.keys())
    
    for gt in sorted_gts:
        preds_for_gt = all_predictions.get(gt, [])
        average_prediction = np.mean(preds_for_gt) if preds_for_gt else 0
        s_p_vector.append(average_prediction)
        print(f"GT={gt}: Predictions={preds_for_gt}, Average={average_prediction:.2f}")

    with open(output_filename, 'w') as f:
        json.dump(s_p_vector, f)
    print(f"\ns_p_vector saved to '{output_filename}'")