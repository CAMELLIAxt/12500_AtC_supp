# sp_generator.py

import cv2
import os
import json
import numpy as np
from collections import defaultdict


EFFECT_CONFIG = {
    # Type: 'WHITEOUT', 'LOCAL_NOISE', 'LOCAL_BLUR'
    'type': 'LOCAL_BLUR',
    
    # (0.0 to 1.0)
    'region_ratio': 0.5,
    
    'params': {
        'sigma': 50,         # LOCAL_NOISE & LOCAL_BLUR
        'kernel_size': (15, 15) # LOCAL_BLUR
    }
}

IMAGES_ROOT_FOLDER = "dots_noisy_blur_k9_s3"

def add_gaussian_noise(image, sigma):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy_image.astype(np.uint8)

def apply_gaussian_blur(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, kernel_size, sigma)

def count_dots_opencv(image):
    if image is None: return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def apply_localized_effect(image, config):
    h, w = image.shape[:2]
    ratio = config['region_ratio']
    effect_type = config['type']
    params = config['params']
    
    start_x = int(w * (1 - ratio))
    start_y = int(h * (1 - ratio))
    
    output_image = image.copy()

    if effect_type == 'WHITEOUT':
        cv2.rectangle(output_image, (start_x, start_y), (w, h), (255, 255, 255), -1)
        
    elif effect_type in ['LOCAL_NOISE', 'LOCAL_BLUR']:
        roi = output_image[start_y:h, start_x:w]
        
        if effect_type == 'LOCAL_NOISE':
            processed_roi = add_gaussian_noise(roi, params['sigma'])
        else: # LOCAL_BLUR
            processed_roi = apply_gaussian_blur(roi, params['kernel_size'], params['sigma'])
        output_image[start_y:h, start_x:w] = processed_roi
        
    return output_image

def generate_sp_vector(effect_config, images_root_folder="dots", save_images=True):
    all_predictions = defaultdict(list)
    
    if save_images:
        output_folder_name = f"dots_plus_{effect_config['type'].lower()}"
        print(f"Save: '{output_folder_name}'")

    subfolders = [d for d in os.listdir(images_root_folder) if os.path.isdir(os.path.join(images_root_folder, d))]
    
    for folder in subfolders:
        folder_path = os.path.join(images_root_folder, folder)
        
        if save_images:
            processed_folder_path = os.path.join(output_folder_name, folder)
            os.makedirs(processed_folder_path, exist_ok=True)

        for filename in [f for f in os.listdir(folder_path) if f.endswith('.png')]:
            try:
                ground_truth = int(os.path.splitext(filename)[0])
            except ValueError:
                continue

            clean_image = cv2.imread(os.path.join(folder_path, filename))
            if clean_image is None: continue

            processed_image = apply_localized_effect(clean_image, effect_config)
            
            if save_images:
                cv2.imwrite(os.path.join(processed_folder_path, filename), processed_image)
            
            predicted_count = count_dots_opencv(processed_image)
            if predicted_count is not None:
                all_predictions[ground_truth].append(predicted_count)

    sorted_gts = sorted(all_predictions.keys())
    s_p_vector = [np.mean(all_predictions.get(gt, [0])) for gt in sorted_gts]
    
    return np.array(s_p_vector), sorted_gts


if __name__ == "__main__":
    
    s_p_vector, sorted_gts = generate_sp_vector(
        effect_config=EFFECT_CONFIG,
        images_root_folder=IMAGES_ROOT_FOLDER,
        save_images=True
    )
    
    print("\n--- All the images have been processed and the average value has been calculated. ---")
    for gt, pred in zip(sorted_gts, s_p_vector):
        print(f"  GT: {gt:2d} -> average: {pred:.2f}")

    config_str = f"plus_{EFFECT_CONFIG['type'].lower()}_r{EFFECT_CONFIG['region_ratio']}"
    if 'sigma' in EFFECT_CONFIG['params']:
        config_str += f"_s{EFFECT_CONFIG['params']['sigma']}"
    if 'kernel_size' in EFFECT_CONFIG['params']:
        config_str += f"_k{EFFECT_CONFIG['params']['kernel_size'][0]}"
        
    output_filename = f"s_p_{config_str}.json"
    
    with open(output_filename, 'w') as f:
        json.dump(s_p_vector.tolist(), f)
        
    print(f"\n✅ s_p saved in: '{output_filename}'")