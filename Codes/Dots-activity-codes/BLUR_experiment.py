# BLUR_experiment.py

import cv2
import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import kendalltau
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt

ALGORITHM_TO_PLOT = 'CrowdBT'
NOISE_TYPE = 'GAUSSIAN_BLUR'

if NOISE_TYPE == 'GAUSSIAN_NOISE':
    NOISE_RANGE = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    GAUSSIAN_BLUR_KERNEL_SIZE = (9, 9)
elif NOISE_TYPE == 'GAUSSIAN_BLUR':
    GAUSSIAN_BLUR_KERNEL_SIZE = (7, 7)
    NOISE_RANGE = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

DATA_FILE = "all_data.json"
IMAGES_ROOT_FOLDER = "dots"
RESULTS_DIR = "RA_result_matlab"

def calculate_kendall_tau(s_estimate, s_true):
    tau, _ = kendalltau(s_estimate, s_true)
    return tau

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

def generate_sp_for_noise_level(noise_type, noise_value):
    all_predictions = defaultdict(list)
    subfolders = [d for d in os.listdir(IMAGES_ROOT_FOLDER) if os.path.isdir(os.path.join(IMAGES_ROOT_FOLDER, d))]
    for folder in subfolders:
        folder_path = os.path.join(IMAGES_ROOT_FOLDER, folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        for filename in image_files:
            try:
                ground_truth = int(os.path.splitext(filename)[0])
            except ValueError:
                continue
            clean_image = cv2.imread(os.path.join(folder_path, filename))
            if clean_image is None: continue
            if noise_type == 'GAUSSIAN_NOISE':
                noisy_image = add_gaussian_noise(clean_image, sigma=noise_value)
            elif noise_type == 'GAUSSIAN_BLUR':
                noisy_image = apply_gaussian_blur(clean_image, GAUSSIAN_BLUR_KERNEL_SIZE, sigma=noise_value)
            predicted_count = count_dots_opencv(noisy_image)
            if predicted_count is not None:
                all_predictions[ground_truth].append(predicted_count)
    s_p_vector = []
    sorted_gts = sorted(all_predictions.keys())
    for gt in sorted_gts:
        s_p_vector.append(np.mean(all_predictions.get(gt, [0])))
    return np.array(s_p_vector), sorted_gts

def aggregate_ratings_by_average(data_file_path, sorted_gts):
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    ratings_by_gt = defaultdict(list)
    for entry in data:
        for gt, rating in zip(entry['groundtruth'], entry['ratings']):
            ratings_by_gt[gt].append(rating)
    avg_ratings = {gt: np.mean(ratings_list) for gt, ratings_list in ratings_by_gt.items()}
    return np.array([avg_ratings[gt] for gt in sorted_gts])


def plot_radar_chart(df_plot_data, algorithm, noise_type):
    COLOR_MAP = {
        '$s_p$': "#2b78ac",
        '$\hat s$ (Rank)': "#25ab5d",
        '$\hat s$ (Rate)': "#CA83E8",
        '$s^*$': "#d77166"
    }

    labels = df_plot_data['noise_level'].astype(str).tolist()
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    lines_to_plot = {
        '$\hat s$ (Rank)': ('s_hat_ranking_tau', 'solid'),
        '$\hat s$ (Rate)': ('s_hat_rating_Avg_tau', '--'),
        '$s_p$': ('s_p_tau', '--'),
        '$s^*$': ('s_star_ranking_tau', '--')
    }

    for label, (column, style) in lines_to_plot.items():
        if label == '$s^*$':
            value = df_plot_data[column].iloc[0]
            values = [value] * num_vars
        else:
            values = df_plot_data[column].tolist()
        
        values += values[:1]
        
        ax.plot(angles, values, color=COLOR_MAP[label], linewidth=2.5, linestyle=style, label=label)
        if style == 'solid':
            ax.fill(angles, values, color=COLOR_MAP[label], alpha=0.1)

    ax.set_ylim(0.6, 1.0) 
    ax.set_rlabel_position(22.5)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_yticks(np.arange(0.6, 1.01, 0.1))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=35)
    ax.tick_params(axis='x', pad=17) 
    ax.grid(color='lightgrey', linestyle='--', linewidth=1)

    ax.spines['polar'].set_edgecolor('black')
    ax.spines['polar'].set_linewidth(1.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15), fontsize=35, 
              frameon=True, edgecolor='lightgrey')

    os.makedirs('figs_radar', exist_ok=True)
    pdf_filename = f"radar_plot_{algorithm}_{noise_type}.pdf"
    output_path = os.path.join('figs_radar', pdf_filename)
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)

def main():
    
    s_star_ranking_file = os.path.join(RESULTS_DIR, f'score_M1_{ALGORITHM_TO_PLOT}.txt')
    try:
        s_star_ranking_scores = np.loadtxt(s_star_ranking_file)
    except FileNotFoundError as e:
        print(f"Error: Unable to locate the aggregated score file for the algorithm '{ALGORITHM_TO_PLOT}': {e.filename}")
        return

    results_over_noise = []
    ir = IsotonicRegression(out_of_bounds='clip')

    for noise_level in NOISE_RANGE:
        print(f"\n--- Noise Level: {noise_level} ---")
        s_p, sorted_gts = generate_sp_for_noise_level(NOISE_TYPE, noise_level)
        s_true = np.array(sorted_gts)

        s_agg_rating_avg = aggregate_ratings_by_average(DATA_FILE, sorted_gts)
        
        # s_hat (from Rankings)
        s_hat_ranking = ir.fit_transform(s_star_ranking_scores, s_p)

        # s_hat (from Ratings_Avg)
        s_hat_rating_avg = ir.fit_transform(s_agg_rating_avg, s_p)

        results_over_noise.append({
            'noise_level': noise_level,
            's_p_tau': calculate_kendall_tau(s_p, s_true),
            's_star_ranking_tau': calculate_kendall_tau(s_star_ranking_scores, s_true),
            's_hat_ranking_tau': calculate_kendall_tau(s_hat_ranking, s_true),
            's_hat_rating_Avg_tau': calculate_kendall_tau(s_hat_rating_avg, s_true),
        })
        print(f"Noise Level {noise_level} done")

    if not results_over_noise:
        print("No results can be plotted.")
        return
        
    df_plot_data = pd.DataFrame(results_over_noise)
    print("\n--- The experiment is completed. Preparation for drawing is underway. ---")
    print(df_plot_data)

    plot_radar_chart(df_plot_data, ALGORITHM_TO_PLOT, NOISE_TYPE)

if __name__ == '__main__':
    main()