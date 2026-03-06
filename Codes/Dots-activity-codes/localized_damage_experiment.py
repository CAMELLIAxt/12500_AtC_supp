# localized_damage_experiment.py

import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import kendalltau
from sklearn.isotonic import IsotonicRegression

from sp_generator import generate_sp_vector

ALGORITHM_TO_RUN = 'BTL'


EFFECT_CONFIGS = [
    {
        'type': 'WHITEOUT',
        'params': {} # WHITEOUT 
    },
    {
        'type': 'LOCAL_NOISE',
        'params': {'sigma': 6}
    },
    {
        'type': 'LOCAL_BLUR',
        'params': {'sigma': 6, 'kernel_size': (11, 11)} 
    }
]

REGION_RATIO_RANGE = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

DATA_FILE = "all_data.json"
RESULTS_DIR = "RA_result_matlab"
DATA_OUTPUT_DIR = "experimental_data2"

def calculate_kendall_tau(s_estimate, s_true):
    tau, _ = kendalltau(s_estimate, s_true)
    return tau

def aggregate_ratings_by_average(data_file_content, sorted_gts):
    ratings_by_gt = defaultdict(list)
    for entry in data_file_content:
        for gt, rating in zip(entry['groundtruth'], entry['ratings']):
            ratings_by_gt[gt].append(rating)
    avg_ratings = {gt: np.mean(lst) for gt, lst in ratings_by_gt.items()}
    return np.array([avg_ratings.get(gt, 0) for gt in sorted_gts])


def main():
    with open(DATA_FILE, 'r') as f:
        json_data = json.load(f)
    
    s_star_ranking_scores = np.loadtxt(os.path.join(RESULTS_DIR, f'score_M1_{ALGORITHM_TO_RUN}.txt'))
    s_agg_rating_ra_scores = np.loadtxt(os.path.join(RESULTS_DIR, f'score_M2_{ALGORITHM_TO_RUN}.txt'))

    dummy_sp, sorted_gts = generate_sp_vector({'type': 'WHITEOUT', 'region_ratio': 0.0, 'params': {}}, save_images=False)
    s_true = np.array(sorted_gts)
    s_agg_rating_avg = aggregate_ratings_by_average(json_data, s_true)
    
    s_star_ranking_tau = calculate_kendall_tau(s_star_ranking_scores, s_true)

    for effect_config_template in EFFECT_CONFIGS:
        effect_type = effect_config_template['type']
        
        results_list = []
        ir = IsotonicRegression(out_of_bounds='clip')

        for ratio in REGION_RATIO_RANGE:
            current_config = effect_config_template.copy()
            current_config['region_ratio'] = ratio
            s_p, _ = generate_sp_vector(effect_config=current_config, save_images=False)
            
            s_p_tau = calculate_kendall_tau(s_p, s_true)
            
            s_hat_ranking = ir.fit_transform(s_star_ranking_scores, s_p)
            s_hat_ranking_tau = calculate_kendall_tau(s_hat_ranking, s_true)
            
            s_hat_rating_avg = ir.fit_transform(s_agg_rating_avg, s_p)
            s_hat_rating_Avg_tau = calculate_kendall_tau(s_hat_rating_avg, s_true)

            results_list.append({
                'region_ratio': ratio,
                's_star_ranking_tau': s_star_ranking_tau,
                's_p_tau': s_p_tau,
                's_hat_ranking_tau': s_hat_ranking_tau,
                's_hat_rating_Avg_tau': s_hat_rating_Avg_tau,
            })
            
        df_results = pd.DataFrame(results_list)
        os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
        
        filename = f"results_{ALGORITHM_TO_RUN}_{effect_type}.csv"
        output_path = os.path.join(DATA_OUTPUT_DIR, filename)
        
        df_results.to_csv(output_path, index=False, float_format='%.6f')

if __name__ == "__main__":
    main()