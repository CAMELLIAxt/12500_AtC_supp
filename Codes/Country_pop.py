import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import kendalltau, wasserstein_distance, ks_2samp
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", font_scale=1.5)


algorithms = {
    'HRA_G': 'Country_pop/Country_pop_HRA_G_scores.xlsx',
    'HRA_E': 'Country_pop/Country_pop_HRA_E_scores.xlsx',
    'HRA_N': 'Country_pop/Country_pop_HRA_N_scores.xlsx'
}

all_pair_path = 'Country_pop/Country_pop_all_pair.txt'
doc_info_path = 'Country_pop/Country_pop_doc_info.txt'
s_p_path = 'Country_pop/s_p.txt'

all_pair = np.loadtxt(all_pair_path, dtype=int)
doc_info = np.loadtxt(doc_info_path) 
true_scores = doc_info 
s_p = np.loadtxt(s_p_path)

def center_scores(scores):
    return scores - np.mean(scores)

n_items = 15 
true_scores = center_scores(true_scores)
s_p = center_scores(s_p)


def compute_kde_density(scores, x_grid, bw_adjust=0.8):
    kde = sns.kdeplot(scores, bw_adjust=bw_adjust, gridsize=500)
    density = kde.get_lines()[0].get_data()[1]
    kde.get_lines()[0].remove()
    integral = simps(density, x_grid)
    density = density / integral
    density = np.clip(density, 1e-6, None)
    return density

def kl_divergence(p, q):
    return np.sum(p * np.log(p / q + 1e-5))

def js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m + 1e-5))
    kl_qm = np.sum(q * np.log(q / m + 1e-5))
    return 0.5 * (kl_pm + kl_qm)

def count_inversions(rank1, rank2):
    n = len(rank1)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            if rank1[i] > rank1[j] and rank2[i] < rank2[j]:
                inversions += 1
            elif rank1[i] < rank1[j] and rank2[i] > rank2[j]:
                inversions += 1
    return inversions


x_range = max(np.max(true_scores), np.max(s_p)) - min(np.min(true_scores), np.min(s_p))
x_min = min(np.min(true_scores), np.min(s_p)) - 0.1 * x_range
x_max = max(np.max(true_scores), np.max(s_p)) + 0.1 * x_range
x_grid = np.linspace(x_min, x_max, 500)

true_density = compute_kde_density(true_scores, x_grid)
s_p_density = compute_kde_density(s_p, x_grid)

metrics_list = []

for algo_name, scores_file in algorithms.items():
    print(f"Processing {algo_name}...")

    scores_path = f'{scores_file}'
    s_star = pd.read_excel(scores_path, header=None).values.flatten()
    s_star = center_scores(s_star)

    x_range = max(np.max(true_scores), np.max(s_p), np.max(s_star)) - \
              min(np.min(true_scores), np.min(s_p), np.min(s_star))
    x_min = min(np.min(true_scores), np.min(s_p), np.min(s_star)) - 0.1 * x_range
    x_max = max(np.max(true_scores), np.max(s_p), np.max(s_star)) + 0.1 * x_range
    x_grid = np.linspace(x_min, x_max, 500)

    true_density = compute_kde_density(true_scores, x_grid)
    s_p_density = compute_kde_density(s_p, x_grid)

    order = np.argsort(s_star)
    X = np.arange(n_items)
    y = s_p[order]
    iso_reg = IsotonicRegression(increasing=True, out_of_bounds='clip')
    hat_s_ordered = iso_reg.fit_transform(X, y)
    hat_s = np.zeros(n_items)
    hat_s[order] = hat_s_ordered
    hat_s = center_scores(hat_s)

    x_range = max(np.max(true_scores), np.max(s_p), np.max(s_star), np.max(hat_s)) - \
              min(np.min(true_scores), np.min(s_p), np.min(s_star), np.min(hat_s))
    x_min = min(np.min(true_scores), np.min(s_p), np.min(s_star), np.min(hat_s)) - 0.1 * x_range
    x_max = max(np.max(true_scores), np.max(s_p), np.max(s_star), np.max(hat_s)) + 0.1 * x_range
    x_grid = np.linspace(x_min, x_max, 500)

    true_density = compute_kde_density(true_scores, x_grid)
    s_p_density = compute_kde_density(s_p, x_grid)
    s_star_density = compute_kde_density(s_star, x_grid)
    hat_s_density = compute_kde_density(hat_s, x_grid)

    kl_s_star = kl_divergence(true_density, s_star_density)
    kl_s_p = kl_divergence(true_density, s_p_density)
    kl_hat_s = kl_divergence(true_density, hat_s_density)

    js_s_star = js_divergence(true_density, s_star_density)
    js_s_p = js_divergence(true_density, s_p_density)
    js_hat_s = js_divergence(true_density, hat_s_density)

    metrics_list.append({
        'Algorithm': algo_name,
        'KL_s_star': kl_s_star,
        'KL_s_p': kl_s_p,
        'KL_hat_s': kl_hat_s,
        'JS_s_star': js_s_star,
        'JS_s_p': js_s_p,
        'JS_hat_s': js_hat_s
    })


metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)