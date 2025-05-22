import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.integrate import simps
import seaborn as sns
sns.set(style="whitegrid", palette="deep", font_scale=1.5)

algorithms = {
    'HRA_G': 'Datas/Reading_level_datas_reg10/reg10_score_Ones_+_HRA-G_s->a.txt',
    'HRA_E': 'Datas/Reading_level_datas_reg10/reg10_score_Ones_+_HRA-E_s->a.txt',
    'HRA_N': 'Datas/Reading_level_datas_reg10/reg10_score_Ones_+_HRA-N_s->a.txt'
}

doc_info_path = 'Datas/Reading_level_datas_reg10/Reading_level_doc_info.txt'
s_p_path = 'Datas/Reading_level_datas_reg10/s_p.txt'


doc_info = np.loadtxt(doc_info_path)
true_scores = doc_info[:, 1]
s_p = np.loadtxt(s_p_path)


def center_scores(scores):
    return scores - np.mean(scores)

n_items = 490
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


def count_inversions_adjusted(scores_a, scores_b):
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("The lengths of the two sequence of fractions must be the same.")

    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            order_a_ij = 0 # 0: s_a[i] == s_a[j], 1: s_a[i] < s_a[j], -1: s_a[i] > s_a[j]
            if scores_a[i] < scores_a[j]:
                order_a_ij = 1
            elif scores_a[i] > scores_a[j]:
                order_a_ij = -1

            order_b_ij = 0 # 0: s_b[i] == s_b[j], 1: s_b[i] < s_b[j], -1: s_b[i] > s_b[j]
            if scores_b[i] < scores_b[j]:
                order_b_ij = 1
            elif scores_b[i] > scores_b[j]:
                order_b_ij = -1

            if order_a_ij != 0 and order_b_ij != 0 and order_a_ij * order_b_ij == -1:
                inversions += 1
    return inversions

# tau
def kendall_tau_b_adjusted(scores_a, scores_b):
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("The lengths of the two sequence of fractions must be the same.")
    if n < 2:
        return np.nan 

    num_concordant = 0  
    num_discordant = 0
    num_ties_a = 0     
    num_ties_b = 0

    for i in range(n):
        for j in range(i + 1, n):
            val_a_i, val_a_j = scores_a[i], scores_a[j]
            val_b_i, val_b_j = scores_b[i], scores_b[j]

            sign_a = 0
            if val_a_i < val_a_j: sign_a = 1
            elif val_a_i > val_a_j: sign_a = -1

            sign_b = 0
            if val_b_i < val_b_j: sign_b = 1
            elif val_b_i > val_b_j: sign_b = -1

            if sign_a != 0 and sign_b != 0: 
                if sign_a == sign_b:
                    num_concordant += 1
                else: # sign_a * sign_b == -1
                    num_discordant += 1
            elif sign_a == 0 and sign_b != 0:
                num_ties_a += 1
            elif sign_a != 0 and sign_b == 0: 
                num_ties_b += 1

    numerator = num_concordant - num_discordant
    n_pairs = n * (n - 1) / 2.0
    if n_pairs == 0: return np.nan

    ties_x_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if scores_a[i] == scores_a[j]:
                ties_x_total += 1

    ties_y_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if scores_b[i] == scores_b[j]:
                ties_y_total += 1
    
    # Calculate P
    P = n * (n - 1) / 2.0

    # Calculate sum_t_half for scores_a
    _, counts_a = np.unique(scores_a, return_counts=True)
    sum_t_half = np.sum(counts_a * (counts_a - 1) / 2.0)

    # Calculate sum_u_half for scores_b
    _, counts_b = np.unique(scores_b, return_counts=True)
    sum_u_half = np.sum(counts_b * (counts_b - 1) / 2.0)

    denominator_val_part1 = P - sum_t_half
    denominator_val_part2 = P - sum_u_half

    if denominator_val_part1 <= 0 or denominator_val_part2 <= 0:
        return 0.0 if numerator == 0 else np.nan

    tau_b = numerator / np.sqrt(denominator_val_part1 * denominator_val_part2)
    return tau_b


x_range = max(np.max(true_scores), np.max(s_p)) - min(np.min(true_scores), np.min(s_p))
x_min = min(np.min(true_scores), np.min(s_p)) - 0.1 * x_range
x_max = max(np.max(true_scores), np.max(s_p)) + 0.1 * x_range
x_grid = np.linspace(x_min, x_max, 500)

true_density = compute_kde_density(true_scores, x_grid)
s_p_density = compute_kde_density(s_p, x_grid)

metrics_list = []

for algo_name, scores_file in algorithms.items():
    print(f"Processing {algo_name}...")

    s_star = np.loadtxt(scores_file)
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

    tau_s_star = kendall_tau_b_adjusted(s_star, true_scores)
    tau_s_p = kendall_tau_b_adjusted(s_p, true_scores)
    tau_hat_s = kendall_tau_b_adjusted(hat_s, true_scores)

    l2_dist_s_star = np.sqrt(np.sum((s_star - true_scores) ** 2))
    l2_dist_s_p = np.sqrt(np.sum((s_p - true_scores) ** 2))
    l2_dist_hat_s = np.sqrt(np.sum((hat_s - true_scores) ** 2))

    mse_s_star = np.mean((s_star - true_scores) ** 2)
    mse_s_p = np.mean((s_p - true_scores) ** 2)
    mse_hat_s = np.mean((hat_s - true_scores) ** 2)

    wass_s_star = wasserstein_distance(true_scores, s_star)
    wass_s_p = wasserstein_distance(true_scores, s_p)
    wass_hat_s = wasserstein_distance(true_scores, hat_s)

    ks_s_star = ks_2samp(true_scores, s_star)[0]
    ks_s_p = ks_2samp(true_scores, s_p)[0]
    ks_hat_s = ks_2samp(true_scores, hat_s)[0]


    metrics_list.append({
        'Algorithm': algo_name,
        'Kendall_Tau_s_star': tau_s_star,
        'Kendall_Tau_s_p': tau_s_p,
        'Kendall_Tau_hat_s': tau_hat_s,
        'L2_dist_s_star': l2_dist_s_star,
        'L2_dist_s_p': l2_dist_s_p,
        'L2_dist_hat_s': l2_dist_hat_s,
        'MSE_s_star': mse_s_star,
        'MSE_s_p': mse_s_p,
        'MSE_hat_s': mse_hat_s,
        'Wasserstein_s_star': wass_s_star,
        'Wasserstein_s_p': wass_s_p,
        'Wasserstein_hat_s': wass_hat_s,
        'KS_s_star': ks_s_star,
        'KS_s_p': ks_s_p,
        'KS_hat_s': ks_hat_s,
    })

metrics_df = pd.DataFrame(metrics_list)
print(metrics_df)