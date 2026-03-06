import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error

from sklearn.isotonic import IsotonicRegression as isotonic_regression_sklearn

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    print("Pandas not installed. Using dictionary-based plotting.")
    PANDAS_AVAILABLE = False

np.random.seed(42)
FIGURE_OUTPUT_DIR = "iso_fig_final"
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

def count_inversions_adjusted(scores_a, scores_b):
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("The lengths of the two sequence of fractions must be the same.")

    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            order_a_ij = 0
            if scores_a[i] < scores_a[j]: order_a_ij = 1
            elif scores_a[i] > scores_a[j]: order_a_ij = -1

            order_b_ij = 0
            if scores_b[i] < scores_b[j]: order_b_ij = 1
            elif scores_b[i] > scores_b[j]: order_b_ij = -1

            if order_a_ij != 0 and order_b_ij != 0 and order_a_ij * order_b_ij == -1:
                inversions += 1
    return inversions

def kendall_tau_b_adjusted(scores_a, scores_b):
    n = len(scores_a)
    if n != len(scores_b):
        raise ValueError("The lengths of the two sequence of fractions must be the same.")
    if n < 2:
        return np.nan

    num_concordant = 0
    num_discordant = 0

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
                else:
                    num_discordant += 1

    numerator = num_concordant - num_discordant
    P = n * (n - 1) / 2.0
    if P == 0: return np.nan 

    _, counts_a = np.unique(scores_a, return_counts=True)
    sum_t_half = np.sum(counts_a * (counts_a - 1) / 2.0)

    _, counts_b = np.unique(scores_b, return_counts=True)
    sum_u_half = np.sum(counts_b * (counts_b - 1) / 2.0)

    denominator_val_part1 = P - sum_t_half
    denominator_val_part2 = P - sum_u_half

    if denominator_val_part1 <= 0 or denominator_val_part2 <= 0:
        return 0.0 if numerator == 0 else np.nan

    tau_b = numerator / np.sqrt(denominator_val_part1 * denominator_val_part2)
    return tau_b


def generate_sequence_with_target_inversions(n, target_inversions, base_slope=1.0, noise_std_factor=0.1, tolerance=10, max_iterations=1000):
    x_base = np.arange(n)
    noise_std = n * noise_std_factor if n > 0 else 0
    y = base_slope * x_base + np.random.normal(0, noise_std, n)
    
    max_inversions = n * (n - 1) // 2
    target_inversions = min(max(target_inversions, 0), max_inversions)
    
    def count_inversions_orig(arr): 
        def merge_and_count(left, right):
            merged = []
            count = 0
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    merged.append(left[i])
                    i += 1
                else:
                    merged.append(right[j])
                    count += len(left) - i
                    j += 1
            merged.extend(left[i:])
            merged.extend(right[j:])
            return merged, count

        def sort_and_count(arr_inner):
            if len(arr_inner) <= 1:
                return arr_inner, 0
            mid = len(arr_inner) // 2
            left, left_count = sort_and_count(arr_inner[:mid])
            right, right_count = sort_and_count(arr_inner[mid:])
            merged, merge_count = merge_and_count(left, right)
            return merged, left_count + right_count + merge_count

        _, inversions = sort_and_count(list(arr))
        return inversions
    
    current_inversions = count_inversions_orig(y)
    iteration = 0
    
    while abs(current_inversions - target_inversions) > tolerance and iteration < max_iterations:
        y_temp = y.copy()
        idx1, idx2 = np.random.choice(n, 2, replace=False)
        y_temp[idx1], y_temp[idx2] = y_temp[idx2], y_temp[idx1]
        
        new_inversions = count_inversions_orig(y_temp)
        
        if abs(new_inversions - target_inversions) < abs(current_inversions - target_inversions):
            y = y_temp
            current_inversions = new_inversions
        
        iteration += 1
    
    if iteration >= max_iterations:
        print(f"Warning: Max iterations reached for target_inversions={target_inversions}. Actual inversions: {current_inversions}")
    
    return y

def count_inversions(arr): 
    def merge_and_count(left, right):
        merged = []
        count = 0
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                count += len(left) - i
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged, count

    def sort_and_count(arr_inner):
        if len(arr_inner) <= 1:
            return arr_inner, 0
        mid = len(arr_inner) // 2
        left, left_count = sort_and_count(arr_inner[:mid])
        right, right_count = sort_and_count(arr_inner[mid:])
        merged, merge_count = merge_and_count(left, right)
        return merged, left_count + right_count + merge_count

    _, inversions = sort_and_count(list(arr))
    return inversions


def get_num_blocks(fitted_y):
    if len(fitted_y) == 0: return 0
    return np.sum(np.diff(fitted_y) != 0) + 1

def evaluate_isotonic_fit(x, y, y_fit):
    mse = mean_squared_error(y, y_fit) if len(y) > 0 else 0
    num_blocks = get_num_blocks(y_fit)
    tau = kendall_tau_b_adjusted(y, y_fit) if len(y) > 1 else 1.0
    var_fitted = np.var(y_fit) if len(y_fit) > 1 else 0
    return {
        'mse': mse,
        'num_blocks': num_blocks,
        'kendall tau': tau,
        'variance': var_fitted
    }

def main():
    n = 50
    max_inversions_val = n * (n - 1) // 2
    num_levels = 50
    target_inversions_list = np.linspace(0, max_inversions_val, num_levels, dtype=int).tolist()
    
    results = {
        'n': [],
        'target_inversions': [],
        'input_inversions_orig': [],
        'input_inversions_adj': [], 
        'input_kendall_tau_adj': [],
        'mse': [],
        'num_blocks': [],
        'kendall tau': [],
        'variance': []
    }
    
    x = np.arange(n)
    
    for inv_target in target_inversions_list:
        print(f"Processing sequence with target inversions: {inv_target}")
        y = generate_sequence_with_target_inversions(n, inv_target)
        
        iso_reg = isotonic_regression_sklearn(increasing=True)
        try:
            y_fit = iso_reg.fit_transform(x, y)
        except Exception as e:
            print(f"Error in isotonic_regression for target_inversions={inv_target}: {e}")
            continue
        
        actual_inversions_orig = count_inversions(y)
        y_ideal_monotonic = x.copy()
        input_inv_adj = count_inversions_adjusted(y, y_ideal_monotonic)
        input_tau_adj = kendall_tau_b_adjusted(y, y_ideal_monotonic)

        metrics = evaluate_isotonic_fit(x, y, y_fit)
        
        results['n'].append(n)
        results['target_inversions'].append(inv_target)
        results['input_inversions_orig'].append(actual_inversions_orig)
        results['input_inversions_adj'].append(input_inv_adj)
        results['input_kendall_tau_adj'].append(input_tau_adj)
        results['mse'].append(metrics['mse'])
        results['num_blocks'].append(metrics['num_blocks'])
        results['kendall tau'].append(metrics['kendall tau'])
        results['variance'].append(metrics['variance'])
        
        print(f"  Target Inversions: {inv_target}")
        print(f"  Actual Original Inversions (for generation): {actual_inversions_orig}")
        print(f"  Adjusted Inversions (y vs x): {input_inv_adj}")
        print(f"  Adjusted Kendall Tau (y vs x): {input_tau_adj:.3f}")
        print(f"  MSE (y vs y_fit): {metrics['mse']:.3f}")
        print(f"  Number of Blocks (y_fit): {metrics['num_blocks']}")
        print(f"  Variance of Fit (y_fit): {metrics['variance']:.3f}")

if __name__ == "__main__":
    main()