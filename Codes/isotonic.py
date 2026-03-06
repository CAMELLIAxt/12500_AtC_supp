import numpy as np

def _pava_core(y_values, weights):
    n = len(y_values)
    if n == 0:
        return []

    pool = []
    for j in range(n):
        pool.append([y_values[j], weights[j], 1])

        while len(pool) >= 2:
            block2_mean, block2_weight, block2_count = pool[-1]
            block1_mean, block1_weight, block1_count = pool[-2]

            if block1_mean > block2_mean:  # Violation for increasing isotonic
                pool.pop() # remove block2
                pool.pop() # remove block1

                merged_weight = block1_weight + block2_weight
                
                if merged_weight <= 1e-12: # Effectively zero weight for the merged block
                    merged_mean = (block1_mean + block2_mean) / 2.0
                else:
                    merged_mean = (block1_mean * block1_weight + block2_mean * block2_weight) / merged_weight
                
                merged_count = block1_count + block2_count # Number of unique_x points in the new block
                pool.append([merged_mean, merged_weight, merged_count])
            else:
                # No violation, stop pooling for this point
                break
    return pool


def isotonic_regression(x, y, sample_weight=None, increasing=True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.ndim != 1:
        raise ValueError("x and y must be 1-dimensional.")
    
    n = len(x)
    if n == 0:
        return np.array([])

    if sample_weight is None:
        sample_weight = np.ones(n, dtype=float)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if sample_weight.shape != x.shape:
            raise ValueError("sample_weight must have the same shape as x and y.")
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight must be non-negative.")

    original_indices = np.arange(n)
    perm = np.lexsort((original_indices, x)) 
    
    x_sorted = x[perm]
    y_sorted = y[perm]
    sw_sorted = sample_weight[perm]

    y_transform_factor = 1.0
    if not increasing:
        y_transform_factor = -1.0
    
    y_transformed = y_sorted * y_transform_factor

    unique_x_s, V_j_inverse_indices, V_j_counts = np.unique(
        x_sorted, return_inverse=True, return_counts=True
    )
    
    num_unique_x = len(unique_x_s)
    y_pava_input = np.zeros(num_unique_x, dtype=float)
    w_pava_input = np.zeros(num_unique_x, dtype=float)

    # Sum of weights for each unique x (W_j)
    np.add.at(w_pava_input, V_j_inverse_indices, sw_sorted)
    
    # Sum of (y_transformed * sw_sorted) for each unique x
    sum_yw_pava_input = np.zeros(num_unique_x, dtype=float)
    np.add.at(sum_yw_pava_input, V_j_inverse_indices, y_transformed * sw_sorted)
    
    # Calculate y_pava_input (V_j = weighted average of y_transformed for each unique x)
    non_zero_weights_mask = w_pava_input > 1e-12 # Use a small epsilon for float comparison
    zero_weights_mask = ~non_zero_weights_mask

    y_pava_input[non_zero_weights_mask] = sum_yw_pava_input[non_zero_weights_mask] / w_pava_input[non_zero_weights_mask]

    if np.any(zero_weights_mask):
        sum_y_for_zero_weight_blocks = np.zeros(num_unique_x, dtype=float)
        np.add.at(sum_y_for_zero_weight_blocks, V_j_inverse_indices, y_transformed)
        
        safe_counts = np.maximum(V_j_counts, 1) 
        y_pava_input[zero_weights_mask] = sum_y_for_zero_weight_blocks[zero_weights_mask] / safe_counts[zero_weights_mask]

    pooled_blocks = _pava_core(y_pava_input, w_pava_input)

    solution_for_unique_x = np.zeros(num_unique_x, dtype=float)
    current_pos = 0
    for block_mean, _, block_num_unique_points in pooled_blocks:
        for _ in range(block_num_unique_points):
            if current_pos < num_unique_x: # Safety check for array bounds
                 solution_for_unique_x[current_pos] = block_mean
            current_pos += 1
    
    solution_for_unique_x *= y_transform_factor

    fitted_values_sorted = solution_for_unique_x[V_j_inverse_indices]

    inv_perm = np.argsort(perm)
    final_solution = fitted_values_sorted[inv_perm]
    

    return final_solution

if __name__ == '__main__':
    # Example 
    np.random.seed(0)
    x_ex1 = np.arange(1, 21)
    y_ex1_true = x_ex1 * 0.8
    y_ex1 = y_ex1_true + np.random.normal(0, 3, 20)
    
    fitted_ex1_increasing = isotonic_regression(x_ex1, y_ex1)
    fitted_ex1_decreasing = isotonic_regression(x_ex1, y_ex1, increasing=False)

    print("Toy Example (Increasing):")
    print("Fitted:", np.round(fitted_ex1_increasing, 2))