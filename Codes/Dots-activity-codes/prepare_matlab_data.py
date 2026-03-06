import json
import numpy as np

def rankings_list_to_pairs(ranking_list: list, item_ids: list) -> list:
    pairs = []
    n_items = len(ranking_list)
    for i in range(n_items):
        for j in range(i + 1, n_items):
            rank_i = ranking_list[i]
            rank_j = ranking_list[j]
            id_i = item_ids[i]
            id_j = item_ids[j]
            
            if rank_i < rank_j:
                pairs.append([id_i, id_j])  # item i wins over item j
            elif rank_i > rank_j:
                pairs.append([id_j, id_i])  # item j wins over item i
            
    return pairs

def main():
    DATA_FILE = "all_data.json"
    
    print(f"From '{DATA_FILE}' loading data...")
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)

    all_gt_values = sorted(list(set(gt for entry in data for gt in entry["groundtruth"])))
    gt_to_id = {gt_val: i + 1 for i, gt_val in enumerate(all_gt_values)}
    
    doc_info_data = [[gt_to_id[gt], gt] for gt in all_gt_values]
    
    np.savetxt("doc_info.txt", doc_info_data, fmt='%d %.4f', delimiter=' ')
    print("Save 'doc_info.txt'.")
    
    all_pairs_M1 = [] # 'rankings'
    
    for annotator_id, entry in enumerate(data, 1):
        item_ids_1based = [gt_to_id[gt] for gt in entry["groundtruth"]]
        
        pairs_from_ranking = rankings_list_to_pairs(entry['rankings'], item_ids_1based)
        for winner, loser in pairs_from_ranking:
            all_pairs_M1.append([annotator_id, winner, loser])
            
    np.savetxt("all_pair_M1.txt", all_pairs_M1, fmt='%d', delimiter=' ')
    print("Save 'all_pair_M1.txt' (From rankings).")

    
    print("\nDone.")

if __name__ == '__main__':
    main()