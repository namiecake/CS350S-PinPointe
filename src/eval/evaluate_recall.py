#!/usr/bin/env python3
"""
Compare multiple recommendation algorithms by calculating recall@k metrics.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_standard_recs(filepath: str) -> dict[str, list[int]]:
    """Load recommendations in standard format (mf, itemknn, userknn, poprec)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['recommendations']


def load_pinpointe_recs(filepath: str) -> dict[str, list[int]]:
    """Load recommendations in pinpointe format."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    recommendations = {}
    for user_id, user_data in data['user_recommendations'].items():
        recommendations[user_id] = user_data['recommendations']
    return recommendations


def load_test_profiles(filepath: str, min_rating: int = 4) -> dict[str, set[int]]:
    """Load test user profiles and extract relevant items (items user interacted with)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    relevant_items = {}
    for user_id, ratings in data['user_vectors'].items():
        # All items in the test set are considered relevant
        # Keys are item IDs (as strings), convert to int
        relevant_items[user_id] = set(
            int(item_id) for item_id, rating in ratings.items() if rating >= min_rating
        )
    return relevant_items


def calculate_recall_at_k(recommendations: list[int], relevant_items: set[int], k: int) -> float:
    """Calculate recall@k for a single user."""
    if not relevant_items:
        return 0.0
    
    top_k_recs = set(recommendations[:k])
    hits = len(top_k_recs & relevant_items)
    return hits / len(relevant_items)


def evaluate_algorithm(
    recommendations: dict[str, list[int]], 
    test_profiles: dict[str, set[int]],
    k_values: list[int]
) -> dict[int, float]:
    """Evaluate an algorithm's recommendations against test profiles."""
    recalls = {k: [] for k in k_values}
    
    # Only evaluate users that appear in both recommendations and test profiles
    common_users = set(recommendations.keys()) & set(test_profiles.keys())
    
    for user_id in common_users:
        user_recs = recommendations[user_id]
        relevant = test_profiles[user_id]
        
        for k in k_values:
            recall = calculate_recall_at_k(user_recs, relevant, k)
            recalls[k].append(recall)
    
    # Calculate mean recall for each k
    mean_recalls = {}
    for k in k_values:
        if recalls[k]:
            mean_recalls[k] = sum(recalls[k]) / len(recalls[k])
        else:
            mean_recalls[k] = 0.0
    
    return mean_recalls, len(common_users)


def print_results_table(results: dict[str, tuple[dict[int, float], int]], k_values: list[int]):
    """Print results in a formatted table."""
    # Header
    header = f"{'Algorithm':<15} | {'Users':>7}"
    for k in k_values:
        header += f" | {'Recall@' + str(k):>10}"
    
    separator = "-" * len(header)
    
    print("\n" + "=" * len(header))
    print("RECOMMENDATION ALGORITHM COMPARISON")
    print("=" * len(header))
    print(header)
    print(separator)
    
    # Sort by recall@10 descending
    sorted_results = sorted(results.items(), key=lambda x: x[1][0].get(k_values[0], 0), reverse=True)
    
    for algo_name, (recalls, n_users) in sorted_results:
        row = f"{algo_name:<15} | {n_users:>7}"
        for k in k_values:
            recall = recalls.get(k, 0.0)
            row += f" | {recall:>10.4f}"
        print(row)
    
    print(separator)
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare recommendation algorithms using recall@k metrics')
    # maybe fix this
    # parser.add_argument('--data-dir', type=str, default='.', 
                        # help='Directory containing the JSON files (default: current directory)')
    parser.add_argument('--test-file', type=str, default='test_user_profiles.json',
                        help='Test user profiles file (default: test_user_profiles.json)')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir.parent / 'data' / 'eval'
    
    # K values for recall calculation
    k_values = [10, 20, 50, 100]
    
    # Define algorithms and their file formats
    algorithms = {
        'mf': ('mf_train_recs.json', 'standard'),
        'itemknn': ('itemknn_train_recs.json', 'standard'),
        'userknn': ('userknn_train_recs.json', 'standard'),
        'poprec': ('poprec_train_recs.json', 'standard'),
        'pinpointe': ('pinpointe_train_recs.json', 'pinpointe'),
    }
    
    # Load test profiles
    test_file = data_dir / args.test_file
    print(f"Loading test profiles from {test_file}...")
    try:
        test_profiles = load_test_profiles(test_file)
        print(f"  Loaded {len(test_profiles)} test users")
    except FileNotFoundError:
        print(f"Error: Test file '{test_file}' not found!")
        return
    
    # Load and evaluate each algorithm
    results = {}
    
    for algo_name, (filename, format_type) in algorithms.items():
        filepath = data_dir / 'inputs' / filename
        
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping {algo_name}")
            continue
        
        print(f"Loading {algo_name} recommendations from {filename}...")
        
        try:
            if format_type == 'standard':
                recommendations = load_standard_recs(filepath)
            else:  # pinpointe format
                recommendations = load_pinpointe_recs(filepath)
            
            print(f"  Loaded recommendations for {len(recommendations)} users")
            
            # Evaluate
            recalls, n_users = evaluate_algorithm(recommendations, test_profiles, k_values)
            results[algo_name] = (recalls, n_users)
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
    
    if not results:
        print("No algorithms successfully loaded!")
        return
    
    # Print results table
    print_results_table(results, k_values)


if __name__ == '__main__':
    main()