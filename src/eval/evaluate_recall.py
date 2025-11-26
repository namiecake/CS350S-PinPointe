#!/usr/bin/env python3
"""
Evaluate recommendation algorithms using recall@k metrics.

This script compares multiple recommendation algorithms by calculating
recall@10, recall@20, and recall@50. It uses test user ratings to 
determine relevant items (4-5 star ratings).

Usage:
    python evaluate_recall.py --test-file test_users.json \
                              --rec-files mf_recs.json popularity_recs.json pinpointe_recs.json \
                              --output results.txt

so buggy oml i am deleting
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
import sys


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def get_relevant_items(user_ratings, relevance_threshold=4):
    """
    Get relevant items for a user based on rating threshold.
    
    Args:
        user_ratings: Dictionary mapping book_id -> rating
        relevance_threshold: Minimum rating to be considered relevant (default: 4)
    
    Returns:
        Set of relevant book IDs (as strings)
    """
    relevant = set()
    for book_id, rating in user_ratings.items():
        if rating >= relevance_threshold:
            relevant.add(book_id)
    return relevant


def calculate_recall_at_k(recommendations, relevant_items, k):
    """
    Calculate recall@k for a single user.
    
    Recall@k = |recommended ∩ relevant|_k / |relevant|
    
    Args:
        recommendations: List of recommended book IDs (ordered)
        relevant_items: Set of relevant book IDs
        k: Number of top recommendations to consider
    
    Returns:
        Recall score (float between 0 and 1)
    """
    if len(relevant_items) == 0:
        return 0.0
    
    # Get top-k recommendations
    top_k_recs = set(recommendations[:k])
    
    # Count hits
    hits = len(top_k_recs & relevant_items)
    
    # Calculate recall
    recall = hits / len(relevant_items)
    
    return recall


def evaluate_algorithm(rec_data, test_users, k_values=[10, 20, 50], relevance_threshold=4):
    """
    Evaluate a recommendation algorithm across all users.
    
    Args:
        rec_data: Dictionary with 'recommendations' mapping user_id -> list of book_ids
        test_users: Dictionary mapping user_id -> test ratings
        k_values: List of k values to evaluate
        relevance_threshold: Minimum rating for relevance
    
    Returns:
        Dictionary mapping k -> average recall@k
    """
    recommendations = rec_data.get('recommendations', {})
    
    recall_scores = {k: [] for k in k_values}
    users_evaluated = 0
    users_skipped = 0
    
    for user_id, test_ratings in test_users.items():
        # Get relevant items for this user
        relevant_items = get_relevant_items(test_ratings, relevance_threshold)
        
        # Skip users with no relevant items
        if len(relevant_items) == 0:
            print("skipped user: ", user_id)
            users_skipped += 1
            continue
        
        # Get recommendations for this user
        if user_id not in recommendations:
            users_skipped += 1
            continue
        
        user_recs = recommendations[user_id]
        users_evaluated += 1
        
        # Calculate recall@k for each k
        for k in k_values:
            recall = calculate_recall_at_k(user_recs, relevant_items, k)
            recall_scores[k].append(recall)
    
    # Calculate average recall for each k
    avg_recall = {}
    for k in k_values:
        if len(recall_scores[k]) > 0:
            avg_recall[k] = sum(recall_scores[k]) / len(recall_scores[k])
        else:
            avg_recall[k] = 0.0
    print(users_skipped)
    return avg_recall, users_evaluated, users_skipped


def print_results_table(results_dict, k_values):
    """
    Print evaluation results in a formatted table.
    
    Args:
        results_dict: Dictionary mapping algorithm_name -> (recall_dict, n_users, n_skipped)
        k_values: List of k values that were evaluated
    """
    print("\n" + "="*70)
    print("RECOMMENDATION ALGORITHM COMPARISON")
    print("="*70)
    
    # Print header
    header = f"{'Algorithm':<20}"
    for k in k_values:
        header += f"Recall@{k:<3} "
    header += f"{'Users':<8}"
    print(header)
    print("-"*70)
    
    # Sort algorithms by recall@10 (descending)
    sorted_algos = sorted(results_dict.items(), 
                         key=lambda x: x[1][0].get(k_values[0], 0), 
                         reverse=True)
    
    # Print each algorithm's results
    for algo_name, (recall_dict, n_users, n_skipped) in sorted_algos:
        row = f"{algo_name:<20}"
        for k in k_values:
            recall_val = recall_dict.get(k, 0.0)
            row += f"{recall_val:.4f}   "
        row += f"{n_users:<8}"
        print(row)
    
    print("-"*70)
    
    # Print summary statistics
    if results_dict:
        first_algo = next(iter(results_dict.values()))
        n_users = first_algo[1]
        n_skipped = first_algo[2]
        print(f"\nEvaluation Summary:")
        print(f"  Users evaluated: {n_users}")
        print(f"  Users skipped: {n_skipped}")
        print(f"  Relevance threshold: 4+ stars")


def save_results_to_file(results_dict, k_values, output_path):
    """Save results to a text file."""
    with open(output_path, 'w') as f:
        # Write table
        f.write("="*70 + "\n")
        f.write("RECOMMENDATION ALGORITHM COMPARISON\n")
        f.write("="*70 + "\n\n")
        
        # Header
        header = f"{'Algorithm':<20}"
        for k in k_values:
            header += f"Recall@{k:<3} "
        header += f"{'Users':<8}\n"
        f.write(header)
        f.write("-"*70 + "\n")
        
        # Sort and write results
        sorted_algos = sorted(results_dict.items(), 
                             key=lambda x: x[1][0].get(k_values[0], 0), 
                             reverse=True)
        
        for algo_name, (recall_dict, n_users, n_skipped) in sorted_algos:
            row = f"{algo_name:<20}"
            for k in k_values:
                recall_val = recall_dict.get(k, 0.0)
                row += f"{recall_val:.4f}   "
            row += f"{n_users:<8}\n"
            f.write(row)
        
        f.write("-"*70 + "\n")
        
        # Summary
        if results_dict:
            first_algo = next(iter(results_dict.values()))
            n_users = first_algo[1]
            n_skipped = first_algo[2]
            f.write(f"\nEvaluation Summary:\n")
            f.write(f"  Users evaluated: {n_users}\n")
            f.write(f"  Users skipped: {n_skipped}\n")
            f.write(f"  Relevance threshold: 4+ stars\n")
    
    print(f"\nResults saved to: {output_path}")


def extract_algorithm_name(rec_data, filepath):
    """
    Extract algorithm name from recommendation data or filename.
    
    Args:
        rec_data: Loaded recommendation JSON data
        filepath: Path to the recommendation file
    
    Returns:
        Algorithm name string
    """
    filename = Path(filepath).stem
    
    # Remove common suffixes
    for suffix in ['_recs', '_train_recs']:
        filename = filename.replace(suffix, '')
    
    return filename


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate recommendation algorithms using recall@k'
    )
    
    parser.add_argument('--test-file', type=str, 
                        default='test_users.json',
                        help='Test users JSON file (default: test_users.json)')
    parser.add_argument('--rec-files', type=str, nargs='+', required=True,
                        help='One or more recommendation JSON files to evaluate (paths relative to src/eval/)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[10, 20, 50],
                        help='K values for recall@k (default: 10 20 50)')
    parser.add_argument('--relevance-threshold', type=int, default=4,
                        help='Minimum rating for relevance (default: 4)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional output file for results (default: print to console only)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LOADING DATA")
    print("="*70)
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir.parent / 'data' 
    baselines_dir = data_dir / 'eval' / 'baselines'

    # Load test data
    test_filepath = data_dir / 'eval' / args.test_file
    test_data = load_json_file(test_filepath)
    test_users = test_data['user_vectors']
    print(f"  Test users: {len(test_users)}")

    # Load all recommendation files
    rec_datasets = []
    for rec_file in args.rec_files:
        rec_filepath = baselines_dir / rec_file
        try:
            rec_data = load_json_file(rec_filepath)
            algo_name = extract_algorithm_name(rec_data, rec_file)
            rec_datasets.append((algo_name, rec_data))
        except Exception as e:
            print(f"ERROR loading {rec_file}: {e}")
            continue
    
    if not rec_datasets:
        print("ERROR: No recommendation files could be loaded")
        sys.exit(1)
    
    print(f"  Loaded {len(rec_datasets)} recommendation datasets")
    
    # Evaluate each algorithm
    print("\n" + "="*70)
    print("EVALUATING ALGORITHMS")
    print("="*70)
    
    results = {}
    for algo_name, rec_data in rec_datasets:
        print(f"\nEvaluating {algo_name}...")
        recall_dict, n_users, n_skipped = evaluate_algorithm(
            rec_data, test_users, args.k_values, args.relevance_threshold
        )
        results[algo_name] = (recall_dict, n_users, n_skipped)
        
        # Print quick summary
        for k in args.k_values:
            print(f"  Recall@{k}: {recall_dict[k]:.4f}")
    
    # Print final results table
    print_results_table(results, args.k_values)
    
    # Save to file if requested
    if args.output:
        save_results_to_file(results, args.k_values, args.output)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()