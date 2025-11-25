#!/usr/bin/env python3
"""
Analyze the distribution of ratings per user in the Goodreads dataset.
run it like: python user_rating_stats.py path/to/your/reviews.json
"""

import json
import sys
import numpy as np
from collections import Counter


def analyze_ratings(filepath):
    print(f"Loading data from {filepath}...")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    metadata = data.get('metadata', {})
    user_vectors = data.get('user_vectors', {})
    
    print(f"\n{'='*60}")
    print("DATASET METADATA")
    print(f"{'='*60}")
    print(f"Total users in file: {metadata.get('n_users', 'N/A')}")
    print(f"Total books in file: {metadata.get('n_books', 'N/A')}")
    print(f"Vector format: {metadata.get('vector_format', 'N/A')}")
    
    # Count ratings per user
    ratings_per_user = [len(ratings) for ratings in user_vectors.values()]
    ratings_per_user = np.array(ratings_per_user)
    
    print(f"\n{'='*60}")
    print("BASIC STATISTICS")
    print(f"{'='*60}")
    print(f"Number of users: {len(ratings_per_user)}")
    print(f"Total ratings: {sum(ratings_per_user)}")
    print(f"Min ratings per user: {np.min(ratings_per_user)}")
    print(f"Max ratings per user: {np.max(ratings_per_user)}")
    print(f"Mean ratings per user: {np.mean(ratings_per_user):.2f}")
    print(f"Median ratings per user: {np.median(ratings_per_user):.2f}")
    print(f"Std dev: {np.std(ratings_per_user):.2f}")
    
    print(f"\n{'='*60}")
    print("PERCENTILES")
    print(f"{'='*60}")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        value = np.percentile(ratings_per_user, p)
        print(f"{p}th percentile: {value:.0f} ratings")
    
    print(f"\n{'='*60}")
    print("USERS ABOVE THRESHOLD")
    print(f"{'='*60}")
    thresholds = [0, 5, 10, 15, 20, 25, 30, 50, 100, 500, 1000, 2000]
    for thresh in thresholds:
        count = np.sum(ratings_per_user >= thresh)
        pct = 100 * count / len(ratings_per_user)
        print(f">= {thresh:3d} ratings: {count:6d} users ({pct:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_json_file>")
        sys.exit(1)
    
    analyze_ratings(sys.argv[1])