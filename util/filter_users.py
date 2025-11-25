#!/usr/bin/env python3
"""
Filter the Goodreads dataset to only include users with at least
a minimum number of ratings.

python filter_users.py path-to-input-json path-to-output-json -m 20
"""

import json
import sys
import argparse


def filter_users(input_path, output_path, min_ratings):
    print(f"Loading data from {input_path}...")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    user_vectors = data.get('user_vectors', {})
    original_count = len(user_vectors)
    
    print(f"Original number of users: {original_count}")
    print(f"Filtering to users with >= {min_ratings} ratings...")
    
    # Filter users
    filtered_vectors = {
        user_id: ratings
        for user_id, ratings in user_vectors.items()
        if len(ratings) >= min_ratings
    }
    
    filtered_count = len(filtered_vectors)
    
    # Get the set of all books that appear in filtered users' ratings
    all_books = set()
    for ratings in filtered_vectors.values():
        all_books.update(ratings.keys())
    
    # Build output data
    output_data = {
        "metadata": {
            "n_users": filtered_count,
            "n_books": len(all_books),
            "vector_format": data.get('metadata', {}).get('vector_format', 'sparse_dict'),
            "min_ratings_filter": min_ratings,
            "original_n_users": original_count
        },
        "user_vectors": filtered_vectors
    }
    
    print(f"Filtered number of users: {filtered_count}")
    print(f"Users removed: {original_count - filtered_count}")
    print(f"Unique books in filtered data: {len(all_books)}")
    
    print(f"\nSaving filtered data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter users by minimum number of ratings"
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument(
        "-m", "--min-ratings",
        type=int,
        default=20,
        help="Minimum number of ratings required (default: 20)"
    )
    
    args = parser.parse_args()
    filter_users(args.input, args.output, args.min_ratings)