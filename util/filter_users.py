#!/usr/bin/env python3
"""
Filter the Goodreads dataset to only include users with at least
a minimum number of total ratings AND a minimum number of highly-rated 
items (4-5 stars).

python filter_users.py path-to-input-json path-to-output-json --min-total 20 --min-relevant 10
"""

import json
import sys
import argparse


def filter_users(input_path, output_path, min_total_ratings, min_relevant_ratings):
    print(f"Loading data from {input_path}...")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    num_books = data.get('metadata')['n_books']
    user_vectors = data.get('user_vectors', {})
    original_count = len(user_vectors)
    
    print(f"Original number of users: {original_count}")
    print(f"Filtering to users with >= {min_total_ratings} total ratings")
    print(f"  AND >= {min_relevant_ratings} ratings of 4-5 stars...")
    
    # Filter users based on both criteria
    filtered_vectors = {}
    for user_id, ratings in user_vectors.items():
        total_count = len(ratings)
        # Count ratings that are 4 or 5 stars
        relevant_count = sum(1 for rating in ratings.values() if rating >= 4)
        
        if total_count >= min_total_ratings and relevant_count >= min_relevant_ratings:
            filtered_vectors[user_id] = ratings
    
    filtered_count = len(filtered_vectors)
    
    # Get the set of all books that appear in filtered users' ratings
    all_books = set()
    for ratings in filtered_vectors.values():
        all_books.update(ratings.keys())
    
    # Calculate statistics about relevant ratings
    total_relevant = sum(
        sum(1 for rating in ratings.values() if rating >= 4)
        for ratings in filtered_vectors.values()
    )
    total_ratings = sum(len(ratings) for ratings in filtered_vectors.values())
    
    # Build output data
    output_data = {
        "metadata": {
            "n_users": filtered_count,
            "n_books": num_books, # caeley note: changed this back to be the original number of books bc otherwise get OOB errors for book indices
            "vector_format": data.get('metadata', {}).get('vector_format', 'sparse_dict'),
            "min_total_ratings_filter": min_total_ratings,
            "min_relevant_ratings_filter": min_relevant_ratings,
            "original_n_users": original_count,
            "total_ratings": total_ratings,
            "total_relevant_ratings": total_relevant
        },
        "user_vectors": filtered_vectors
    }
    
    print(f"\nFiltered number of users: {filtered_count}")
    print(f"Users removed: {original_count - filtered_count}")
    print(f"Unique books in filtered data: {len(all_books)}")
    print(f"Overall number of books (value saved in metadata): {num_books}")
    print(f"Total ratings in filtered data: {total_ratings}")
    print(f"Total relevant ratings (4-5 stars): {total_relevant}")

    if total_ratings > 0:
        print(f"Percentage relevant: {100 * total_relevant / total_ratings:.1f}%")
    else:
        print("Percentage relevant: N/A (no ratings)")
    
    print(f"\nSaving filtered data to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter users by minimum total ratings AND minimum relevant ratings (4-5 stars)"
    )
    parser.add_argument("input", help="Path to input JSON file")
    parser.add_argument("output", help="Path to output JSON file")
    parser.add_argument(
        "--min-total",
        type=int,
        default=20,
        help="Minimum total number of ratings required (default: 20)"
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=10,
        help="Minimum number of 4-5 star ratings required (default: 10)"
    )
    
    args = parser.parse_args()
    filter_users(args.input, args.output, args.min_total, args.min_relevant)