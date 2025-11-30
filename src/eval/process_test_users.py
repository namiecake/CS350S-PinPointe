# make a train test split of user profiles

#!/usr/bin/env python3
"""
Split user ratings into train/test sets for evaluation.

This script takes a user embeddings file and splits each user's ratings
into 80% training and 20% test sets. Users must have at least 20 ratings
to be included in the split. After splitting, only users with at least 5
relevant items (4-5 stars) in the test set are kept.

Usage:
    python train_test_split.py --input active_users_embeddings_test.json \
                                --output-train train_users.json \
                                --output-test test_users.json
"""

import json
import random
import argparse
from pathlib import Path
from collections import defaultdict


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def count_relevant_items(ratings_dict):
    """
    Count the number of relevant items (ratings >= 4) in a ratings dictionary.
    
    Args:
        ratings_dict: Dictionary mapping book_id -> rating
    
    Returns:
        Number of items with rating >= 4
    """
    return sum(1 for rating in ratings_dict.values() if rating >= 4)


def split_user_ratings(user_ratings, train_ratio=0.8, min_test_items=4):
    """
    Split a user's ratings into train and test sets.
    
    Args:
        user_ratings: Dictionary mapping book_id -> rating
        train_ratio: Fraction of ratings to use for training (default: 0.8)
        min_test_items: Minimum number of items required in test set
    
    Returns:
        Tuple of (train_dict, test_dict) or (None, None) if split is invalid
    """
    book_ids = list(user_ratings.keys())
    n_ratings = len(book_ids)
    
    # Calculate split sizes
    n_train = int(n_ratings * train_ratio)
    n_test = n_ratings - n_train
    
    # Ensure minimum test set size
    if n_test < min_test_items:
        return None, None
    
    # Randomly shuffle and split
    random.shuffle(book_ids)
    train_books = book_ids[:n_train]
    test_books = book_ids[n_train:]
    
    # Create train and test dictionaries
    train_dict = {book_id: user_ratings[book_id] for book_id in train_books}
    test_dict = {book_id: user_ratings[book_id] for book_id in test_books}
    
    return train_dict, test_dict


def split_all_users(embeddings_data, train_ratio=0.8, min_test_items=4, 
                    min_relevant_test_items=5, random_seed=42):
    """
    Split all users' ratings into train/test sets, with post-filtering.
    
    Args:
        embeddings_data: Loaded embeddings JSON data
        train_ratio: Fraction of ratings for training
        min_test_items: Minimum items required in test set
        min_relevant_test_items: Minimum relevant items (4-5 stars) in test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data, statistics)
    """
    random.seed(random_seed)
    
    user_vectors = embeddings_data['user_vectors']
    train_users = {}
    test_users = {}
    
    skipped_min_items = 0
    skipped_relevant_items = 0
    rating_stats = {
        'train_ratings': [],
        'test_ratings': [],
        'test_relevant': []
    }
    
    print(f"\nSplitting {len(user_vectors)} users...")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Min test items: {min_test_items}")
    print(f"  Min relevant test items (4-5 stars): {min_relevant_test_items}")
    
    for user_id, ratings in user_vectors.items():
        train_dict, test_dict = split_user_ratings(ratings, train_ratio, min_test_items)
        
        # Skip if not enough test items
        if train_dict is None or test_dict is None:
            skipped_min_items += 1
            continue
        
        # # Post-filter: Check if user has enough relevant items in test set
        n_relevant_test = count_relevant_items(test_dict)
        if n_relevant_test < min_relevant_test_items:
            skipped_relevant_items += 1
            continue
        
        train_users[user_id] = train_dict
        test_users[user_id] = test_dict
        
        rating_stats['train_ratings'].append(len(train_dict))
        rating_stats['test_ratings'].append(len(test_dict))
        rating_stats['test_relevant'].append(n_relevant_test)
    
    # Calculate statistics
    n_users = len(train_users)
    avg_train = sum(rating_stats['train_ratings']) / n_users if n_users > 0 else 0
    avg_test = sum(rating_stats['test_ratings']) / n_users if n_users > 0 else 0
    avg_relevant = sum(rating_stats['test_relevant']) / n_users if n_users > 0 else 0
    
    stats = {
        'n_users': n_users,
        'skipped_min_items': skipped_min_items,
        'skipped_relevant_items': skipped_relevant_items,
        'total_skipped': skipped_min_items + skipped_relevant_items,
        'avg_train_ratings': round(avg_train, 2),
        'avg_test_ratings': round(avg_test, 2),
        'avg_relevant_test_ratings': round(avg_relevant, 2),
        'min_train_ratings': min(rating_stats['train_ratings']) if rating_stats['train_ratings'] else 0,
        'max_train_ratings': max(rating_stats['train_ratings']) if rating_stats['train_ratings'] else 0,
        'min_test_ratings': min(rating_stats['test_ratings']) if rating_stats['test_ratings'] else 0,
        'max_test_ratings': max(rating_stats['test_ratings']) if rating_stats['test_ratings'] else 0,
        'min_relevant_test': min(rating_stats['test_relevant']) if rating_stats['test_relevant'] else 0,
        'max_relevant_test': max(rating_stats['test_relevant']) if rating_stats['test_relevant'] else 0
    }
    
    print(f"\nSplit Statistics:")
    print(f"  Users included: {n_users}")
    print(f"  Users skipped (too few test items): {skipped_min_items}")
    print(f"  Users skipped (too few relevant test items): {skipped_relevant_items}")
    print(f"  Total skipped: {stats['total_skipped']}")
    print(f"  Avg train ratings per user: {stats['avg_train_ratings']}")
    print(f"  Avg test ratings per user: {stats['avg_test_ratings']}")
    print(f"  Avg relevant test ratings per user: {stats['avg_relevant_test_ratings']}")
    
    return train_users, test_users, stats


def save_split_data(users_dict, output_path, metadata, split_type):
    """
    Save train or test split to JSON file.
    
    Args:
        users_dict: Dictionary of user_id -> ratings
        output_path: Path to save the file
        metadata: Original metadata plus split statistics
        split_type: 'train' or 'test'
    """
    print(f"\nSaving {split_type} data to {output_path}...")
    
    # Update metadata
    output_metadata = metadata.copy()
    output_metadata['split_type'] = split_type
    output_metadata['n_users'] = len(users_dict)
    
    output_data = {
        'metadata': output_metadata,
        'user_vectors': users_dict
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved {len(users_dict)} users")


def main():
    parser = argparse.ArgumentParser(
        description='Split user ratings into train/test sets'
    )
    
    parser.add_argument('--input', type=str, default='active_users_embeddings_test.json',
                        help='Input user embeddings JSON file')
    parser.add_argument('--output-train', type=str, default='train_user_profiles.json',
                        help='Output path for training data (default: train_user_profiles.json)')
    parser.add_argument('--output-test', type=str, default='test_user_profiles.json',
                        help='Output path for test data (default: test_user_profiles.json)')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Fraction of ratings for training (default: 0.8)')
    parser.add_argument('--min-test-items', type=int, default=4,
                        help='Minimum items required in test set (default: 4)')
    parser.add_argument('--min-relevant-test-items', type=int, default=5,
                        help='Minimum relevant items (4-5 stars) in test set (default: 5)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir.parent / 'data'

    # Resolve paths
    input_path = data_dir / 'clients' / Path(args.input)
    output_train_path = data_dir / 'eval' / Path(args.output_train)
    output_test_path = data_dir / 'eval' / Path(args.output_test)
    
    print("="*60)
    print("TRAIN/TEST SPLIT")
    print("="*60)
    
    # Load data
    embeddings_data = load_json_file(input_path)
    original_metadata = embeddings_data.get('metadata', {})
    
    # Perform split
    train_users, test_users, stats = split_all_users(
        embeddings_data,
        train_ratio=args.train_ratio,
        min_test_items=args.min_test_items,
        min_relevant_test_items=args.min_relevant_test_items,
        random_seed=args.random_seed
    )
    
    # Prepare metadata
    split_metadata = original_metadata.copy()
    split_metadata.update(stats)
    split_metadata['train_ratio'] = args.train_ratio
    split_metadata['min_test_items'] = args.min_test_items
    split_metadata['min_relevant_test_items'] = args.min_relevant_test_items
    split_metadata['random_seed'] = args.random_seed
    
    # Save both splits
    save_split_data(train_users, output_train_path, split_metadata, 'train')
    save_split_data(test_users, output_test_path, split_metadata, 'test')
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Train data: {output_train_path}")
    print(f"Test data: {output_test_path}")


if __name__ == '__main__':
    main()