#!/usr/bin/env python3
"""
Replace user embeddings in user_embeddings.json with training versions
from train_user_profiles.json for users that appear in both files.

This creates a user_embeddings_for_baselines.json file suitable for
evaluating recommendation algorithms on a specific subset of users.
"""

import json
import argparse
from pathlib import Path


def replace_user_embeddings(
    full_embeddings_path: str,
    train_profiles_path: str,
    output_path: str
) -> dict:
    """
    Replace user embeddings for users that exist in both files.
    
    Args:
        full_embeddings_path: Path to user_embeddings.json
        train_profiles_path: Path to train_user_profiles.json
        output_path: Path for output file
        
    Returns:
        Dictionary with statistics about the replacement
    """
    # Load the full user embeddings
    print(f"Loading full embeddings from {full_embeddings_path}...")
    with open(full_embeddings_path, 'r') as f:
        full_data = json.load(f)
    
    # Load the training user profiles
    print(f"Loading training profiles from {train_profiles_path}...")
    with open(train_profiles_path, 'r') as f:
        train_data = json.load(f)
    
    # Get user vectors from both files
    full_user_vectors = full_data.get('user_vectors', {})
    train_user_vectors = train_data.get('user_vectors', {})
    
    # Track statistics
    stats = {
        'total_users_in_full': len(full_user_vectors),
        'total_users_in_train': len(train_user_vectors),
        'users_replaced': 0,
        'users_not_found_in_full': []
    }
    
    # Create output data structure (copy metadata if present)
    output_data = {}
    if 'metadata' in full_data:
        output_data['metadata'] = full_data['metadata'].copy()
    
    # Start with a copy of the full user vectors
    output_user_vectors = full_user_vectors.copy()
    
    # Replace embeddings for users that exist in both files
    for user_id, train_embedding in train_user_vectors.items():
        if user_id in output_user_vectors:
            output_user_vectors[user_id] = train_embedding
            stats['users_replaced'] += 1
        else:
            stats['users_not_found_in_full'].append(user_id)
    
    output_data['user_vectors'] = output_user_vectors
    
    # Write output file
    print(f"Writing output to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Replace user embeddings with training versions for baseline evaluation'
    )
    parser.add_argument(
        '--full-embeddings',
        default='user_embeddings.json',
        help='Path to full user embeddings file (default: user_embeddings.json)'
    )
    parser.add_argument(
        '--train-profiles',
        default='train_user_profiles.json',
        help='Path to training user profiles file (default: train_user_profiles.json)'
    )
    parser.add_argument(
        '--output',
        default='user_embeddings_for_baselines.json',
        help='Path for output file (default: user_embeddings_for_baselines.json)'
    )
    
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    full_embeddings_path = data_dir / args.full_embeddings
    train_profiles_path = data_dir / 'eval' / args.train_profiles
    output_path = data_dir / 'eval' / args.output

    
    # Run the replacement
    stats = replace_user_embeddings(
        full_embeddings_path,
        train_profiles_path,
        output_path
    )
    
    # Print summary
    print("\n" + "=" * 50)
    print("Replacement Summary")
    print("=" * 50)
    print(f"Total users in full embeddings:  {stats['total_users_in_full']:,}")
    print(f"Total users in training set:     {stats['total_users_in_train']:,}")
    print(f"Users replaced:                  {stats['users_replaced']:,}")
    
    if stats['users_not_found_in_full']:
        print(f"\nWarning: {len(stats['users_not_found_in_full'])} users in training set "
              f"were not found in full embeddings")
        if len(stats['users_not_found_in_full']) <= 10:
            print(f"  Missing users: {stats['users_not_found_in_full']}")
        else:
            print(f"  First 10 missing: {stats['users_not_found_in_full'][:10]}")
    
    print(f"\nOutput written to: {output_path}")


if __name__ == '__main__':
    main()