#!/usr/bin/env python3
"""
Generate cluster-based recommendations using LightFM.
Trains LightFM on all user data, then generates recommendations for each cluster
by treating cluster aggregate profiles as virtual users.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from lightfm import LightFM
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_user_data(embeddings_path, clusters_path):
    """Load user embeddings and cluster assignments."""
    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    
    embeddings_data = load_json_file(embeddings_path)
    cluster_data = load_json_file(clusters_path)
    
    user_vectors = embeddings_data['user_vectors']
    n_users = embeddings_data['metadata']['n_users']
    n_books = embeddings_data['metadata']['n_books']
    
    user_to_cluster = cluster_data['user_to_cluster']
    
    print(f"  Users: {n_users}")
    print(f"  Books: {n_books}")
    print(f"  Clusters: {len(set(user_to_cluster.values()))}")
    
    return user_vectors, user_to_cluster, n_users, n_books

def group_users_by_cluster(user_to_cluster):
    """Group users into clusters."""
    print("\n" + "="*50)
    print("GROUPING USERS BY CLUSTER")
    print("="*50)
    
    cluster_to_users = defaultdict(list)
    for user_id, cluster_id in user_to_cluster.items():
        cluster_to_users[cluster_id].append(user_id)
    
    for cluster_id, user_ids in sorted(cluster_to_users.items()):
        print(f"  Cluster {cluster_id}: {len(user_ids)} users")
    
    return cluster_to_users


def build_interaction_matrix(user_vectors, n_books):
    """
    Build interaction matrix for LightFM training.
    Only includes real users (no cluster virtual users).
    """
    print("\n" + "="*50)
    print("BUILDING INTERACTION MATRIX")
    print("="*50)
    
    # Create user ID mappings
    all_user_ids = list(user_vectors.keys())
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
    
    n_users = len(all_user_ids)
    
    print(f"  Users: {n_users}")
    print(f"  Books: {n_books}")
    
    # Build sparse interaction matrix
    print("  Building sparse matrix...")
    interactions = lil_matrix((n_users, n_books), dtype=np.float32)
    
    # Add user interactions
    for user_id, ratings in user_vectors.items():
        user_idx = user_id_to_idx[user_id]
        for book_id, rating in ratings.items():
            book_idx = int(book_id)
            interactions[user_idx, book_idx] = rating
    
    # Convert to CSR format for efficient operations
    interactions = interactions.tocsr()
    
    print(f"  Interaction matrix shape: {interactions.shape}")
    print(f"  Matrix sparsity: {(1 - interactions.nnz / (n_users * n_books)) * 100:.4f}%")
    
    return interactions, user_id_to_idx


def train_lightfm_model(interactions, model_params):
    """Train LightFM model on interaction matrix."""
    print("\n" + "="*50)
    print("TRAINING LIGHTFM MODEL")
    print("="*50)
    
    print(f"  Loss: {model_params['loss']}")
    print(f"  Components: {model_params['no_components']}")
    print(f"  Learning rate: {model_params['learning_rate']}")
    print(f"  Epochs: {model_params['epochs']}")
    print(f"  Threads: {model_params['num_threads']}")
    
    model = LightFM(
        loss=model_params['loss'],
        no_components=model_params['no_components'],
        learning_rate=model_params['learning_rate'],
        random_state=model_params['random_state']
    )
    
    print("\nTraining...")
    model.fit(
        interactions,
        epochs=model_params['epochs'],
        num_threads=model_params['num_threads'],
        verbose=True
    )
    
    print("\nTraining complete!")
    return model


def generate_cluster_recommendations(model, cluster_to_users, user_id_to_idx, 
                                     n_books, top_k):
    """
    Generate top-k recommendations for each cluster by averaging predictions
    across all users in that cluster.
    """
    print("\n" + "="*50)
    print("GENERATING CLUSTER RECOMMENDATIONS")
    print("="*50)
    
    cluster_recommendations = {}
    all_books = np.arange(n_books)
    
    for cluster_id, user_ids in cluster_to_users.items():
        print(f"  Processing Cluster {cluster_id} ({len(user_ids)} users)...")
        
        all_scores = []
        valid_users = 0
        
        # Collect predictions from all users in cluster
        for user_id in user_ids:
            if user_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                
                # Predict scores for all books for this user
                scores = model.predict(user_idx, all_books)
                all_scores.append(scores)
                valid_users += 1
        
        if valid_users == 0:
            print(f"    Warning: No valid users found for Cluster {cluster_id}")
            continue
        
        # Average predictions across all users in cluster
        avg_scores = np.mean(all_scores, axis=0)

        # Sort by score descending
        sorted_indices = np.argsort(-avg_scores)
        
        # Get top-k books
        top_books = [int(book_idx) for book_idx in sorted_indices[:top_k]]
        
        cluster_recommendations[int(cluster_id)] = top_books
        
        # Log statistics
        top_score = avg_scores[sorted_indices[0]]
        median_score = np.median(avg_scores)
        print(f"Top score: {top_score:.4f}, Median score: {median_score:.4f}, "
              f"Valid users: {valid_users}")
    
    return cluster_recommendations


def save_recommendations(cluster_recommendations, cluster_to_users, model_params, 
                        top_k, output_path):
    """Save cluster recommendations to JSON file."""
    print("\n" + "="*50)
    print("SAVING RECOMMENDATIONS")
    print("="*50)
    
    output_data = {
        "cluster_recommendations": cluster_recommendations,
        "metadata": {
            "n_clusters": len(cluster_recommendations),
            "top_k": top_k,
            "cluster_sizes": {
                int(cid): len(users) for cid, users in cluster_to_users.items()
            },
            "model_params": model_params,
            "method": "averaged_user_predictions"
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    print(f"  Clusters: {len(cluster_recommendations)}")
    print(f"  Recommendations per cluster: {top_k}")


def print_summary(cluster_recommendations, cluster_to_users):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"  Total clusters: {len(cluster_recommendations)}")
    print(f"  Recommendations per cluster: {len(next(iter(cluster_recommendations.values())))}")
    
    # Cluster size statistics
    cluster_sizes = [len(users) for users in cluster_to_users.values()]
    print(f"  Avg users per cluster: {np.mean(cluster_sizes):.2f}")
    print(f"  Min users per cluster: {np.min(cluster_sizes)}")
    print(f"  Max users per cluster: {np.max(cluster_sizes)}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate cluster-based recommendations using LightFM'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        default='user_clusters_cosine.json',
        help='Path to user clusters JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='cluster_recs.json',
        help='Path to output recommendations JSON file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of recommendations per cluster'
    )
    parser.add_argument(
        '--components',
        type=int,
        default=30,
        help='Number of latent components for LightFM'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.05,
        help='Learning rate for LightFM'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of threads for training'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data/server'

    embeddings_path = data_dir / 'user_embeddings_train.json'
    clusters_path = data_dir / args.clusters
    output_path = data_dir / args.output
    
    # Model parameters
    model_params = {
        'loss': 'warp',
        'no_components': args.components,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'num_threads': args.threads,
        'random_state': 42
    }
    
    # Load data
    user_vectors, user_to_cluster, n_users, n_books = load_user_data(
        embeddings_path, clusters_path
    )
    
    # Group users by cluster
    cluster_to_users = group_users_by_cluster(user_to_cluster)
    
    # Build interaction matrix (real users only)
    interactions, user_id_to_idx = build_interaction_matrix(
        user_vectors, n_books
    )
    
    # Train LightFM model
    model = train_lightfm_model(interactions, model_params)
    
    # Generate recommendations by averaging predictions across cluster users
    cluster_recommendations = generate_cluster_recommendations(
        model, cluster_to_users, user_id_to_idx, n_books, args.top_k
    )
    
    # Save recommendations
    save_recommendations(
        cluster_recommendations, cluster_to_users, model_params,
        args.top_k, output_path
    )
    
    # Print summary
    print_summary(cluster_recommendations, cluster_to_users)


if __name__ == '__main__':
    main()