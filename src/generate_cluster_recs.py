#!/usr/bin/env python3
"""
Generate cluster-based recommendations using implicit library.
Trains ALS model on all user data, then generates recommendations for each cluster
by averaging predictions across all users in that cluster.

Supports both single-cluster and multi-cluster assignments.
For multi-cluster, users are weighted by their cluster rank (primary cluster gets more weight).
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import json
import argparse
import numpy as np
from pathlib import Path
from implicit.als import AlternatingLeastSquares
from scipy.sparse import lil_matrix, csr_matrix
from collections import defaultdict
from implicit.nearest_neighbours import bm25_weight


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def parse_cluster_assignment(cluster_value, equal_weights=False):
    """
    Parse a cluster assignment value, handling both single and multi-cluster formats.
    
    Args:
        cluster_value: Either an int (single cluster) or a string like "39, 27" (multi-cluster)
        equal_weights: If True, all clusters get equal weight. If False, use 1/rank weighting.
    
    Returns:
        List of (cluster_id, weight) tuples.
    """
    if isinstance(cluster_value, int):
        # Single cluster assignment
        return [(cluster_value, 1.0)]
    elif isinstance(cluster_value, str):
        # Multi-cluster assignment: "39, 27, 15"
        cluster_ids = [int(c.strip()) for c in cluster_value.split(",")]
        n_clusters = len(cluster_ids)
        
        if equal_weights:
            # All clusters get equal weight
            weights = [1.0 / n_clusters] * n_clusters
        else:
            # Assign decreasing weights based on rank
            # Primary cluster gets highest weight, secondary gets less, etc.
            # Using 1/rank weighting: [1.0, 0.5, 0.33, 0.25, ...]
            weights = [1.0 / (i + 1) for i in range(n_clusters)]
            
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]
        
        return list(zip(cluster_ids, weights))
    else:
        raise ValueError(f"Unexpected cluster value type: {type(cluster_value)}")


def load_user_data(embeddings_path, clusters_path, equal_weights=False):
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
    is_multicluster = cluster_data['metadata'].get('multi_cluster_assignment', False)
    
    # Get unique clusters
    all_clusters = set()
    for cluster_value in user_to_cluster.values():
        assignments = parse_cluster_assignment(cluster_value, equal_weights)
        for cluster_id, _ in assignments:
            all_clusters.add(cluster_id)
    
    print(f"  Users: {n_users}")
    print(f"  Books: {n_books}")
    print(f"  Clusters: {len(all_clusters)}")
    print(f"  Multi-cluster assignment: {is_multicluster}")
    
    return user_vectors, user_to_cluster, n_users, n_books, is_multicluster


def group_users_by_cluster(user_to_cluster, is_multicluster, equal_weights=False):
    """
    Group users into clusters with weights.
    
    For multi-cluster assignments, a user can appear in multiple clusters
    with different weights based on their rank (or equal weights if specified).
    
    Returns:
        Dict mapping cluster_id -> list of (user_id, weight) tuples
    """
    print("\n" + "="*50)
    print("GROUPING USERS BY CLUSTER")
    print("="*50)
    
    if is_multicluster:
        if equal_weights:
            print("  Weighting mode: EQUAL (all clusters weighted equally)")
        else:
            print("  Weighting mode: RANK (1/rank weighting: primary=1, secondary=0.5, ...)")
    
    cluster_to_users = defaultdict(list)
    
    for user_id, cluster_value in user_to_cluster.items():
        assignments = parse_cluster_assignment(cluster_value, equal_weights)
        for cluster_id, weight in assignments:
            cluster_to_users[cluster_id].append((user_id, weight))
    
    # Print cluster info
    for cluster_id in sorted(cluster_to_users.keys()):
        user_weights = cluster_to_users[cluster_id]
        n_users = len(user_weights)
        total_weight = sum(w for _, w in user_weights)
        primary_users = sum(1 for _, w in user_weights if w == max(ww for _, ww in user_weights))
        
        if is_multicluster:
            print(f"  Cluster {cluster_id}: {n_users} users (total weight: {total_weight:.2f})")
        else:
            print(f"  Cluster {cluster_id}: {n_users} users")
    
    return cluster_to_users


def build_interaction_matrix(user_vectors, n_books, use_bm25=False):
    """
    Build interaction matrix for ALS training.
    Only includes real users (no cluster virtual users).

    Optionally applies BM25 weighting to reduce impact of power users and popular items.
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

    # Apply BM25 weighting if requested
    if use_bm25:
        print(f"  Applying BM25 weighting (K1={100}, B={0.8})...") # hyperparameters from implicit tutorial page
        print("    This reduces impact of power users and popular items")
        interactions = bm25_weight(interactions, K1=100, B=0.8)
    
    # Convert to CSR format for efficient operations
    interactions = interactions.tocsr()
    
    print(f"  Interaction matrix shape: {interactions.shape}")
    print(f"  Matrix sparsity: {(1 - interactions.nnz / (n_users * n_books)) * 100:.4f}%")
    
    return interactions, user_id_to_idx


def train_als_model(interactions, model_params):
    """Train implicit ALS model on interaction matrix."""
    print("\n" + "="*50)
    print("TRAINING ALS MODEL")
    print("="*50)
    
    print(f"  Factors: {model_params['factors']}")
    print(f"  Regularization: {model_params['regularization']}")
    print(f"  Iterations: {model_params['iterations']}")
    print(f"  Alpha: {model_params['alpha']}")
    
    model = AlternatingLeastSquares(
        factors=model_params['factors'],
        regularization=model_params['regularization'],
        iterations=model_params['iterations'],
        alpha=model_params['alpha'],
        random_state=model_params['random_state']
    )
    
    print("\nTraining...")
    # implicit library expects user-item matrix (users × items)
    # Scale interactions by alpha for confidence weighting
    model.fit(interactions * model_params['alpha'])
    
    print("\nTraining complete!")
    return model


def generate_cluster_recommendations(model, cluster_to_users, user_id_to_idx, 
                                     interactions, n_books, top_k, use_weights=True):
    """
    Generate top-k recommendations for each cluster by averaging predictions
    across all users in that cluster.
    
    For multi-cluster assignments, user contributions are weighted by their
    cluster membership weight (primary cluster gets more influence).
    
    Args:
        model: Trained ALS model
        cluster_to_users: Dict mapping cluster_id -> list of (user_id, weight) tuples
        user_id_to_idx: Dict mapping user_id -> matrix index
        interactions: User-item interaction matrix
        n_books: Number of books
        top_k: Number of recommendations per cluster
        use_weights: Whether to use weights for averaging (True for multi-cluster)
    """
    print("\n" + "="*50)
    print("GENERATING CLUSTER RECOMMENDATIONS")
    print("="*50)
    
    cluster_recommendations = {}
    all_books = np.arange(n_books)
    
    for cluster_id, user_weights in cluster_to_users.items():
        print(f"  Processing Cluster {cluster_id} ({len(user_weights)} users)...")
        
        all_scores = []
        all_weights = []
        valid_users = 0
        
        # Collect predictions from all users in cluster
        for user_id, weight in user_weights:
            if user_id in user_id_to_idx:
                user_idx = user_id_to_idx[user_id]
                
                # Predict scores for all books for this user
                # implicit's recommend method returns (items, scores)
                # We'll use the underlying scoring instead
                scores = model.user_factors[user_idx] @ model.item_factors.T
                all_scores.append(scores)
                all_weights.append(weight)
                valid_users += 1
        
        if valid_users == 0:
            print(f"    Warning: No valid users found for Cluster {cluster_id}")
            continue
        
        # Weighted average predictions across all users in cluster
        all_scores = np.array(all_scores)
        all_weights = np.array(all_weights)
        
        if use_weights:
            # Normalize weights
            all_weights = all_weights / all_weights.sum()
            avg_scores = np.average(all_scores, axis=0, weights=all_weights)
        else:
            avg_scores = np.mean(all_scores, axis=0)

        # Sort by score descending
        sorted_indices = np.argsort(-avg_scores)
        
        # Get top-k books
        top_books = [int(book_idx) for book_idx in sorted_indices[:top_k]]
        
        cluster_recommendations[int(cluster_id)] = top_books
        
        # Log statistics
        top_score = avg_scores[sorted_indices[0]]
        median_score = np.median(avg_scores)
        print(f"    Top score: {top_score:.4f}, Median score: {median_score:.4f}, "
              f"Valid users: {valid_users}")
    
    return cluster_recommendations


def save_recommendations(cluster_recommendations, cluster_to_users, model_params, 
                        top_k, output_path, is_multicluster, equal_weights=False):
    """Save cluster recommendations to JSON file."""
    print("\n" + "="*50)
    print("SAVING RECOMMENDATIONS")
    print("="*50)
    
    # Compute cluster sizes (counting unique users, not weighted)
    cluster_sizes = {
        int(cid): len(users) for cid, users in cluster_to_users.items()
    }
    
    # Determine method description
    if is_multicluster:
        if equal_weights:
            method = "equal_weighted_averaged_user_predictions"
        else:
            method = "rank_weighted_averaged_user_predictions"
    else:
        method = "averaged_user_predictions"
    
    output_data = {
        "cluster_recommendations": cluster_recommendations,
        "metadata": {
            "n_clusters": len(cluster_recommendations),
            "top_k": top_k,
            "cluster_sizes": cluster_sizes,
            "model_params": model_params,
            "method": method,
            "multi_cluster_assignment": is_multicluster,
            "equal_weights": equal_weights if is_multicluster else None
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved to: {output_path}")
    print(f"  Clusters: {len(cluster_recommendations)}")
    print(f"  Recommendations per cluster: {top_k}")


def print_summary(cluster_recommendations, cluster_to_users, is_multicluster):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print(f"  Total clusters: {len(cluster_recommendations)}")
    print(f"  Recommendations per cluster: {len(next(iter(cluster_recommendations.values())))}")
    print(f"  Multi-cluster mode: {is_multicluster}")
    
    # Cluster size statistics (counting users, not weights)
    cluster_sizes = [len(users) for users in cluster_to_users.values()]
    print(f"  Avg users per cluster: {np.mean(cluster_sizes):.2f}")
    print(f"  Min users per cluster: {np.min(cluster_sizes)}")
    print(f"  Max users per cluster: {np.max(cluster_sizes)}")
    
    if is_multicluster:
        # Weight statistics
        total_weights = [sum(w for _, w in users) for users in cluster_to_users.values()]
        print(f"  Avg total weight per cluster: {np.mean(total_weights):.2f}")
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(
        description='Generate cluster-based recommendations using implicit ALS'
    )
    parser.add_argument(
        '--clusters',
        type=str,
        default='user_clusters.json',
        help='Path to user clusters JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='recs_per_cluster.json',
        help='Path to output recommendations JSON file'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=100,
        help='Number of recommendations per cluster'
    )
    parser.add_argument(
        '--factors',
        type=int,
        default=100,
        help='Number of latent factors for ALS'
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=15,
        help='Number of ALS iterations'
    )
    parser.add_argument(
        '--regularization',
        type=float,
        default=0.01,
        help='Regularization parameter for ALS'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=20.0,
        help='Confidence scaling parameter'
    )
    parser.add_argument(
        '--use-bm25',
        action='store_true',
        help='Apply BM25 weighting to reduce impact of power users and popular items'
    )
    parser.add_argument(
        '--no-weights',
        action='store_true',
        help='Disable weighted averaging for multi-cluster (use simple average instead)'
    )
    parser.add_argument(
        '--equal-weights',
        action='store_true',
        help='Use equal weights for all cluster assignments (instead of 1/rank weighting)'
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
        'factors': args.factors,
        'regularization': args.regularization,
        'iterations': args.iterations,
        'alpha': args.alpha,
        'random_state': 42
    }
    
    # Load data
    user_vectors, user_to_cluster, n_users, n_books, is_multicluster = load_user_data(
        embeddings_path, clusters_path, equal_weights=args.equal_weights
    )
    
    # Group users by cluster (with weights for multi-cluster)
    cluster_to_users = group_users_by_cluster(user_to_cluster, is_multicluster, equal_weights=args.equal_weights)
    
    # Build interaction matrix (real users only)
    interactions, user_id_to_idx = build_interaction_matrix(
        user_vectors, n_books, args.use_bm25
    )
    
    # Train ALS model
    model = train_als_model(interactions, model_params)
    
    # Generate recommendations by averaging predictions across cluster users
    use_weights = is_multicluster and not args.no_weights
    cluster_recommendations = generate_cluster_recommendations(
        model, cluster_to_users, user_id_to_idx, interactions, n_books, args.top_k,
        use_weights=use_weights
    )
    
    # Save recommendations
    save_recommendations(
        cluster_recommendations, cluster_to_users, model_params,
        args.top_k, output_path, is_multicluster, equal_weights=args.equal_weights
    )
    
    # Print summary
    print_summary(cluster_recommendations, cluster_to_users, is_multicluster)


if __name__ == '__main__':
    main()
