#!/usr/bin/env python3
"""
Cluster user profile embeddings using K-Means with cosine similarity.
Uses sqrt(n_users) clusters and can optionally assign users to multiple clusters.
This version uses standard scikit-learn without spherecluster dependency.
"""

import json
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from pathlib import Path
import argparse
from collections import defaultdict


def load_embeddings(filepath):
    """Load user embeddings from JSON file."""
    print(f"Loading embeddings from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    n_users = data['metadata']['n_users']
    n_books = data['metadata']['n_books']
    user_vectors = data['user_vectors']
    
    print(f"  Loaded {n_users} users with {n_books}-dimensional vectors")
    return user_vectors, n_users, n_books


def convert_to_sparse_matrix(user_vectors, n_books):
    """Convert dictionary of sparse vectors to scipy sparse matrix."""
    print("\nConverting to sparse matrix...")
    
    user_ids = list(user_vectors.keys())
    n_users = len(user_ids)
    
    # Create sparse matrix
    matrix = lil_matrix((n_users, n_books), dtype=np.float32)
    
    for i, user_id in enumerate(user_ids):
        user_vector = user_vectors[user_id]
        for idx_str, rating in user_vector.items():
            idx = int(idx_str)
            matrix[i, idx] = rating
    
    # Convert to CSR format for efficient operations
    matrix = matrix.tocsr()
    print(f"  Created {matrix.shape[0]} x {matrix.shape[1]} sparse matrix")
    print(f"  Sparsity: {100 * (1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])):.4f}%")
    
    return matrix, user_ids


def perform_dimensionality_reduction(sparse_matrix, n_components=150, random_state=42):
    """
    Apply TruncatedSVD for dimensionality reduction.
    
    Args:
        sparse_matrix: Input sparse matrix
        n_components: Number of components to keep (default: 150)
        random_state: Random seed
    
    Returns:
        Reduced dense matrix and fitted SVD model
    """
    from sklearn.decomposition import TruncatedSVD
    
    print(f"\nPerforming dimensionality reduction with TruncatedSVD...")
    print(f"  Reducing from {sparse_matrix.shape[1]} to {n_components} dimensions")
    
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state,
        algorithm='randomized'
    )
    
    reduced_matrix = svd.fit_transform(sparse_matrix)
    
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"  Dimensionality reduction complete!")
    print(f"  Explained variance: {explained_variance*100:.2f}%")
    
    return reduced_matrix, svd


def perform_clustering(matrix, n_clusters, random_state=42):
    """
    Perform K-Means clustering with cosine similarity (via L2 normalization).
    
    This achieves the same effect as Spherical K-Means by:
    1. L2-normalizing the data
    2. Running standard K-Means (which uses Euclidean distance)
    3. Euclidean distance on normalized vectors = cosine similarity
    
    Args:
        matrix: Input matrix (dense)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Fitted clustering model, cluster labels, and normalized matrix
    """
    print(f"\nPerforming K-Means clustering with cosine similarity...")
    print(f"  Clustering {n_clusters} clusters using normalized vectors")
    print("  (L2 normalization + Euclidean distance = cosine similarity)")
    
    # L2-normalize the data (equivalent to projecting onto unit sphere)
    normalized_matrix = normalize(matrix, norm='l2')
    
    # Standard K-Means on normalized data is equivalent to Spherical K-Means
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iter=300,
        verbose=1,
        random_state=random_state,
        init='k-means++',
        n_init=10
    )
    
    labels = kmeans.fit_predict(normalized_matrix)
    
    print("  Clustering complete!")
    return kmeans, labels, normalized_matrix


def compute_cluster_statistics(labels, n_clusters):
    """Compute statistics about cluster sizes."""
    print("\n" + "="*50)
    print("CLUSTER SIZE STATISTICS")
    print("="*50)
    
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    
    print(f"\nTotal users: {len(labels)}")
    print(f"Number of clusters: {n_clusters}")
    print(f"\nCluster size statistics:")
    print(f"  Min size: {cluster_sizes.min()}")
    print(f"  Max size: {cluster_sizes.max()}")
    print(f"  Mean size: {cluster_sizes.mean():.2f}")
    print(f"  Median size: {np.median(cluster_sizes):.2f}")
    print(f"  Std dev: {cluster_sizes.std():.2f}")
    print(f"  Variance: {cluster_sizes.var():.2f}")
    
    # Show distribution
    print(f"\nCluster size distribution:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(cluster_sizes, p)
        print(f"  {p}th percentile: {val:.0f}")
    
    # Count empty clusters
    empty_clusters = np.sum(cluster_sizes == 0)
    if empty_clusters > 0:
        print(f"\n  WARNING: {empty_clusters} empty clusters!")
    
    # Show top 10 largest and smallest clusters
    print(f"\n10 Largest clusters:")
    largest_indices = np.argsort(cluster_sizes)[-10:][::-1]
    for idx in largest_indices:
        print(f"  Cluster {idx}: {cluster_sizes[idx]} users")
    
    print(f"\n10 Smallest non-empty clusters:")
    non_empty_sizes = cluster_sizes[cluster_sizes > 0]
    if len(non_empty_sizes) >= 10:
        smallest_indices = np.argsort(cluster_sizes)[cluster_sizes > 0][:10]
        for idx in smallest_indices:
            print(f"  Cluster {idx}: {cluster_sizes[idx]} users")
    
    return cluster_sizes


def assign_multiple_clusters(clustering_matrix, kmeans, user_ids, top_k=3, distance_threshold=None):
    """
    Assign users to multiple clusters based on proximity to cluster centers.
    
    Args:
        clustering_matrix: The matrix used for clustering (normalized)
        kmeans: Fitted KMeans model
        user_ids: List of user IDs
        top_k: Number of closest clusters to assign each user to
        distance_threshold: If provided, only assign to clusters within this distance
    
    Returns:
        Dictionary mapping user_id to list of (cluster_id, distance) tuples
    """
    print(f"\nAssigning users to top-{top_k} nearest clusters...")
    
    # Compute distances to all cluster centers
    distances = kmeans.transform(clustering_matrix)
    
    multi_assignments = {}
    for i, user_id in enumerate(user_ids):
        user_distances = distances[i]
        
        # Get top-k closest clusters
        if distance_threshold is not None:
            # Only include clusters within threshold
            valid_clusters = np.where(user_distances <= distance_threshold)[0]
            if len(valid_clusters) == 0:
                # If no clusters within threshold, assign to closest
                valid_clusters = [np.argmin(user_distances)]
            sorted_indices = valid_clusters[np.argsort(user_distances[valid_clusters])]
        else:
            # Just get top-k
            sorted_indices = np.argsort(user_distances)[:top_k]
        
        # Store cluster IDs and distances
        cluster_assignments = [
            (int(cluster_id), float(user_distances[cluster_id])) 
            for cluster_id in sorted_indices
        ]
        multi_assignments[user_id] = cluster_assignments
    
    # Compute statistics
    assignment_counts = [len(assignments) for assignments in multi_assignments.values()]
    print(f"  Average clusters per user: {np.mean(assignment_counts):.2f}")
    print(f"  Max clusters per user: {np.max(assignment_counts)}")
    
    if distance_threshold:
        single_cluster = sum(1 for c in assignment_counts if c == 1)
        print(f"  Users assigned to single cluster: {single_cluster} ({100*single_cluster/len(user_ids):.1f}%)")
    
    return multi_assignments


def compute_silhouette_score_sample(matrix, labels, sample_size=5000, metric='cosine'):
    """
    Compute silhouette score on a sample (full computation is expensive).
    
    Args:
        matrix: Data matrix
        labels: Cluster labels
        sample_size: Number of samples to use
        metric: Distance metric ('euclidean' or 'cosine')
    """
    if len(labels) > sample_size:
        print(f"\nComputing silhouette score on sample of {sample_size} users...")
        indices = np.random.choice(len(labels), sample_size, replace=False)
        sample_matrix = matrix[indices]
        sample_labels = labels[indices]
        score = silhouette_score(sample_matrix, sample_labels, metric=metric)
    else:
        print("\nComputing silhouette score...")
        score = silhouette_score(matrix, labels, metric=metric)
    
    print(f"  Silhouette score: {score:.4f} (metric: {metric})")
    print("  (Range: -1 to 1, higher is better)")
    return score


def save_clustering_results(user_ids, labels, kmeans, cluster_sizes, output_path, 
                            multi_assignments=None, svd_model=None):
    """Save clustering results to JSON file."""
    print(f"\nSaving clustering results to {output_path}...")
    
    # Create user_id to cluster mapping
    user_to_cluster = {user_ids[i]: int(labels[i]) for i in range(len(user_ids))}
    
    # save cluster centroids

    cluster_centroids = {
        cluster_id: centroid.tolist() 
        for cluster_id, centroid in enumerate(kmeans.cluster_centers_)
    }

    # for cluster_id in range(len(kmeans.cluster_centers_)):
    #     centroid = kmeans.cluster_centers_[cluster_id]
    #     # Only store non-zero entries
    #     sparse_centroid = {}
    #     for idx in range(len(centroid)):
    #         if centroid[idx] != 0:
    #             sparse_centroid[idx] = float(centroid[idx])
    #     cluster_centroids[cluster_id] = sparse_centroid
    
    print(f"  Converted {len(cluster_centroids)} centroids to sparse format")
    
    output_data = {
        'metadata': {
            'n_users': len(user_ids),
            'n_clusters': len(cluster_sizes),
            'n_dimensions': kmeans.cluster_centers_.shape[1],
            'clustering_algorithm': 'KMeans_normalized',
            'distance_metric': 'cosine',
            'dimensionality_reduction': svd_model is not None,
            'multi_cluster_assignment': multi_assignments is not None
        },
        'cluster_statistics': {
            'min_size': int(cluster_sizes.min()),
            'max_size': int(cluster_sizes.max()),
            'mean_size': float(cluster_sizes.mean()),
            'median_size': float(np.median(cluster_sizes)),
            'std_size': float(cluster_sizes.std()),
            'variance_size': float(cluster_sizes.var())
        },
        'cluster_centroids': cluster_centroids,
        'user_to_cluster': user_to_cluster
    }
    
    # Add SVD info if available
    if svd_model is not None:
        output_data['metadata']['svd_components'] = svd_model.n_components
        output_data['metadata']['svd_explained_variance'] = float(svd_model.explained_variance_ratio_.sum())
        output_data['metadata']['centroid_space'] = 'reduced_svd_normalized'
    else:
        output_data['metadata']['centroid_space'] = 'original'
    
    # Add multi-cluster assignments if available
    if multi_assignments is not None:
        output_data['multi_cluster_assignments'] = multi_assignments
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved clustering for {len(user_ids)} users with {len(cluster_centroids)} centroids")


def main():
    parser = argparse.ArgumentParser(
        description='Cluster user embeddings using dimensionality reduction + K-Means with cosine similarity'
    )
    parser.add_argument('--multi-cluster', action='store_true',
                        help='Assign users to multiple clusters based on proximity')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of clusters to assign each user to (default: 3)')
    parser.add_argument('--distance-threshold', type=float, default=None,
                        help='Optional: only assign to clusters within this distance')
    parser.add_argument('--n-components', type=int, default=300,
                        help='Number of SVD components for dimensionality reduction (default: 300)')
    
    args = parser.parse_args()
    
    # Define file paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    input_path = data_dir / 'server/user_embeddings_train.json'
    output_path = data_dir / 'server/user_clusters_cosine.json'
    
    # Check if input file exists
    if not input_path.exists():
        print(f"ERROR: Embeddings file not found at {input_path}")
        print(f"Please run create_user_embeddings.py --dataset train first")
        return
    
    # Load embeddings
    user_vectors, n_users, n_books = load_embeddings(input_path)
    
    # Convert to sparse matrix
    sparse_matrix, user_ids = convert_to_sparse_matrix(user_vectors, n_books)
    
    # Calculate number of clusters: sqrt(n_users)
    n_clusters = int(np.sqrt(n_users))
    print(f"\nNumber of users: {n_users}")
    print(f"Number of clusters: {n_clusters} (sqrt of {n_users})")
    
    # Perform dimensionality reduction
    reduced_matrix, svd_model = perform_dimensionality_reduction(
        sparse_matrix, 
        n_components=args.n_components
    )
    
    # Perform clustering on reduced data
    kmeans, labels, clustering_matrix = perform_clustering(
        reduced_matrix, 
        n_clusters
    )
    
    # Compute and display statistics
    cluster_sizes = compute_cluster_statistics(labels, n_clusters)
    
    # Compute silhouette score with cosine metric
    compute_silhouette_score_sample(clustering_matrix, labels, sample_size=5000, metric='cosine')
    
    # Optional: multi-cluster assignment
    multi_assignments = None
    if args.multi_cluster:
        print("\n" + "="*50)
        print("MULTI-CLUSTER ASSIGNMENT")
        print("="*50)
        multi_assignments = assign_multiple_clusters(
            clustering_matrix, kmeans, user_ids, 
            top_k=args.top_k,
            distance_threshold=args.distance_threshold
        )
    
    # Save results
    save_clustering_results(
        user_ids, labels, kmeans, cluster_sizes, output_path,
        multi_assignments=multi_assignments,
        svd_model=svd_model
    )
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    print(f"Clustering results saved to: {output_path}")


if __name__ == '__main__':
    main()