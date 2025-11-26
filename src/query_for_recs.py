#!/usr/bin/env python3
"""
Query-based recommendation system.

This script handles the full pipeline for generating recommendations for querying users:
1. Load the saved SVD model and cluster data
2. Transform the query user's sparse profile into the reduced embedding space
3. Find the nearest cluster centroid using cosine similarity
4. Return the pre-computed recommendations for that cluster

Usage:
    # Generate recommendations for users in a query file
    python query_recommendations.py --query-file user_embeddings.json --output query_recs.json
    
    # Generate recommendations for a single user (by ID from the embeddings file)
    python query_recommendations.py --user-id "345b3a87c34de1889ede1ca429998776" --embeddings-file user_embeddings.json
"""
"""
naomi notes: NOT PRIVATE
for computational efficieny, i propose we first tune and evaluate
 the system to get the best recs (i.e. experiment with clustering, etc)
and then add simplePIR once we are getting the recs we want
to visualize titles, replace recommendations with recs_with_titles on line 203

TO GET RECS FOR TRAIN USER PROFILES: 
python query_for_recs.py --embeddings-file train_user_profiles.json
"""

import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.preprocessing import normalize


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_book_titles(book_title_path):
    with open(book_title_path, 'r') as f:
        data = json.load(f)
    return data['book_index_to_title']

def load_svd_model(model_path):
    """Load the saved TruncatedSVD model."""
    print(f"Loading SVD model from {model_path}...")
    with open(model_path, 'rb') as f:
        svd_model = pickle.load(f)
    print(f"  SVD components: {svd_model.n_components}")
    return svd_model


def load_cluster_data(clusters_path):
    """Load cluster centroids and metadata."""
    print(f"Loading cluster data from {clusters_path}...")
    data = load_json_file(clusters_path)
    
    # Convert centroids to numpy array
    centroids = {}
    for cluster_id, centroid in data['cluster_centroids'].items():
        centroids[int(cluster_id)] = np.array(centroid, dtype=np.float32)
    
    print(f"  Loaded {len(centroids)} cluster centroids")
    print(f"  Centroid dimension: {len(next(iter(centroids.values())))}")
    
    return centroids, data['metadata']


def load_cluster_recommendations(recs_path):
    """Load pre-computed cluster recommendations."""
    print(f"Loading cluster recommendations from {recs_path}...")
    data = load_json_file(recs_path)
    
    # Convert keys to integers
    recommendations = {
        int(k): v for k, v in data['cluster_recommendations'].items()
    }
    
    print(f"  Loaded recommendations for {len(recommendations)} clusters")
    return recommendations, data.get('metadata', {})


def sparse_dict_to_vector(sparse_dict, n_books):
    """Convert a sparse dictionary representation to a sparse matrix row."""
    vector = lil_matrix((1, n_books), dtype=np.float32)
    for idx_str, rating in sparse_dict.items():
        idx = int(idx_str)
        vector[0, idx] = rating
    return vector.tocsr()


def transform_user_profile(user_sparse_dict, n_books, svd_model):
    """
    Transform a user's sparse profile into the reduced embedding space.
    
    Args:
        user_sparse_dict: Dictionary mapping book_index -> rating
        n_books: Total number of books in the system
        svd_model: Fitted TruncatedSVD model
    
    Returns:
        Normalized embedding vector for the user
    """
    # Convert sparse dict to sparse matrix
    sparse_vector = sparse_dict_to_vector(user_sparse_dict, n_books)
    
    # Apply SVD transformation
    reduced_vector = svd_model.transform(sparse_vector)
    
    # L2 normalize (for cosine similarity comparison with centroids)
    normalized_vector = normalize(reduced_vector, norm='l2')
    
    return normalized_vector.flatten()


def find_nearest_cluster(user_embedding, centroids):
    """
    Find the nearest cluster centroid using cosine similarity.
    
    Since both user embedding and centroids are L2-normalized,
    cosine similarity = dot product, and
    smaller Euclidean distance = higher cosine similarity.
    
    Args:
        user_embedding: Normalized user embedding vector
        centroids: Dictionary mapping cluster_id -> centroid vector
    
    Returns:
        Tuple of (best_cluster_id, similarity_score, all_similarities)
    """
    similarities = {}
    
    for cluster_id, centroid in centroids.items():
        # Cosine similarity via dot product (both vectors are normalized)
        similarity = np.dot(user_embedding, centroid)
        similarities[cluster_id] = similarity
    
    # Find cluster with highest similarity
    best_cluster_id = max(similarities, key=similarities.get)
    best_similarity = similarities[best_cluster_id]
    
    return best_cluster_id, best_similarity, similarities


def find_top_k_clusters(user_embedding, centroids, k=3):
    """
    Find the top-k nearest cluster centroids.
    
    Args:
        user_embedding: Normalized user embedding vector
        centroids: Dictionary mapping cluster_id -> centroid vector
        k: Number of top clusters to return
    
    Returns:
        List of (cluster_id, similarity_score) tuples, sorted by similarity
    """
    similarities = {}
    
    for cluster_id, centroid in centroids.items():
        similarity = np.dot(user_embedding, centroid)
        similarities[cluster_id] = similarity
    
    # Sort by similarity descending
    sorted_clusters = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_clusters[:k]


def get_recommendations_for_user(user_sparse_dict, n_books, svd_model, 
                                  centroids, cluster_recommendations, book_map, top_k_clusters=1):
    """
    Full pipeline: transform user profile -> find cluster -> get recommendations.
    
    Args:
        user_sparse_dict: User's sparse rating dictionary
        n_books: Total number of books
        svd_model: Fitted SVD model
        centroids: Cluster centroids
        cluster_recommendations: Pre-computed recommendations per cluster
        top_k_clusters: Number of clusters to consider (1 for single cluster assignment)
    
    Returns:
        Dictionary with cluster info and recommendations
    """
    # Transform user profile to embedding space
    user_embedding = transform_user_profile(user_sparse_dict, n_books, svd_model)
    
    if top_k_clusters == 1:
        # Single cluster assignment
        cluster_id, similarity, _ = find_nearest_cluster(user_embedding, centroids)
        recommendations = cluster_recommendations.get(cluster_id, [])
        
        recs_with_title = []
        for rec in recommendations:
            book_title = book_map[str(rec)]
            recs_with_title.append([rec, book_title])
        
        return {
            'cluster_id': cluster_id,
            'similarity': float(similarity),
            'recommendations': recommendations #use recs_with_title to include titles to visualize recs
        }
    else:
        # Multi-cluster assignment
        top_clusters = find_top_k_clusters(user_embedding, centroids, top_k_clusters)
        
        # Aggregate recommendations from top clusters (deduplicated, ordered by first appearance)
        seen = set()
        aggregated_recs = []
        cluster_info = []
        
        for cluster_id, similarity in top_clusters:
            cluster_info.append({
                'cluster_id': cluster_id,
                'similarity': float(similarity)
            })
            recs = cluster_recommendations.get(cluster_id, [])
            for rec in recs:
                if rec not in seen:
                    seen.add(rec)
                    aggregated_recs.append(rec)
        
        return {
            'top_clusters': cluster_info,
            'recommendations': aggregated_recs
        }


def process_query_users(embeddings_data, svd_model, centroids, 
                        cluster_recommendations, n_books, book_map, top_k_clusters=1):
    """
    Process all users in a query embeddings file.
    
    Args:
        query_embeddings_path: Path to user embeddings JSON file
        svd_model: Fitted SVD model
        centroids: Cluster centroids
        cluster_recommendations: Pre-computed recommendations per cluster
        n_books: Total number of books
        top_k_clusters: Number of clusters to consider per user
    
    Returns:
        Dictionary mapping user_id -> recommendations info
    """
    user_vectors = embeddings_data['user_vectors']
    
    print(f"  Found {len(user_vectors)} users to process")
    
    results = {}
    
    for i, (user_id, sparse_dict) in enumerate(user_vectors.items()):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(user_vectors)} users...")
        
        result = get_recommendations_for_user(
            sparse_dict, n_books, svd_model, centroids, 
            cluster_recommendations, book_map, top_k_clusters
        )
        results[user_id] = result
    
    print(f"  Completed processing {len(results)} users")
    return results


def save_results(results, output_path, metadata=None):
    """Save recommendation results to JSON file."""
    print(f"\nSaving results to {output_path}...")
    
    output_data = {
        'metadata': metadata or {},
        'user_recommendations': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved recommendations for {len(results)} users")


def main():
    parser = argparse.ArgumentParser(
        description='Generate recommendations for querying users'
    )
    
    # Input options
    parser.add_argument('--embeddings-file', type=str, default='active_users_embeddings_test.json',
                        help='Path to querying user embeddings JSON file')
    parser.add_argument('--user-id', type=str,
                        help='Single user ID to query (requires --embeddings-file)')
    
    # Model/data paths
    parser.add_argument('--svd-model', type=str, default='svd_model.pkl',
                        help='Path to saved SVD model (default: svd_model.pkl)')
    parser.add_argument('--clusters', type=str, default='user_clusters.json',
                        help='Path to cluster data JSON (default: user_clusters.json)')
    parser.add_argument('--cluster-recs', type=str, default='recs_per_cluster.json',
                        help='Path to cluster recommendations JSON')
    
    # Output options
    parser.add_argument('--output', type=str, default='pinpointe_train_recs.json',
                        help='Output path for recommendations (default: user_recs.json)')
    
    # Algorithm options
    parser.add_argument('--top-k-clusters', type=int, default=1,
                        help='Number of clusters to consider per user (default: 1)')
    parser.add_argument('--top-k-recs', type=int, default=100,
                        help='Number of recommendations to return per user (default: 100)')
    
    # Data directory
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Base data directory (default: ../data relative to script)')
    
    args = parser.parse_args()
    
    # Resolve paths
    if args.data_dir:
        server_dir = Path(args.data_dir)
    else:
        script_dir = Path(__file__).parent
        server_dir = script_dir.parent / 'data' / 'server'
        clients_dir = script_dir.parent / 'data' / 'clients'
        eval_dir = script_dir.parent / 'data' / 'eval'
    
    svd_model_path = server_dir / args.svd_model
    clusters_path = server_dir / args.clusters
    cluster_recs_path = server_dir / args.cluster_recs
    output_path = script_dir.parent / 'data' / 'eval' / 'inputs' / args.output
    embeddings_path = eval_dir / args.embeddings_file
    book_title_path = script_dir.parent / 'data' / 'book_index_to_title.json'
    
    # Load models and data
    print("="*60)
    print("LOADING MODELS AND DATA")
    print("="*60)
    
    svd_model = load_svd_model(svd_model_path)
    centroids, cluster_metadata = load_cluster_data(clusters_path)
    cluster_recommendations, recs_metadata = load_cluster_recommendations(cluster_recs_path)
    embeddings_data = load_json_file(embeddings_path)
    book_idx_to_title = load_book_titles(book_title_path)
    
    n_books = cluster_metadata.get('n_books') or svd_model.n_features_in_
    print(f"  Number of books: {n_books}")
    
    # Process based on input type
    print("\n" + "="*60)
    print("GENERATING RECOMMENDATIONS")
    print("="*60)
    
    if args.user_id:
        # Single user query
        if not args.embeddings_file:
            print("ERROR: --embeddings-file is required when using --user-id")
            return
        
        user_vectors = embeddings_data['user_vectors']
        
        if args.user_id not in user_vectors:
            print(f"ERROR: User ID '{args.user_id}' not found in embeddings file")
            return
        
        user_sparse_dict = user_vectors[args.user_id]
        result = get_recommendations_for_user(
            user_sparse_dict, n_books, svd_model, centroids,
            cluster_recommendations, book_idx_to_title, args.top_k_clusters
        )
        
        # Truncate recommendations
        if 'recommendations' in result:
            result['recommendations'] = result['recommendations'][:args.top_k_recs]
        
        print(f"\nResults for user {args.user_id}:")
        print(json.dumps(result, indent=2))
        
        # Save single user result
        results = {args.user_id: result}
        
    else:
        # get recommendations for everyone
        results = process_query_users(
            embeddings_data, svd_model, centroids,
            cluster_recommendations, n_books, book_idx_to_title, args.top_k_clusters
        )
        
        # Truncate recommendations for each user
        for user_id in results:
            if 'recommendations' in results[user_id]:
                results[user_id]['recommendations'] = \
                    results[user_id]['recommendations'][:args.top_k_recs]
    
        # Save results
        metadata = {
            'n_users': len(results),
            'top_k_clusters': args.top_k_clusters,
            'top_k_recs': args.top_k_recs,
            'svd_components': svd_model.n_components
        }
        save_results(results, output_path, metadata)
        print(f"Recommendations saved to: {output_path}")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)

if __name__ == '__main__':
    main()
