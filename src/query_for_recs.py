#!/usr/bin/env python3
"""
Query-based recommendation system.

This script handles the full pipeline for generating recommendations for querying users:
1. Load the saved SVD model and cluster data
2. Transform the query user's sparse profile into the reduced embedding space
3. Find the nearest cluster centroid using cosine similarity
4. Return the pre-computed recommendations for that cluster

Usage:
    # Generate recommendations for users in a query file (single cluster mode)
    python query_recommendations.py --query-file user_embeddings.json --output query_recs.json
    
    # Generate recommendations for a single user (by ID from the embeddings file)
    python query_recommendations.py --user-id "345b3a87c34de1889ede1ca429998776" --embeddings-file user_embeddings.json
    
    # Use equal-split multi-assignment to pull from top-k clusters equally
    python query_recommendations.py --multiassignment-equal 3 --embeddings-file user_embeddings.json
    
    # Use proportional multi-assignment to pull from top-k clusters based on similarity
    python query_recommendations.py --multiassignment-prop 3 --embeddings-file user_embeddings.json
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
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Import PIR functionality
from pir.pir_scheme import (
    PIRClient, 
    PIRServer, 
    setup_pir_database,
    retrieve_recommendations_with_pir,
    load_pir_server
)

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


def get_equal_recs_from_clusters(top_clusters, cluster_recommendations, total_recs=100):
    """
    Get recommendations from multiple clusters with equal allocation.
    
    Args:
        top_clusters: List of (cluster_id, similarity) tuples, sorted by similarity desc
        cluster_recommendations: Dict mapping cluster_id -> list of recommendations
        total_recs: Total number of unique recommendations to return
    
    Returns:
        List of unique recommendation IDs
    """
    k = len(top_clusters)
    
    # Calculate equal allocation per cluster
    base_allocation = total_recs // k
    remainder = total_recs % k
    
    allocations = [base_allocation] * k
    # Distribute remainder to first clusters
    for i in range(remainder):
        allocations[i] += 1
    
    # Collect recommendations, respecting allocations but ensuring uniqueness
    seen = set()
    final_recs = []
    cluster_recs_used = {cluster_id: 0 for cluster_id, _ in top_clusters}
    
    # First pass: get allocated number from each cluster
    for (cluster_id, _), allocation in zip(top_clusters, allocations):
        recs = cluster_recommendations.get(cluster_id, [])
        added = 0
        
        for rec in recs:
            if rec not in seen and added < allocation:
                seen.add(rec)
                final_recs.append(rec)
                added += 1
                cluster_recs_used[cluster_id] += 1
        
        if added < allocation:
            print(f"Warning: Cluster {cluster_id} only had {added} unique recs, needed {allocation}")
    
    # Second pass: if we're short due to duplicates, fill from any cluster
    if len(final_recs) < total_recs:
        for cluster_id, _ in top_clusters:
            recs = cluster_recommendations.get(cluster_id, [])
            for rec in recs:
                if rec not in seen:
                    seen.add(rec)
                    final_recs.append(rec)
                    if len(final_recs) >= total_recs:
                        break
            if len(final_recs) >= total_recs:
                break
    
    return final_recs[:total_recs]


def get_proportional_recs_from_clusters(top_clusters, cluster_recommendations, total_recs=100):
    """
    Get recommendations from multiple clusters, with count proportional to similarity.
    
    Args:
        top_clusters: List of (cluster_id, similarity) tuples, sorted by similarity desc
        cluster_recommendations: Dict mapping cluster_id -> list of recommendations
        total_recs: Total number of unique recommendations to return
    
    Returns:
        List of unique recommendation IDs
    """
    # Extract similarities and normalize to get proportions
    similarities = np.array([sim for _, sim in top_clusters])
    
    # Shift similarities to be positive if any are negative (rare but possible)
    if similarities.min() < 0:
        similarities = similarities - similarities.min() + 0.01
    
    # Normalize to get proportions that sum to 1
    proportions = similarities / similarities.sum()
    
    # Calculate initial allocation per cluster
    allocations = (proportions * total_recs).astype(int)
    
    # Distribute any remainder to top clusters
    remainder = total_recs - allocations.sum()
    for i in range(remainder):
        allocations[i % len(allocations)] += 1
    
    # Collect recommendations, respecting allocations but ensuring uniqueness
    seen = set()
    final_recs = []
    
    # First pass: get allocated number from each cluster
    cluster_recs_used = {cluster_id: 0 for cluster_id, _ in top_clusters}
    
    for (cluster_id, _), allocation in zip(top_clusters, allocations):
        recs = cluster_recommendations.get(cluster_id, [])
        added = 0
        for rec in recs:
            if rec not in seen and added < allocation:
                seen.add(rec)
                final_recs.append(rec)
                added += 1
                cluster_recs_used[cluster_id] += 1
    
    # Second pass: if we need more recs (due to overlap), pull from clusters in order
    if len(final_recs) < total_recs:
        for cluster_id, _ in top_clusters:
            recs = cluster_recommendations.get(cluster_id, [])
            for rec in recs:
                if rec not in seen:
                    seen.add(rec)
                    final_recs.append(rec)
                    if len(final_recs) >= total_recs:
                        break
            if len(final_recs) >= total_recs:
                break
    
    return final_recs[:total_recs]


def get_recommendations_for_user(user_sparse_dict, n_books, svd_model, 
                                  centroids, cluster_recommendations, book_map, 
                                  multiassignment_equal=None, multiassignment_prop=None, 
                                  top_k_recs=100,
                                  pir_client=None, pir_server=None, use_pir=True):
    """
    Full pipeline: transform user profile -> find cluster -> get recommendations.
    
    Args:
        user_sparse_dict: User's sparse rating dictionary
        n_books: Total number of books
        svd_model: Fitted SVD model
        centroids: Cluster centroids
        cluster_recommendations: Pre-computed recommendations per cluster
        book_map: Book index to title mapping
        multiassignment_equal: If set, number of clusters for equal split
        multiassignment_prop: If set, number of clusters for proportional split based on similarity
        top_k_recs: Total number of recommendations to return
        pir_client: PIR client instance (optional)
        pir_server: PIR server instance (optional)
        use_pir: Whether to use PIR for private retrieval
    
    Returns:
        Dictionary with cluster info and recommendations
    """
    # Transform user profile to embedding space
    user_embedding = transform_user_profile(user_sparse_dict, n_books, svd_model)
    
    # Multi-assignment proportional mode
    if multiassignment_prop is not None and multiassignment_prop > 1:
        top_clusters = find_top_k_clusters(user_embedding, centroids, multiassignment_prop)
        
        # If using PIR, retrieve recommendations for each cluster privately
        if use_pir and pir_client is not None and pir_server is not None:
            # Create a modified cluster_recommendations dict with PIR-retrieved recs
            pir_cluster_recs = {}
            for cluster_id, _ in top_clusters:
                pir_cluster_recs[cluster_id] = retrieve_recommendations_with_pir(
                    cluster_id, pir_client, pir_server
                )
            # Get proportional recommendations using PIR-retrieved data
            recommendations = get_proportional_recs_from_clusters(
                top_clusters, pir_cluster_recs, total_recs=top_k_recs
            )
        else:
            # Get proportional recommendations using direct lookup
            recommendations = get_proportional_recs_from_clusters(
                top_clusters, cluster_recommendations, total_recs=top_k_recs
            )
        
        # Build cluster info
        cluster_ids = [cluster_id for cluster_id, _ in top_clusters]
        cluster_assignment = ", ".join(str(c) for c in cluster_ids)
        
        cluster_info = [
            {'cluster_id': cluster_id, 'similarity': float(similarity)}
            for cluster_id, similarity in top_clusters
        ]
        
        recs_with_title = []
        for rec in recommendations:
            book_title = book_map.get(str(rec), "Unknown")
            recs_with_title.append([rec, book_title])
        
        return {
            'cluster_assignment': cluster_assignment,
            'top_clusters': cluster_info,
            'recommendations': recommendations,  # use recs_with_title to include titles
            'privacy_preserved': use_pir  # track whether PIR was used
        }
    
    # Multi-assignment equal mode
    elif multiassignment_equal is not None and multiassignment_equal > 1:
        top_clusters = find_top_k_clusters(user_embedding, centroids, multiassignment_equal)
        
        # If using PIR, retrieve recommendations for each cluster privately
        if use_pir and pir_client is not None and pir_server is not None:
            # Create a modified cluster_recommendations dict with PIR-retrieved recs
            pir_cluster_recs = {}
            for cluster_id, _ in top_clusters:
                pir_cluster_recs[cluster_id] = retrieve_recommendations_with_pir(
                    cluster_id, pir_client, pir_server
                )
            # Get equal-split recommendations using PIR-retrieved data
            recommendations = get_equal_recs_from_clusters(
                top_clusters, pir_cluster_recs, total_recs=top_k_recs
            )
        else:
            # Get equal-split recommendations using direct lookup
            recommendations = get_equal_recs_from_clusters(
                top_clusters, cluster_recommendations, total_recs=top_k_recs
            )
        
        # Build cluster info
        cluster_ids = [cluster_id for cluster_id, _ in top_clusters]
        cluster_assignment = ", ".join(str(c) for c in cluster_ids)
        
        cluster_info = [
            {'cluster_id': cluster_id, 'similarity': float(similarity)}
            for cluster_id, similarity in top_clusters
        ]
        
        recs_with_title = []
        for rec in recommendations:
            book_title = book_map.get(str(rec), "Unknown")
            recs_with_title.append([rec, book_title])
        
        return {
            'cluster_assignment': cluster_assignment,
            'top_clusters': cluster_info,
            'recommendations': recommendations,  # use recs_with_title to include titles
            'privacy_preserved': use_pir  # track whether PIR was used
        }
    
    # Single cluster mode (default)
    else:
        # Single cluster assignment
        cluster_id, similarity, _ = find_nearest_cluster(user_embedding, centroids)
        
        # Use PIR if available, otherwise direct lookup
        if use_pir and pir_client is not None and pir_server is not None:
            # PRIVATE retrieval using PIR
            recommendations = retrieve_recommendations_with_pir(
                cluster_id, pir_client, pir_server
            )
        # Non-private direct lookup
        else:
            recommendations = cluster_recommendations.get(cluster_id, [])
        
        recs_with_title = []
        for rec in recommendations:
            book_title = book_map.get(str(rec), "Unknown")
            recs_with_title.append([rec, book_title])
        
        return {
            'cluster_id': cluster_id,
            'similarity': float(similarity),
            'recommendations': recommendations, #use recs_with_title to include titles to visualize recs
            'privacy-preserved': use_pir # track whether or not PIR was used to serve this rec
        }


def process_query_users(embeddings_data, svd_model, centroids, 
                        cluster_recommendations, n_books, book_map, 
                        multiassignment_equal=None, multiassignment_prop=None, 
                        top_k_recs=100,
                        pir_client=None, pir_server=None, use_pir=True):
    """
    Process all users in a query embeddings file.
    
    Args:
        embeddings_data: Loaded embeddings data
        svd_model: Fitted SVD model
        centroids: Cluster centroids
        cluster_recommendations: Pre-computed recommendations per cluster
        n_books: Total number of books
        book_map: Book index to title mapping
        multiassignment_equal: If set, number of clusters for equal split
        multiassignment_prop: If set, number of clusters for proportional assignment
        top_k_recs: Number of recommendations per user
        pir_client: PIR client instance (optional)
        pir_server: PIR server instance (optional)
        use_pir: Whether to use PIR for private retrieval
    
    Returns:
        Dictionary mapping user_id -> recommendations info
    """
    user_vectors = embeddings_data['user_vectors']
    
    print(f"  Found {len(user_vectors)} users to process")
    if multiassignment_prop:
        print(f"  Using multi-assignment proportional with {multiassignment_prop} clusters")
    elif multiassignment_equal:
        print(f"  Using multi-assignment equal split with {multiassignment_equal} clusters")
    else:
        print(f"  Using single cluster assignment")
    
    results = {}
    
    for i, (user_id, sparse_dict) in enumerate(user_vectors.items()):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(user_vectors)} users...")
        
        result = get_recommendations_for_user(
            sparse_dict, n_books, svd_model, centroids, 
            cluster_recommendations, book_map, 
            multiassignment_equal, multiassignment_prop, top_k_recs,
            pir_client=pir_client, pir_server=pir_server, use_pir=use_pir
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
    parser.add_argument('--embeddings-file', type=str, default='train_user_profiles.json',
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
    parser.add_argument('--multiassignment-equal', type=int, default=None,
                        help='Pull recommendations from top-k clusters with equal split')
    parser.add_argument('--multiassignment-prop', type=int, default=None,
                        help='Pull recommendations from top-k clusters proportionally based on similarity')
    parser.add_argument('--top-k-recs', type=int, default=100,
                        help='Number of recommendations to return per user (default: 100)')
    
    # Data directory
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Base data directory (default: ../data relative to script)')
    
    # Option to disable PIR (for testing purposes)
    parser.add_argument('--no-pir', dest='use_pir', action='store_false', default=True,
                        help='Disable PIR and use direct lookup')
    
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
    
    pir_server_path = server_dir.parent / 'pir_server.npz'
    pir_params_path = server_dir.parent / 'pir_params.json'

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
    
    #  Load PIR system unless otherwise requested
    pir_client = None
    pir_server = None
    if args.use_pir:
        print("\nLoading PIR system...")
        try:
            pir_server = load_pir_server(pir_server_path)
            with open(pir_params_path, 'r') as f:
                pir_params = json.load(f)
            pir_client = PIRClient(pir_params)
            print("  ✓ PIR system loaded successfully!")
        except FileNotFoundError as e:
            print(f"  ✗ PIR files not found: {e}")
            print(f"  → Run 'python setup_pir.py' first to generate PIR database")
            print(f"  → Falling back to non-private retrieval")
            args.use_pir = False
    else:
        print("\nPIR disabled - using non-private retrieval")

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
            cluster_recommendations, book_idx_to_title, 
            args.multiassignment_equal, args.multiassignment_prop, args.top_k_recs,
            pir_client=pir_client, pir_server=pir_server, use_pir=args.use_pir
        )
        
        print(f"\nResults for user {args.user_id}:")
        print(json.dumps(result, indent=2))
        
        # Save single user result
        results = {args.user_id: result}
        
    else:
        # get recommendations for everyone
        results = process_query_users(
            embeddings_data, svd_model, centroids,
            cluster_recommendations, n_books, book_idx_to_title, 
            args.multiassignment_equal, args.multiassignment_prop, args.top_k_recs,
            pir_client=pir_client, pir_server=pir_server, use_pir=args.use_pir
        )
    
        # Save results
        metadata = {
            'n_users': len(results),
            'multiassignment_equal': args.multiassignment_equal,
            'multiassignment_prop': args.multiassignment_prop,
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