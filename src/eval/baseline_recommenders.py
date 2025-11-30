#!/usr/bin/env python3
"""
Book Recommendation System

Implements multiple recommendation algorithms:
- PopRec: Popularity-based recommendations
- ItemKNN: Item-based collaborative filtering
- UserKNN: User-based collaborative filtering
- MF: Matrix Factorization using ALS (implicit library)

The script trains on the full dataset (user_embeddings_for_baselines.json) 
but only generates recommendations for users in the evaluation set 
(train_user_profiles.json).

Usage:
    python baseline_recommenders.py --algorithm <algo> [options]
    
Examples:
    # Train on full data, evaluate on subset
    python baseline_recommenders.py --algorithm poprec
    
    # Specify custom input files
    python baseline_recommenders.py --algorithm mf \\
        --train-input user_embeddings_for_baselines.json \\
        --eval-users train_user_profiles.json
    
    # Run all algorithms
    python baseline_recommenders.py --algorithm all --top-k 100

    # Get recommendations for a single user
    python baseline_recommenders.py -a mf --user-id "6a85b590f81e32bb05fe65c07ac73d4d"

To visualize titles, replace recs with recs_with_titles on line 380-something
"""

import os
import argparse
import json
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from pathlib import Path


def load_data(filepath):
    """Load JSON data and return user vectors and metadata."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('metadata', {}), data['user_vectors']


def load_eval_user_ids(filepath):
    """Load just the user IDs from a user profiles file (for evaluation subset)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return set(data['user_vectors'].keys())

def load_book_titles(book_title_path):
    with open(book_title_path, 'r') as f:
        data = json.load(f)
    return data['book_index_to_title']

def build_matrices(user_vectors, n_users, n_books):
    """
    Build user-item rating matrix and mappings.
    Returns sparse matrix and index mappings.
    """
    # Create mappings from IDs to indices
    user_ids = list(user_vectors.keys())
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    
    # Collect all book IDs
    book_ids = set()
    for ratings in user_vectors.values():
        book_ids.update(ratings.keys())
    book_ids = sorted(book_ids, key=int)
    book_to_idx = {bid: i for i, bid in enumerate(book_ids)}
    idx_to_book = {i: bid for bid, i in book_to_idx.items()}
    idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    
    # Build sparse matrix (users x books)
    n_actual_users = len(user_ids)
    n_actual_books = len(book_ids)
    
    matrix = lil_matrix((n_actual_users, n_actual_books), dtype=np.float32)
    for uid, ratings in user_vectors.items():
        user_idx = user_to_idx[uid]
        for book_id, rating in ratings.items():
            if book_id in book_to_idx:
                book_idx = book_to_idx[book_id]
                matrix[user_idx, book_idx] = rating
    
    return (csr_matrix(matrix), user_to_idx, book_to_idx, 
            idx_to_user, idx_to_book, user_ids, book_ids)


class PopRecRecommender:
    """Popularity-based recommender using weighted score of count and average rating."""
    
    def __init__(self, rating_matrix, book_to_idx, idx_to_book):
        self.rating_matrix = rating_matrix
        self.book_to_idx = book_to_idx
        self.idx_to_book = idx_to_book
        self.popularity_scores = None
        self.ranked_books = None
        
    def fit(self):
        """Calculate popularity scores for all books."""
        # Count non-zero ratings per book
        rating_counts = np.array((self.rating_matrix > 0).sum(axis=0)).flatten()
        
        # Calculate sum of ratings per book
        rating_sums = np.array(self.rating_matrix.sum(axis=0)).flatten()
        
        # Average rating (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_ratings = np.where(rating_counts > 0, 
                                   rating_sums / rating_counts, 0)
        
        # Weighted popularity score: combines count and average
        # Using Bayesian average to handle items with few ratings
        C = np.mean(rating_counts[rating_counts > 0])  # Average count
        m = np.mean(avg_ratings[avg_ratings > 0])  # Global average rating
        
        # Weighted rating formula (similar to IMDb)
        self.popularity_scores = (rating_counts / (rating_counts + C)) * avg_ratings + \
                                  (C / (rating_counts + C)) * m
        
        # Rank books by popularity
        self.ranked_books = np.argsort(-self.popularity_scores)
        
    def recommend(self, user_idx, user_rated_books, top_k=100):
        """Return top-k popular books not already rated by user."""
        recommendations = []
        for book_idx in self.ranked_books:
            if book_idx not in user_rated_books:
                recommendations.append(self.idx_to_book[book_idx])
                if len(recommendations) >= top_k:
                    break
        return recommendations


class ItemKNNRecommender:
    """Item-based collaborative filtering using cosine similarity."""
    
    def __init__(self, rating_matrix, book_to_idx, idx_to_book, n_neighbors=50):
        self.rating_matrix = rating_matrix
        self.book_to_idx = book_to_idx
        self.idx_to_book = idx_to_book
        self.n_neighbors = n_neighbors
        self.item_similarity = None
        
    def fit(self):
        """Compute item-item similarity matrix."""
        # Transpose to get items as rows
        item_matrix = self.rating_matrix.T.tocsr()
        
        # Compute cosine similarity between items
        # Process in batches to manage memory
        n_items = item_matrix.shape[0]
        batch_size = 1000
        
        # Store top-k neighbors for each item to save memory
        self.item_neighbors = {}
        self.item_similarities = {}
        
        print("Computing item similarities...")
        for start in tqdm(range(0, n_items, batch_size)):
            end = min(start + batch_size, n_items)
            batch = item_matrix[start:end]
            
            # Compute similarity of this batch against all items
            sim_batch = cosine_similarity(batch, item_matrix)
            
            for i, row in enumerate(sim_batch):
                item_idx = start + i
                # Get top neighbors (excluding self)
                row[item_idx] = -1  # Exclude self
                top_indices = np.argsort(-row)[:self.n_neighbors]
                top_scores = row[top_indices]
                
                # Only keep positive similarities
                mask = top_scores > 0.001
                self.item_neighbors[item_idx] = top_indices[mask]
                self.item_similarities[item_idx] = top_scores[mask]
    
    def recommend(self, user_idx, user_ratings, top_k=100):
        """
        Recommend items based on similarity to items user rated highly.
        user_ratings: dict of {book_idx: rating}
        """
        scores = defaultdict(float)
        weights = defaultdict(float)
        
        # For each item the user rated
        for rated_item, rating in user_ratings.items():
            if rating <= 0:
                continue
            if rated_item not in self.item_neighbors:
                continue
                
            # Add weighted scores from similar items
            neighbors = self.item_neighbors[rated_item]
            sims = self.item_similarities[rated_item]
            
            for neighbor, sim in zip(neighbors, sims):
                if neighbor not in user_ratings:
                    scores[neighbor] += sim * rating
                    weights[neighbor] += sim
        
        # Normalize scores
        final_scores = {}
        for item, score in scores.items():
            if weights[item] > 0:
                final_scores[item] = score / weights[item]
        
        # Sort and return top-k
        sorted_items = sorted(final_scores.items(), key=lambda x: -x[1])
        return [self.idx_to_book[item] for item, _ in sorted_items[:top_k]]


class UserKNNRecommender:
    """User-based collaborative filtering using cosine similarity."""
    
    def __init__(self, rating_matrix, book_to_idx, idx_to_book, n_neighbors=50):
        self.rating_matrix = rating_matrix
        self.book_to_idx = book_to_idx
        self.idx_to_book = idx_to_book
        self.n_neighbors = n_neighbors
        
    def fit(self):
        """Precompute user similarities."""
        print("Computing user similarities...")
        n_users = self.rating_matrix.shape[0]
        batch_size = 1000
        
        self.user_neighbors = {}
        self.user_similarities = {}
        
        for start in tqdm(range(0, n_users, batch_size)):
            end = min(start + batch_size, n_users)
            batch = self.rating_matrix[start:end]
            
            sim_batch = cosine_similarity(batch, self.rating_matrix)
            
            for i, row in enumerate(sim_batch):
                user_idx = start + i
                row[user_idx] = -1  # Exclude self
                top_indices = np.argsort(-row)[:self.n_neighbors]
                top_scores = row[top_indices]
                
                mask = top_scores > 0
                self.user_neighbors[user_idx] = top_indices[mask]
                self.user_similarities[user_idx] = top_scores[mask]
    
    def recommend(self, user_idx, user_rated_items, top_k=100):
        """Recommend items based on what similar users liked."""
        if user_idx not in self.user_neighbors:
            return []
            
        scores = defaultdict(float)
        weights = defaultdict(float)
        
        neighbors = self.user_neighbors[user_idx]
        sims = self.user_similarities[user_idx]
        
        # Aggregate ratings from similar users
        for neighbor_idx, sim in zip(neighbors, sims):
            neighbor_ratings = self.rating_matrix[neighbor_idx].toarray().flatten()
            
            for item_idx, rating in enumerate(neighbor_ratings):
                if rating > 0 and item_idx not in user_rated_items:
                    scores[item_idx] += sim * rating
                    weights[item_idx] += sim
        
        # Normalize
        final_scores = {}
        for item, score in scores.items():
            if weights[item] > 0:
                final_scores[item] = score / weights[item]
        
        sorted_items = sorted(final_scores.items(), key=lambda x: -x[1])
        return [self.idx_to_book[item] for item, _ in sorted_items[:top_k]]


class MFRecommender:
    """Matrix Factorization using ALS from implicit library."""
    
    def __init__(self, rating_matrix, book_to_idx, idx_to_book, factors=100, iterations=15):
        self.rating_matrix = rating_matrix
        self.book_to_idx = book_to_idx
        self.idx_to_book = idx_to_book
        self.factors = factors
        self.iterations = iterations
        self.model = None
        
    def fit(self):
        """Train ALS model."""
        from implicit.als import AlternatingLeastSquares
        
        # implicit expects items x users, and confidence values
        # Convert ratings to confidence (higher rating = higher confidence)
        # Using formula: confidence = 1 + alpha * rating
        alpha = 20
        confidence_matrix = self.rating_matrix.copy()
        confidence_matrix.data = 1 + alpha * confidence_matrix.data
        
        # Transpose to items x users format
        # implicit.als treats the first dimension as "users" internally
        # but we want items as the first dimension
        self.item_user_matrix = confidence_matrix.T.tocsr()
        
        print("Training ALS model...")
        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=0.01,
            random_state=42
        )
        self.model.fit(self.item_user_matrix, show_progress=True)
        
    def recommend(self, user_idx, user_rated_items, top_k=100):
        """Get recommendations for a user."""
        # implicit library with items x users matrix:
        # - model.user_factors corresponds to items (shape: n_items x factors)
        # - model.item_factors corresponds to users (shape: n_users x factors)
        
        # Get user's latent factors (from item_factors since we transposed)
        user_factors = self.model.item_factors[user_idx]
        
        # Compute scores for all items (using user_factors which are item factors)
        item_factors = self.model.user_factors  # Shape: (n_items, factors)
        scores = item_factors.dot(user_factors)
        
        # Set scores of already rated items to -inf
        for item_idx in user_rated_items:
            if item_idx < len(scores):
                scores[item_idx] = float('-inf')
        
        # Get top-k items
        top_indices = np.argsort(-scores)[:top_k]
        
        # Filter out any -inf items and convert to book IDs
        recommendations = []
        for item_idx in top_indices:
            if scores[item_idx] > float('-inf'):
                recommendations.append(self.idx_to_book[item_idx])
        
        return recommendations


def generate_recommendations(recommender, user_vectors, user_to_idx, book_to_idx, 
                            idx_to_user, top_k=100, eval_user_ids=None):
    """
    Generate recommendations for users.
    
    Args:
        recommender: The trained recommender model
        user_vectors: All user vectors (from training data)
        user_to_idx: Mapping from user ID to matrix index
        book_to_idx: Mapping from book ID to matrix index
        idx_to_user: Mapping from matrix index to user ID
        top_k: Number of recommendations per user
        eval_user_ids: Optional set of user IDs to generate recommendations for.
                       If None, generates for all users.
    """
    results = {}
    
    # Determine which users to generate recommendations for
    if eval_user_ids is not None:
        # Only evaluate users that exist in both the training data and eval set
        target_users = [uid for uid in eval_user_ids if uid in user_to_idx]
        print(f"Generating recommendations for {len(target_users)} evaluation users "
              f"(out of {len(eval_user_ids)} requested, {len(user_vectors)} in training data)...")
    else:
        target_users = list(user_vectors.keys())
        print(f"Generating recommendations for {len(target_users)} users...")
    
    for uid in tqdm(target_users):
        user_idx = user_to_idx[uid]
        
        # Get user's rated items as indices
        user_ratings = user_vectors[uid]
        user_rated_indices = {book_to_idx[bid] for bid in user_ratings.keys() 
                             if bid in book_to_idx}
        
        # For ItemKNN, we need ratings dict
        if isinstance(recommender, ItemKNNRecommender):
            ratings_dict = {book_to_idx[bid]: rating 
                          for bid, rating in user_ratings.items() 
                          if bid in book_to_idx}
            recs = recommender.recommend(user_idx, ratings_dict, top_k)
        else:
            recs = recommender.recommend(user_idx, user_rated_indices, top_k)
        
        recs = [int(rec) for rec in recs]
        results[uid] = recs
    
    return results

def generate_recommendations_for_one_user(recommender, user_vectors, uid, user_to_idx, book_to_idx, 
                            idx_to_user, book_idx_to_title, top_k=100):
    """Generate recommendations for all users."""
    results = {}
    
    print(f"Generating recommendations for user {uid}")
    user_idx = user_to_idx[uid]
        
    # Get user's rated items as indices
    user_ratings = user_vectors[uid]
    user_rated_indices = {book_to_idx[bid] for bid in user_ratings.keys() 
                            if bid in book_to_idx}
    
    # For ItemKNN, we need ratings dict
    if isinstance(recommender, ItemKNNRecommender):
        ratings_dict = {book_to_idx[bid]: rating 
                        for bid, rating in user_ratings.items() 
                        if bid in book_to_idx}
        recs = recommender.recommend(user_idx, ratings_dict, top_k)
    else:
        recs = recommender.recommend(user_idx, user_rated_indices, top_k)
    
    recs_with_title = []
    for rec in recs:
        book_title = book_idx_to_title[rec]
        recs_with_title.append([rec, book_title])

    recs = [int(rec) for rec in recs]

    results[uid] = recs # recs_with_title
    
    return results


def save_results(results, output_path, algorithm_name):
    """Save recommendations to JSON file."""
    output = {
        "algorithm": algorithm_name,
        "top_k": len(next(iter(results.values()))) if results else 0,
        "n_users": len(results),
        "recommendations": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Saved {len(results)} user recommendations to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Book Recommendation System')
    parser.add_argument('--train-input', default='user_embeddings_for_baselines.json',
                        help='Path to training data JSON file (default: user_embeddings_for_baselines.json)')
    parser.add_argument('--eval-users', default='train_user_profiles.json',
                        help='Path to JSON file containing users to evaluate (default: train_user_profiles.json)')
    parser.add_argument('--user-id', type=str,
                        help='Single user ID to get recs for')
    parser.add_argument('--algorithm', '-a', required=True,
                       choices=['poprec', 'itemknn', 'userknn', 'mf', 'all'],
                       help='Recommendation algorithm to use')
    parser.add_argument('--output', '-o', help='Output JSON file path (default: <algorithm>_recommendations.json)')
    parser.add_argument('--top-k', '-k', type=int, default=100,
                       help='Number of recommendations per user (default: 100)')
    parser.add_argument('--neighbors', '-n', type=int, default=50,
                       help='Number of neighbors for KNN methods (default: 50)')
    parser.add_argument('--factors', '-f', type=int, default=100,
                       help='Number of latent factors for MF (default: 100)')
    
    args = parser.parse_args()
    
    # Determine output path
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir.parent / 'data'
    
    # Training data: full dataset with training profiles substituted
    train_input_path = data_dir / 'eval' / args.train_input
    
    # Evaluation users: subset of users we want recommendations for
    eval_users_path = data_dir / 'eval' / args.eval_users
    
    book_title_path = data_dir / 'book_index_to_title.json'

    # Load training data
    print(f"Loading training data from {train_input_path}...")
    metadata, user_vectors = load_data(train_input_path)
    n_users = metadata.get('n_users', len(user_vectors))
    n_books = metadata.get('n_books')  # Default if not in metadata
    print(f"Loaded {n_users} users, {n_books} books")
    
    # Load evaluation user IDs
    print(f"Loading evaluation user IDs from {eval_users_path}...")
    eval_user_ids = load_eval_user_ids(eval_users_path)
    print(f"Will generate recommendations for {len(eval_user_ids)} evaluation users")

    book_index_to_title = load_book_titles(book_title_path)
    
    # Build matrices from training data
    print("Building rating matrix from training data...")
    (rating_matrix, user_to_idx, book_to_idx, 
     idx_to_user, idx_to_book, user_ids, book_ids) = build_matrices(
        user_vectors, n_users, n_books
    )
    print(f"Matrix shape: {rating_matrix.shape}, non-zeros: {rating_matrix.nnz}")
    
    algorithms = []
    if args.algorithm == 'all':
        algorithms = ['poprec', 'itemknn', 'userknn', 'mf']
    else:
        algorithms = [args.algorithm]
    
    for algo in algorithms:
        print(f"\n{'='*50}")
        print(f"Running {algo.upper()}")
        print('='*50)
        
        if algo == 'poprec':
            recommender = PopRecRecommender(rating_matrix, book_to_idx, idx_to_book)
            recommender.fit()
        elif algo == 'itemknn':
            recommender = ItemKNNRecommender(rating_matrix, book_to_idx, idx_to_book, 
                                            n_neighbors=args.neighbors)
            recommender.fit()
        elif algo == 'userknn':
            recommender = UserKNNRecommender(rating_matrix, book_to_idx, idx_to_book,
                                            n_neighbors=args.neighbors)
            recommender.fit()
        elif algo == 'mf':
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            recommender = MFRecommender(rating_matrix, book_to_idx, idx_to_book,
                                       factors=args.factors)
            recommender.fit()
        
        if args.user_id:
            results = generate_recommendations_for_one_user(
                recommender, user_vectors, args.user_id, user_to_idx, book_to_idx,
                idx_to_user, book_index_to_title, top_k=args.top_k
            )
        else:
            # Generate recommendations only for evaluation users
            results = generate_recommendations(
                recommender, user_vectors, user_to_idx, book_to_idx,
                idx_to_user, top_k=args.top_k, eval_user_ids=eval_user_ids
            )

        if args.output and args.algorithm != 'all':
            output_path = data_dir / 'eval' / 'inputs' / args.output
        else:
            if args.user_id:
                output_path = data_dir / 'eval' / 'inputs' / f"{algo}_{str(args.user_id)}_recs.json"
            else:
                output_path = data_dir / 'eval' / 'inputs' / f"{algo}_train_recs.json"
        
        save_results(results, output_path, algo)


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# """
# Book Recommendation System

# Implements multiple recommendation algorithms:
# - PopRec: Popularity-based recommendations
# - ItemKNN: Item-based collaborative filtering
# - UserKNN: User-based collaborative filtering
# - MF: Matrix Factorization using ALS (implicit library)

# Usage:
#     python recommender.py <input_json> --algorithm <algo> [--output <output_json>] [--top-k <k>]
    
# Examples:
#     python recommender.py reviews.json --algorithm poprec
#     python recommender.py reviews.json --algorithm itemknn --top-k 100
#     python recommender.py reviews.json --algorithm all

# example:
# python3 eval/baseline_recommenders.py -a mf --user-id "6a85b590f81e32bb05fe65c07ac73d4d"

# to visualize titles, replace recs with recs_with_titles on line 380-something
# """

# import os
# import argparse
# import json
# import numpy as np
# from collections import defaultdict
# from scipy.sparse import csr_matrix, lil_matrix
# from sklearn.metrics.pairwise import cosine_similarity
# from tqdm import tqdm
# from pathlib import Path


# def load_data(filepath):
#     """Load JSON data and return user vectors and metadata."""
#     with open(filepath, 'r') as f:
#         data = json.load(f)
#     return data['metadata'], data['user_vectors']

# def load_book_titles(book_title_path):
#     with open(book_title_path, 'r') as f:
#         data = json.load(f)
#     return data['book_index_to_title']

# def build_matrices(user_vectors, n_users, n_books):
#     """
#     Build user-item rating matrix and mappings.
#     Returns sparse matrix and index mappings.
#     """
#     # Create mappings from IDs to indices
#     user_ids = list(user_vectors.keys())
#     user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    
#     # Collect all book IDs
#     book_ids = set()
#     for ratings in user_vectors.values():
#         book_ids.update(ratings.keys())
#     book_ids = sorted(book_ids, key=int)
#     book_to_idx = {bid: i for i, bid in enumerate(book_ids)}
#     idx_to_book = {i: bid for bid, i in book_to_idx.items()}
#     idx_to_user = {i: uid for uid, i in user_to_idx.items()}
    
#     # Build sparse matrix (users x books)
#     n_actual_users = len(user_ids)
#     n_actual_books = len(book_ids)
    
#     matrix = lil_matrix((n_actual_users, n_actual_books), dtype=np.float32)
#     for uid, ratings in user_vectors.items():
#         user_idx = user_to_idx[uid]
#         for book_id, rating in ratings.items():
#             if book_id in book_to_idx:
#                 book_idx = book_to_idx[book_id]
#                 matrix[user_idx, book_idx] = rating
    
#     return (csr_matrix(matrix), user_to_idx, book_to_idx, 
#             idx_to_user, idx_to_book, user_ids, book_ids)


# class PopRecRecommender:
#     """Popularity-based recommender using weighted score of count and average rating."""
    
#     def __init__(self, rating_matrix, book_to_idx, idx_to_book):
#         self.rating_matrix = rating_matrix
#         self.book_to_idx = book_to_idx
#         self.idx_to_book = idx_to_book
#         self.popularity_scores = None
#         self.ranked_books = None
        
#     def fit(self):
#         """Calculate popularity scores for all books."""
#         # Count non-zero ratings per book
#         rating_counts = np.array((self.rating_matrix > 0).sum(axis=0)).flatten()
        
#         # Calculate sum of ratings per book
#         rating_sums = np.array(self.rating_matrix.sum(axis=0)).flatten()
        
#         # Average rating (avoid division by zero)
#         with np.errstate(divide='ignore', invalid='ignore'):
#             avg_ratings = np.where(rating_counts > 0, 
#                                    rating_sums / rating_counts, 0)
        
#         # Weighted popularity score: combines count and average
#         # Using Bayesian average to handle items with few ratings
#         C = np.mean(rating_counts[rating_counts > 0])  # Average count
#         m = np.mean(avg_ratings[avg_ratings > 0])  # Global average rating
        
#         # Weighted rating formula (similar to IMDb)
#         self.popularity_scores = (rating_counts / (rating_counts + C)) * avg_ratings + \
#                                   (C / (rating_counts + C)) * m
        
#         # Rank books by popularity
#         self.ranked_books = np.argsort(-self.popularity_scores)
        
#     def recommend(self, user_idx, user_rated_books, top_k=100):
#         """Return top-k popular books not already rated by user."""
#         recommendations = []
#         for book_idx in self.ranked_books:
#             if book_idx not in user_rated_books:
#                 recommendations.append(self.idx_to_book[book_idx])
#                 if len(recommendations) >= top_k:
#                     break
#         return recommendations


# class ItemKNNRecommender:
#     """Item-based collaborative filtering using cosine similarity."""
    
#     def __init__(self, rating_matrix, book_to_idx, idx_to_book, n_neighbors=50):
#         self.rating_matrix = rating_matrix
#         self.book_to_idx = book_to_idx
#         self.idx_to_book = idx_to_book
#         self.n_neighbors = n_neighbors
#         self.item_similarity = None
        
#     def fit(self):
#         """Compute item-item similarity matrix."""
#         # Transpose to get items as rows
#         item_matrix = self.rating_matrix.T.tocsr()
        
#         # Compute cosine similarity between items
#         # Process in batches to manage memory
#         n_items = item_matrix.shape[0]
#         batch_size = 1000
        
#         # Store top-k neighbors for each item to save memory
#         self.item_neighbors = {}
#         self.item_similarities = {}
        
#         print("Computing item similarities...")
#         for start in tqdm(range(0, n_items, batch_size)):
#             end = min(start + batch_size, n_items)
#             batch = item_matrix[start:end]
            
#             # Compute similarity of this batch against all items
#             sim_batch = cosine_similarity(batch, item_matrix)
            
#             for i, row in enumerate(sim_batch):
#                 item_idx = start + i
#                 # Get top neighbors (excluding self)
#                 row[item_idx] = -1  # Exclude self
#                 top_indices = np.argsort(-row)[:self.n_neighbors]
#                 top_scores = row[top_indices]
                
#                 # Only keep positive similarities
#                 mask = top_scores > 0.001
#                 self.item_neighbors[item_idx] = top_indices[mask]
#                 self.item_similarities[item_idx] = top_scores[mask]
    
#     def recommend(self, user_idx, user_ratings, top_k=100):
#         """
#         Recommend items based on similarity to items user rated highly.
#         user_ratings: dict of {book_idx: rating}
#         """
#         scores = defaultdict(float)
#         weights = defaultdict(float)
        
#         # For each item the user rated
#         for rated_item, rating in user_ratings.items():
#             if rating <= 0:
#                 continue
#             if rated_item not in self.item_neighbors:
#                 continue
                
#             # Add weighted scores from similar items
#             neighbors = self.item_neighbors[rated_item]
#             sims = self.item_similarities[rated_item]
            
#             for neighbor, sim in zip(neighbors, sims):
#                 if neighbor not in user_ratings:
#                     scores[neighbor] += sim * rating
#                     weights[neighbor] += sim
        
#         # Normalize scores
#         final_scores = {}
#         for item, score in scores.items():
#             if weights[item] > 0:
#                 final_scores[item] = score / weights[item]
        
#         # Sort and return top-k
#         sorted_items = sorted(final_scores.items(), key=lambda x: -x[1])
#         return [self.idx_to_book[item] for item, _ in sorted_items[:top_k]]


# class UserKNNRecommender:
#     """User-based collaborative filtering using cosine similarity."""
    
#     def __init__(self, rating_matrix, book_to_idx, idx_to_book, n_neighbors=50):
#         self.rating_matrix = rating_matrix
#         self.book_to_idx = book_to_idx
#         self.idx_to_book = idx_to_book
#         self.n_neighbors = n_neighbors
        
#     def fit(self):
#         """Precompute user similarities."""
#         print("Computing user similarities...")
#         n_users = self.rating_matrix.shape[0]
#         batch_size = 1000
        
#         self.user_neighbors = {}
#         self.user_similarities = {}
        
#         for start in tqdm(range(0, n_users, batch_size)):
#             end = min(start + batch_size, n_users)
#             batch = self.rating_matrix[start:end]
            
#             sim_batch = cosine_similarity(batch, self.rating_matrix)
            
#             for i, row in enumerate(sim_batch):
#                 user_idx = start + i
#                 row[user_idx] = -1  # Exclude self
#                 top_indices = np.argsort(-row)[:self.n_neighbors]
#                 top_scores = row[top_indices]
                
#                 mask = top_scores > 0
#                 self.user_neighbors[user_idx] = top_indices[mask]
#                 self.user_similarities[user_idx] = top_scores[mask]
    
#     def recommend(self, user_idx, user_rated_items, top_k=100):
#         """Recommend items based on what similar users liked."""
#         if user_idx not in self.user_neighbors:
#             return []
            
#         scores = defaultdict(float)
#         weights = defaultdict(float)
        
#         neighbors = self.user_neighbors[user_idx]
#         sims = self.user_similarities[user_idx]
        
#         # Aggregate ratings from similar users
#         for neighbor_idx, sim in zip(neighbors, sims):
#             neighbor_ratings = self.rating_matrix[neighbor_idx].toarray().flatten()
            
#             for item_idx, rating in enumerate(neighbor_ratings):
#                 if rating > 0 and item_idx not in user_rated_items:
#                     scores[item_idx] += sim * rating
#                     weights[item_idx] += sim
        
#         # Normalize
#         final_scores = {}
#         for item, score in scores.items():
#             if weights[item] > 0:
#                 final_scores[item] = score / weights[item]
        
#         sorted_items = sorted(final_scores.items(), key=lambda x: -x[1])
#         return [self.idx_to_book[item] for item, _ in sorted_items[:top_k]]


# class MFRecommender:
#     """Matrix Factorization using ALS from implicit library."""
    
#     def __init__(self, rating_matrix, book_to_idx, idx_to_book, factors=100, iterations=15):
#         self.rating_matrix = rating_matrix
#         self.book_to_idx = book_to_idx
#         self.idx_to_book = idx_to_book
#         self.factors = factors
#         self.iterations = iterations
#         self.model = None
        
#     def fit(self):
#         """Train ALS model."""
#         from implicit.als import AlternatingLeastSquares
        
#         # implicit expects items x users, and confidence values
#         # Convert ratings to confidence (higher rating = higher confidence)
#         # Using formula: confidence = 1 + alpha * rating
#         alpha = 40
#         confidence_matrix = self.rating_matrix.copy()
#         confidence_matrix.data = 1 + alpha * confidence_matrix.data
        
#         # Transpose to items x users format
#         # implicit.als treats the first dimension as "users" internally
#         # but we want items as the first dimension
#         self.item_user_matrix = confidence_matrix.T.tocsr()
        
#         print("Training ALS model...")
#         self.model = AlternatingLeastSquares(
#             factors=self.factors,
#             iterations=self.iterations,
#             regularization=0.01,
#             random_state=42
#         )
#         self.model.fit(self.item_user_matrix, show_progress=True)
        
#     def recommend(self, user_idx, user_rated_items, top_k=100):
#         """Get recommendations for a user."""
#         # implicit library with items x users matrix:
#         # - model.user_factors corresponds to items (shape: n_items x factors)
#         # - model.item_factors corresponds to users (shape: n_users x factors)
        
#         # Get user's latent factors (from item_factors since we transposed)
#         user_factors = self.model.item_factors[user_idx]
        
#         # Compute scores for all items (using user_factors which are item factors)
#         item_factors = self.model.user_factors  # Shape: (n_items, factors)
#         scores = item_factors.dot(user_factors)
        
#         # Set scores of already rated items to -inf
#         for item_idx in user_rated_items:
#             if item_idx < len(scores):
#                 scores[item_idx] = float('-inf')
        
#         # Get top-k items
#         top_indices = np.argsort(-scores)[:top_k]
        
#         # Filter out any -inf items and convert to book IDs
#         recommendations = []
#         for item_idx in top_indices:
#             if scores[item_idx] > float('-inf'):
#                 recommendations.append(self.idx_to_book[item_idx])
        
#         return recommendations


# def generate_recommendations(recommender, user_vectors, user_to_idx, book_to_idx, 
#                             idx_to_user, top_k=100):
#     """Generate recommendations for all users."""
#     results = {}
    
#     print(f"Generating recommendations for {len(user_vectors)} users...")
#     for uid in tqdm(user_vectors.keys()):
#         user_idx = user_to_idx[uid]
        
#         # Get user's rated items as indices
#         user_ratings = user_vectors[uid]
#         user_rated_indices = {book_to_idx[bid] for bid in user_ratings.keys() 
#                              if bid in book_to_idx}
        
#         # For ItemKNN, we need ratings dict
#         if isinstance(recommender, ItemKNNRecommender):
#             ratings_dict = {book_to_idx[bid]: rating 
#                           for bid, rating in user_ratings.items() 
#                           if bid in book_to_idx}
#             recs = recommender.recommend(user_idx, ratings_dict, top_k)
#         else:
#             recs = recommender.recommend(user_idx, user_rated_indices, top_k)
        
#         recs = [int(rec) for rec in recs]
#         results[uid] = recs
    
#     return results

# def generate_recommendations_for_one_user(recommender, user_vectors, uid, user_to_idx, book_to_idx, 
#                             idx_to_user, book_idx_to_title, top_k=100):
#     """Generate recommendations for all users."""
#     results = {}
    
#     print(f"Generating recommendations for user {uid}")
#     user_idx = user_to_idx[uid]
        
#     # Get user's rated items as indices
#     user_ratings = user_vectors[uid]
#     user_rated_indices = {book_to_idx[bid] for bid in user_ratings.keys() 
#                             if bid in book_to_idx}
    
#     # For ItemKNN, we need ratings dict
#     if isinstance(recommender, ItemKNNRecommender):
#         ratings_dict = {book_to_idx[bid]: rating 
#                         for bid, rating in user_ratings.items() 
#                         if bid in book_to_idx}
#         recs = recommender.recommend(user_idx, ratings_dict, top_k)
#     else:
#         recs = recommender.recommend(user_idx, user_rated_indices, top_k)
    
#     recs_with_title = []
#     for rec in recs:
#         book_title = book_idx_to_title[rec]
#         recs_with_title.append([rec, book_title])

#     recs = [int(rec) for rec in recs]

#     results[uid] = recs # recs_with_title
    
#     return results


# def save_results(results, output_path, algorithm_name):
#     """Save recommendations to JSON file."""
#     output = {
#         "algorithm": algorithm_name,
#         "top_k": len(next(iter(results.values()))) if results else 0,
#         "n_users": len(results),
#         "recommendations": results
#     }
    
#     with open(output_path, 'w') as f:
#         json.dump(output, f, indent=2)
    
#     print(f"Saved {len(results)} user recommendations to {output_path}")


# def main():
#     parser = argparse.ArgumentParser(description='Book Recommendation System')
#     parser.add_argument('--input', default='train_user_profiles.json',
#                         help='Path to input JSON file with user ratings')
#     parser.add_argument('--user-id', type=str,
#                         help='Single user ID to get recs for')
#     parser.add_argument('--algorithm', '-a', required=True,
#                        choices=['poprec', 'itemknn', 'userknn', 'mf', 'all'],
#                        help='Recommendation algorithm to use')
#     parser.add_argument('--output', '-o', help='Output JSON file path (default: <algorithm>_recommendations.json)')
#     parser.add_argument('--top-k', '-k', type=int, default=100,
#                        help='Number of recommendations per user (default: 100)')
#     parser.add_argument('--neighbors', '-n', type=int, default=50,
#                        help='Number of neighbors for KNN methods (default: 50)')
#     parser.add_argument('--factors', '-f', type=int, default=100,
#                        help='Number of latent factors for MF (default: 100)')
    
#     args = parser.parse_args()
    
#     # Determine output path
#     script_dir = Path(__file__).parent.parent
#     data_dir = script_dir.parent / 'data'
    
#     # input_path = data_dir / 'clients' / args.input
#     input_path = data_dir / 'eval' / args.input
#     book_title_path = data_dir / 'book_index_to_title.json'

#     # Load data
#     print(f"Loading data from {input_path}...")
#     metadata, user_vectors = load_data(input_path)
#     print(f"Loaded {metadata['n_users']} users, {metadata['n_books']} books")

#     book_index_to_title = load_book_titles(book_title_path)
    
#     # Build matrices
#     print("Building rating matrix...")
#     (rating_matrix, user_to_idx, book_to_idx, 
#      idx_to_user, idx_to_book, user_ids, book_ids) = build_matrices(
#         user_vectors, metadata['n_users'], metadata['n_books']
#     )
#     print(f"Matrix shape: {rating_matrix.shape}, non-zeros: {rating_matrix.nnz}")
    
#     algorithms = []
#     if args.algorithm == 'all':
#         algorithms = ['poprec', 'itemknn', 'userknn', 'mf']
#     else:
#         algorithms = [args.algorithm]
    
#     for algo in algorithms:
#         print(f"\n{'='*50}")
#         print(f"Running {algo.upper()}")
#         print('='*50)
        
#         if algo == 'poprec':
#             recommender = PopRecRecommender(rating_matrix, book_to_idx, idx_to_book)
#             recommender.fit()
#         elif algo == 'itemknn':
#             recommender = ItemKNNRecommender(rating_matrix, book_to_idx, idx_to_book, 
#                                             n_neighbors=args.neighbors)
#             recommender.fit()
#         elif algo == 'userknn':
#             recommender = UserKNNRecommender(rating_matrix, book_to_idx, idx_to_book,
#                                             n_neighbors=args.neighbors)
#             recommender.fit()
#         elif algo == 'mf':
#             os.environ['OPENBLAS_NUM_THREADS'] = '1'
#             recommender = MFRecommender(rating_matrix, book_to_idx, idx_to_book,
#                                        factors=args.factors)
#             recommender.fit()
        
#         if args.user_id:
#             results = generate_recommendations_for_one_user(
#                 recommender, user_vectors, args.user_id, user_to_idx, book_to_idx,
#                 idx_to_user, book_index_to_title, top_k=args.top_k
#             )
#         else:
#             # Generate recommendations
#             results = generate_recommendations(
#                 recommender, user_vectors, user_to_idx, book_to_idx,
#                 idx_to_user, top_k=args.top_k
#             )

#         if args.output and args.algorithm != 'all':
#             output_path = data_dir / 'eval' / 'inputs' / args.output
#         else:
#             if args.user_id:
#                 output_path = data_dir / 'eval' / 'inputs' / f"{algo}_{str(args.user_id)}_recs.json"
#             else:
#                 output_path = data_dir / 'eval' / 'inputs' / f"{algo}_train_recs.json"
        
#         save_results(results, output_path, algo)


# if __name__ == '__main__':
#     main()