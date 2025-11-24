#!/usr/bin/env python3
"""
Create sparse user profile vectors from Goodreads rating data.
Each user vector has dimensions equal to the number of books, with ratings (1-5) or 0.
"""
# made with claude !!
# first makes a json file for book_id -> book_index

# instead of storing a bunch of super sparse vectors (of length like 124k),
# this stores user vectors as dictionaries with book_index -> rating if the book was rated (sparse dictionary format)
# also creates a unified book_id → index mapping



import json
import numpy as np
from scipy.sparse import lil_matrix, save_npz
from collections import defaultdict
import argparse
from pathlib import Path


def load_json_file(filepath):
    """Load a JSON file and return the data."""
    print(f"Loading {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} users")
    return data


def get_all_book_ids(user_map, user_train, user_test):
    """Extract all unique book IDs from all datasets."""
    print("\nExtracting all unique book IDs...")
    book_ids = set()
    
    for user_ratings in [user_map, user_train, user_test]:
        for user_id, ratings in user_ratings.items():
            for rating_entry in ratings:
                book_ids.add(rating_entry['book_id'])
    
    # Sort to ensure consistent ordering
    book_ids = sorted(book_ids)
    print(f"  Found {len(book_ids)} unique books")
    return book_ids


def create_book_id_to_index(book_ids):
    """Create a mapping from book_id to vector index."""
    return {book_id: idx for idx, book_id in enumerate(book_ids)}


def create_user_vectors_sparse(user_data, book_id_to_index):
    """
    Create sparse user profile vectors.
    Returns a dictionary mapping user_id to scipy sparse matrix (1 x n_books).
    """
    n_books = len(book_id_to_index)
    user_vectors = {}
    
    print(f"Creating vectors for {len(user_data)} users...")
    for user_id, ratings in user_data.items():
        # Create a sparse row vector for this user
        vector = lil_matrix((1, n_books), dtype=np.float32)
        
        for rating_entry in ratings:
            book_id = rating_entry['book_id']
            rating = rating_entry['rating']
            
            if book_id in book_id_to_index:
                idx = book_id_to_index[book_id]
                vector[0, idx] = rating
        
        # Convert to CSR format for efficient storage and operations
        user_vectors[user_id] = vector.tocsr()
    
    return user_vectors


def create_user_vectors_dict(user_data, book_id_to_index):
    """
    Create user profile vectors as dictionaries (only storing non-zero entries).
    More compact for very sparse data.
    """
    user_vectors = {}
    
    print(f"Creating vectors for {len(user_data)} users...")
    for user_id, ratings in user_data.items():
        # Only store non-zero entries
        vector_dict = {}
        
        for rating_entry in ratings:
            book_id = rating_entry['book_id']
            rating = rating_entry['rating']
            
            if book_id in book_id_to_index:
                idx = book_id_to_index[book_id]
                vector_dict[idx] = rating
        
        user_vectors[user_id] = vector_dict
    
    return user_vectors


def save_embeddings_json(user_vectors, n_books, output_file):
    """Save user vectors and metadata to JSON file."""
    print(f"\nSaving to {output_file}...")
    
    output_data = {
        'metadata': {
            'n_users': len(user_vectors),
            'n_books': n_books,
            'vector_format': 'sparse_dict'
        },
        'user_vectors': user_vectors
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"  Saved {len(user_vectors)} user vectors")


def save_book_index_mapping(book_id_to_index, output_file):
    """Save book_id to index mapping to a separate JSON file."""
    print(f"\nSaving book index mapping to {output_file}...")
    
    mapping_data = {
        'metadata': {
            'n_books': len(book_id_to_index)
        },
        'book_id_to_index': book_id_to_index
    }
    
    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"  Saved mapping for {len(book_id_to_index)} books")


def calculate_sparsity(user_vectors, n_books):
    """Calculate average sparsity of user vectors."""
    total_ratings = sum(len(v) for v in user_vectors.values())
    possible_ratings = len(user_vectors) * n_books
    sparsity = 1 - (total_ratings / possible_ratings)
    return sparsity, total_ratings


def load_book_index_mapping(filepath):
    """Load book_id to index mapping from JSON file."""
    print(f"Loading book index mapping from {filepath}...")
    with open(filepath, 'r') as f:
        data = json.load(f)
    book_id_to_index = data['book_id_to_index']
    print(f"  Loaded mapping for {len(book_id_to_index)} books")
    return book_id_to_index


def main():
    parser = argparse.ArgumentParser(
        description='Create user profile vectors from Goodreads rating data'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['books', 'map', 'train', 'test', 'all'],
                        help='Which dataset to create embeddings for: books (create book index), map, train, test, or all')
    
    args = parser.parse_args()
    
    # Define file paths relative to script location
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    user_map_path = data_dir / 'user_map.json'
    user_train_path = data_dir / 'user_train.json'
    user_test_path = data_dir / 'user_test.json'
    
    output_map_path = data_dir / 'user_embeddings.json'
    output_train_path = data_dir / 'user_embeddings_train.json'
    output_test_path = data_dir / 'user_embeddings_test.json'
    book_index_path = data_dir / 'book_id_to_index.json'
    
    # Handle 'books' option - just create the book index mapping and exit
    if args.dataset == 'books':
        print("="*50)
        print("Creating book index mapping...")
        print("="*50)
        
        # Load all data to get consistent book indexing
        user_map = load_json_file(user_map_path)
        user_train = load_json_file(user_train_path)
        user_test = load_json_file(user_test_path)
        
        # Get all unique book IDs and create index mapping
        book_ids = get_all_book_ids(user_map, user_train, user_test)
        book_id_to_index = create_book_id_to_index(book_ids)
        
        # Save book index mapping
        save_book_index_mapping(book_id_to_index, book_index_path)
        
        print("\n" + "="*50)
        print("DONE!")
        print("="*50)
        print(f"Book index mapping saved to: {book_index_path}")
        return
    
    # For all other options, load the existing book index mapping
    if not book_index_path.exists():
        print(f"ERROR: Book index mapping not found at {book_index_path}")
        print("Please run with --dataset books first to create the book index mapping")
        return
    
    book_id_to_index = load_book_index_mapping(book_index_path)
    n_books = len(book_id_to_index)
    
    # Determine which datasets to process and load only necessary data
    datasets_to_process = []
    if args.dataset == 'all':
        user_map = load_json_file(user_map_path)
        user_train = load_json_file(user_train_path)
        user_test = load_json_file(user_test_path)
        datasets_to_process = [
            ('map', user_map, output_map_path),
            ('train', user_train, output_train_path),
            ('test', user_test, output_test_path)
        ]
    elif args.dataset == 'map':
        user_map = load_json_file(user_map_path)
        datasets_to_process = [('map', user_map, output_map_path)]
    elif args.dataset == 'train':
        user_train = load_json_file(user_train_path)
        datasets_to_process = [('train', user_train, output_train_path)]
    elif args.dataset == 'test':
        user_test = load_json_file(user_test_path)
        datasets_to_process = [('test', user_test, output_test_path)]
    
    # Process selected datasets
    print("\n" + "="*50)
    print("STATISTICS")
    print("="*50)
    
    output_files = []
    for name, user_data, output_path in datasets_to_process:
        print("\n" + "="*50)
        print(f"Creating user vectors for USER_{name.upper()}...")
        print("="*50)
        user_vectors = create_user_vectors_dict(user_data, book_id_to_index)
        
        # Calculate sparsity
        sparsity, total_ratings = calculate_sparsity(user_vectors, n_books)
        print(f"\n{name} statistics:")
        print(f"  Users: {len(user_vectors)}")
        print(f"  Total ratings: {total_ratings}")
        print(f"  Avg ratings per user: {total_ratings / len(user_vectors):.2f}")
        print(f"  Sparsity: {sparsity*100:.4f}%")
        
        # Save to JSON
        save_embeddings_json(user_vectors, n_books, output_path)
        output_files.append(output_path)
    
    print("\n" + "="*50)
    print("DONE!")
    print("="*50)
    print(f"\nCreated {n_books}-dimensional user vectors")
    print(f"User embeddings saved to:")
    for output_file in output_files:
        print(f"  - {output_file}")


if __name__ == '__main__':
    main()