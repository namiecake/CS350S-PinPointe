#!/usr/bin/env python3
import json
import random
from collections import defaultdict
from pathlib import Path

def load_json(filepath):
    """Load a JSON file."""
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'

    datapath = data_dir / filepath
    with open(datapath, 'r') as f:
        return json.load(f)

def load_jsonl(filepath):
    """Load a JSONL file (one JSON object per line)."""
    books = {}
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / 'data'
    
    datapath = data_dir / filepath
    with open(datapath, 'r') as f:
        for line in f:
            book = json.loads(line.strip())
            books[str(book['book_id'])] = book['title']
    return books

def get_user_top_books(user_ratings, n=2):
    """Get the top N highest rated books for a user."""
    sorted_ratings = sorted(user_ratings, key=lambda x: x['rating'], reverse=True)
    return sorted_ratings[:n]

def main():
    # Get cluster ID from user
    cluster_id = input("Enter cluster ID: ").strip()
    
    print(f"\n{'='*80}")
    print(f"CLUSTER {cluster_id} SANITY CHECK")
    print(f"{'='*80}\n")
    
    # Load all data files
    print("Loading data files...")
    books = load_jsonl('goodreads_books_children.json')
    book_id_to_index = load_json('book_id_to_index.json')['book_id_to_index']
    book_index_to_id = {v: k for k, v in book_id_to_index.items()}
    cluster_recs = load_json('server/cluster_recs.json')['cluster_recommendations']
    user_to_cluster = load_json('server/user_clusters_cosine.json')['user_to_cluster']
    user_train = load_json('server/user_train.json')
    
    # Check if cluster exists
    if cluster_id not in cluster_recs:
        print(f"Error: Cluster {cluster_id} not found in cluster_recs.json")
        return
    
    # Get users in this cluster
    users_in_cluster = [user_id for user_id, cid in user_to_cluster.items() 
                        if str(cid) == cluster_id]
    
    if not users_in_cluster:
        print(f"Error: No users found in cluster {cluster_id}")
        return
    
    print(f"Found {len(users_in_cluster)} users in cluster {cluster_id}")
    
    # Randomly select up to 10 users
    selected_users = random.sample(users_in_cluster, min(10, len(users_in_cluster)))
    
    print(f"\n{'='*80}")
    print(f"TOP 2 HIGHEST RATED BOOKS FOR {len(selected_users)} RANDOM USERS IN CLUSTER {cluster_id}")
    print(f"{'='*80}\n")
    
    # Display top books for each selected user
    for i, user_id in enumerate(selected_users, 1):
        if user_id not in user_train:
            print(f"{i}. User {user_id[:8]}... - No ratings found")
            continue
        
        user_ratings = user_train[user_id]
        top_books = get_user_top_books(user_ratings, n=2)
        
        print(f"{i}. User {user_id[:8]}... (rated {len(user_ratings)} books)")
        for j, book_rating in enumerate(top_books, 1):
            book_id = book_rating['book_id']
            rating = book_rating['rating']
            title = books.get(book_id, f"Unknown (ID: {book_id})")
            print(f"   {j}. [{rating}★] {title}")
        print()
    
    # Display cluster recommendations
    print(f"{'='*80}")
    print(f"TOP 20 RECOMMENDATIONS FOR CLUSTER {cluster_id}")
    print(f"{'='*80}\n")
    
    recs = cluster_recs[cluster_id][:20]
    
    for i, book_index in enumerate(recs, 1):
        book_id_str = str(book_index_to_id[book_index])
        title = books.get(book_id_str, f"Unknown (ID: {book_id_str})")
        print(f"{i:2d}. {title}")
    
    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()