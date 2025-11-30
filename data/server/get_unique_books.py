#!/usr/bin/env python3
"""
Cluster user profile embeddings using K-Means with cosine similarity.
Uses sqrt(n_users) clusters and can optionally assign users to multiple clusters.
This version uses standard scikit-learn without spherecluster dependency.

literally the same as the other script but saves the svd model
MODIFIED: Also saves the TruncatedSVD model for query-time transformations.
"""

import json
import pickle
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from pathlib import Path
import argparse
from collections import defaultdict

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

def main():
    # Load embeddings
    user_vectors, n_users, n_books = load_embeddings('user_embeddings_train.json')
    user_vectorstrainprofiles, n_users, n_books = load_embeddings('../eval/train_user_profiles.json')
    user_vectorstestprofiles, n_users, n_books = load_embeddings('../eval/test_user_profiles.json')

    trainbooks = set()
    testbooks = set()
    trainprofilesbooks = set()
    testprofilesbooks = set()
    
    for user in list(user_vectors.keys()):
        for book in user_vectors[user].keys():
            book = int(book)
            trainbooks.add(book)

    for user in list(user_vectorstrainprofiles.keys()):
        for book in user_vectorstrainprofiles[user].keys():
            book = int(book)
            trainprofilesbooks.add(book)

    for user in list(user_vectorstestprofiles.keys()):
        for book in user_vectorstestprofiles[user].keys():
            book = int(book)
            testprofilesbooks.add(book)

    testbooks = set()
    for book in trainprofilesbooks:
        testbooks.add(book)

    for book in testprofilesbooks:
        testbooks.add(book)
    print("number of test books is: ", len(testbooks))

    allbooks = testbooks
    for book in trainbooks:
        allbooks.add(book)

    print("total number of books is: ", len(allbooks))

    for book in trainprofilesbooks:
        testbooks.remove(book)
    remainder = testbooks - trainbooks
    print("the number of test books not in train books is: ", len(remainder))

    print("number of test books train profile is: ", len(trainprofilesbooks))

    print("number of test books test profile is: ", len(testprofilesbooks))

    print("the number of train books is: ", len(trainbooks))


if __name__ == '__main__':
    main()
