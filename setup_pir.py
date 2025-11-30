"""
One-time setup script for PIR in PinPointe.
This prepares the PIR server and saves it for future use.

File: setup_pir.py

Run this ONCE after generating your cluster recommendations:
    python setup_pir.py

This will create:
    - data/pir_server.npz (the PIR database)
    - data/pir_params.json (parameters for the client)
"""

import json
import numpy as np
import os
from src.pir.pir_scheme import setup_pir_database, save_pir_server, PIRClient


def load_cluster_recommendations(recs_path):
    """Load pre-computed recommendations per cluster."""
    print(f"Loading cluster recommendations from {recs_path}...")
    with open(recs_path, 'r') as f:
        data = json.load(f)
    # Convert string keys to integers
    cluster_recs = {int(k): v for k, v in data["cluster_recommendations"].items()}
    print(f"  Loaded {len(cluster_recs)} clusters")
    return cluster_recs


def determine_rec_list_size(cluster_recs):
    """Determine the maximum recommendation list size."""
    max_size = max(len(recs) for recs in cluster_recs.values())
    avg_size = sum(len(recs) for recs in cluster_recs.values()) / len(cluster_recs)
    
    print(f"\nRecommendation list sizes:")
    print(f"  Maximum: {max_size}")
    print(f"  Average: {avg_size:.1f}")
    
    # Use next power of 2 above average, or 100, whichever is larger
    rec_list_size = max(100, 2 ** int(np.ceil(np.log2(avg_size))))
    print(f"  Using fixed size: {rec_list_size}")
    
    return rec_list_size


def main():
    """Main setup function."""
    print("="*70)
    print("PIR Setup for PinPointe")
    print("="*70)
    
    # Paths - adjust these if your files are in different locations
    recs_path = 'data/server/recs_per_cluster.json'
    pir_server_path = 'data/pir_server.npz'
    pir_params_path = 'data/pir_params.json'
    
    # Check if files exist
    if not os.path.exists(recs_path):
        print(f"\n❌ Error: Could not find {recs_path}")
        print("Please run generate_cluster_recs.py first!")
        return
    
    # Load cluster recommendations
    cluster_recs = load_cluster_recommendations(recs_path)
    num_clusters = len(cluster_recs)
    
    # Determine recommendation list size
    rec_list_size = determine_rec_list_size(cluster_recs)
    
    # Setup PIR database
    print(f"\nSetting up PIR database...")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Recommendations per cluster: {rec_list_size}")
    
    pir_server, pir_params = setup_pir_database(
        cluster_recs, 
        num_clusters, 
        rec_list_size
    )
    
    # Save PIR server
    print(f"\nSaving PIR server to {pir_server_path}...")
    save_pir_server(pir_server, pir_server_path)
    
    # Save PIR parameters
    print(f"Saving PIR parameters to {pir_params_path}...")
    with open(pir_params_path, 'w') as f:
        json.dump(pir_params, f, indent=2)
    
    # Calculate sizes
    server_size_mb = os.path.getsize(pir_server_path) / (1024 * 1024)
    params_size_kb = os.path.getsize(pir_params_path) / 1024
    
    print("\n" + "="*70)
    print("✓ PIR Setup Complete!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {pir_server_path} ({server_size_mb:.2f} MB)")
    print(f"     - This is the PIR database (server-side)")
    print(f"  2. {pir_params_path} ({params_size_kb:.2f} KB)")
    print(f"     - These are public parameters (client needs these)")
    
    print(f"\nDatabase statistics:")
    print(f"  - Total clusters: {num_clusters}")
    print(f"  - Recommendations per cluster: {rec_list_size}")
    print(f"  - Total items in database: {num_clusters * rec_list_size:,}")
    
    # Test that client can initialize
    print(f"\nTesting client initialization...")
    try:
        pir_client = PIRClient(pir_params)
        print("✓ Client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing client: {e}")
        return
    
    # Quick test
    print(f"\nRunning quick test...")
    from src.pir.pir_scheme import retrieve_recommendations_with_pir
    
    test_cluster = 0
    retrieved = retrieve_recommendations_with_pir(
        test_cluster, pir_client, pir_server
    )
    expected = cluster_recs[test_cluster][:rec_list_size]
    
    # Compare (allowing for padding zeros)
    retrieved_no_pad = [r for r in retrieved if r != 0]
    expected_no_pad = [e for e in expected if e != 0]
    
    if retrieved_no_pad == expected_no_pad:
        print(f"✓ Test passed! Retrieved {len(retrieved_no_pad)} recommendations correctly")
    else:
        print(f"❌ Test failed!")
        print(f"  Expected: {expected_no_pad[:5]}...")
        print(f"  Got: {retrieved_no_pad[:5]}...")


if __name__ == "__main__":
    main()