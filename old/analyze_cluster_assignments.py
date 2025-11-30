#!/usr/bin/env python3
import json
from collections import Counter

def analyze_cluster_distribution(filepath="recommendations.json"):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract cluster assignments
    cluster_ids = [user_data["cluster_id"] for user_data in data["user_recommendations"].values()]
    
    # Count occurrences
    cluster_counts = Counter(cluster_ids)
    
    # All possible clusters (0-165)
    all_clusters = set(range(166))
    used_clusters = set(cluster_counts.keys())
    empty_clusters = sorted(all_clusters - used_clusters)
    
    # Basic stats
    n_users = len(cluster_ids)
    n_clusters = len(cluster_counts)
    
    print(f"Total users: {n_users}")
    print(f"Total possible clusters: 166 (0-165)")
    print(f"Clusters with users: {n_clusters}")
    print(f"Empty clusters: {len(empty_clusters)}")
    print()
    
    # Most popular clusters
    print("Top 10 most popular clusters:")
    for rank, (cluster_id, count) in enumerate(cluster_counts.most_common(10), 1):
        pct = count / n_users * 100
        print(f"  {rank}. Cluster {cluster_id}: {count} users ({pct:.1f}%)")
    
    print()
    
    # Least popular clusters
    print("10 least popular clusters:")
    for cluster_id, count in cluster_counts.most_common()[-10:]:
        pct = count / n_users * 100
        print(f"  Cluster {cluster_id}: {count} users ({pct:.1f}%)")
    
    print()
    
    # Distribution stats
    counts = list(cluster_counts.values())
    avg_users = n_users / n_clusters
    max_users = max(counts)
    min_users = min(counts)
    
    print("Distribution stats:")
    print(f"  Avg users per cluster: {avg_users:.1f}")
    print(f"  Max users in a cluster: {max_users}")
    print(f"  Min users in a cluster: {min_users}")
    
    print()
    
    # Empty clusters
    if empty_clusters:
        print(f"Empty clusters ({len(empty_clusters)}):")
        # Print in rows of 10 for readability
        for i in range(0, len(empty_clusters), 10):
            row = empty_clusters[i:i+10]
            print(f"  {', '.join(map(str, row))}")
    else:
        print("No empty clusters - all 166 clusters have at least one user.")

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "recommendations.json"
    analyze_cluster_distribution(filepath)