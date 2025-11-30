import json
import numpy as np
from collections import defaultdict

# Load data - UPDATE THESE PATHS to match your local file locations
USER_CLUSTERS_PATH = 'active_user_clusters.json'
USER_TRAIN_PATH = 'active_user_embeddings_train.json'

with open(USER_CLUSTERS_PATH, 'r') as f:
    cluster_data = json.load(f)

with open(USER_TRAIN_PATH, 'r') as f:
    user_train_data = json.load(f)

user_to_cluster = cluster_data['user_to_cluster']
user_vectors = user_train_data['user_vectors']

# Print metadata
print("=" * 70)
print("DATASET METADATA")
print("=" * 70)
metadata = user_train_data.get('metadata', {})
for key, value in metadata.items():
    print(f"  {key}: {value}")

# Build cluster -> users mapping and compute ratings per user
cluster_to_users = defaultdict(list)
user_to_num_ratings = {}

for user_id, cluster_id in user_to_cluster.items():
    cluster_to_users[cluster_id].append(user_id)
    if user_id in user_vectors:
        # user_vectors[user_id] is a dict of {book_id: rating}
        user_to_num_ratings[user_id] = len(user_vectors[user_id])
    else:
        user_to_num_ratings[user_id] = 0

# Print cluster sizes
print("\n" + "=" * 70)
print("CLUSTER SIZES")
print("=" * 70)
all_clusters = sorted(cluster_to_users.keys())
for cluster_id in all_clusters:
    print(f"  Cluster {cluster_id}: {len(cluster_to_users[cluster_id])} users")

# Compute statistics per cluster
print("\n" + "=" * 70)
print("CLUSTER STATISTICS: Average Ratings per User")
print("=" * 70)

cluster_stats = []
for cluster_id in sorted(cluster_to_users.keys()):
    users = cluster_to_users[cluster_id]
    num_users = len(users)
    ratings_counts = [user_to_num_ratings.get(u, 0) for u in users]
    
    avg_ratings = np.mean(ratings_counts) if ratings_counts else 0
    median_ratings = np.median(ratings_counts) if ratings_counts else 0
    min_ratings = np.min(ratings_counts) if ratings_counts else 0
    max_ratings = np.max(ratings_counts) if ratings_counts else 0
    std_ratings = np.std(ratings_counts) if ratings_counts else 0
    
    cluster_stats.append({
        'cluster_id': cluster_id,
        'num_users': num_users,
        'avg_ratings': avg_ratings,
        'median_ratings': median_ratings,
        'min_ratings': min_ratings,
        'max_ratings': max_ratings,
        'std_ratings': std_ratings
    })

# Sort by average ratings to see the pattern
cluster_stats_by_avg = sorted(cluster_stats, key=lambda x: x['avg_ratings'])

print("\n" + "=" * 70)
print("CLUSTERS SORTED BY AVERAGE RATINGS (lowest first)")
print("=" * 70)
print(f"{'Cluster':<10} {'Users':<10} {'Avg Ratings':<15} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 70)

for stats in cluster_stats_by_avg[:20]:  # Show 20 lowest
    print(f"{stats['cluster_id']:<10} {stats['num_users']:<10} {stats['avg_ratings']:<15.2f} {stats['median_ratings']:<10.1f} {stats['std_ratings']:<10.1f} {stats['min_ratings']:<10} {stats['max_ratings']:<10}")

print("\n... (showing 20 lowest) ...\n")

print("=" * 70)
print("CLUSTERS WITH HIGHEST AVERAGE RATINGS")
print("=" * 70)
print(f"{'Cluster':<10} {'Users':<10} {'Avg Ratings':<15} {'Median':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
print("-" * 70)

for stats in cluster_stats_by_avg[-20:]:  # Show 20 highest
    print(f"{stats['cluster_id']:<10} {stats['num_users']:<10} {stats['avg_ratings']:<15.2f} {stats['median_ratings']:<10.1f} {stats['std_ratings']:<10.1f} {stats['min_ratings']:<10} {stats['max_ratings']:<10}")

# Specific comparison: Cluster 1 vs Cluster 24
print("\n" + "=" * 70)
print("SPECIFIC COMPARISON: Cluster 1 vs Cluster 24")
print("=" * 70)

for cluster_id in [1, 24]:
    users = cluster_to_users[cluster_id]
    ratings_counts = [user_to_num_ratings.get(u, 0) for u in users]
    print(f"\nCluster {cluster_id}:")
    print(f"  Number of users: {len(users)}")
    print(f"  Avg ratings per user: {np.mean(ratings_counts):.2f}")
    print(f"  Median ratings per user: {np.median(ratings_counts):.1f}")
    print(f"  Std ratings per user: {np.std(ratings_counts):.1f}")
    print(f"  Min ratings: {np.min(ratings_counts)}")
    print(f"  Max ratings: {np.max(ratings_counts)}")
    print(f"  Users with <=5 ratings: {sum(1 for r in ratings_counts if r <= 5)} ({100*sum(1 for r in ratings_counts if r <= 5)/len(ratings_counts):.1f}%)")
    print(f"  Users with <=10 ratings: {sum(1 for r in ratings_counts if r <= 10)} ({100*sum(1 for r in ratings_counts if r <= 10)/len(ratings_counts):.1f}%)")
    print(f"  Users with >20 ratings: {sum(1 for r in ratings_counts if r > 20)} ({100*sum(1 for r in ratings_counts if r > 20)/len(ratings_counts):.1f}%)")

# Overall statistics
print("\n" + "=" * 70)
print("OVERALL STATISTICS")
print("=" * 70)

all_ratings = list(user_to_num_ratings.values())
print(f"Total users: {len(all_ratings)}")
print(f"Overall avg ratings per user: {np.mean(all_ratings):.2f}")
print(f"Overall median ratings per user: {np.median(all_ratings):.1f}")
print(f"Users with <=5 ratings: {sum(1 for r in all_ratings if r <= 5)} ({100*sum(1 for r in all_ratings if r <= 5)/len(all_ratings):.1f}%)")
print(f"Users with <=10 ratings: {sum(1 for r in all_ratings if r <= 10)} ({100*sum(1 for r in all_ratings if r <= 10)/len(all_ratings):.1f}%)")
print(f"Users with >20 ratings: {sum(1 for r in all_ratings if r > 20)} ({100*sum(1 for r in all_ratings if r > 20)/len(all_ratings):.1f}%)")

# Correlation between cluster size and average ratings
print("\n" + "=" * 70)
print("CORRELATION: Cluster Size vs Average Ratings")
print("=" * 70)

sizes = [s['num_users'] for s in cluster_stats]
avgs = [s['avg_ratings'] for s in cluster_stats]
correlation = np.corrcoef(sizes, avgs)[0, 1]
print(f"Correlation coefficient: {correlation:.3f}")
if correlation < -0.3:
    print("-> Negative correlation: Larger clusters tend to have lower average ratings (sparse users)")
elif correlation > 0.3:
    print("-> Positive correlation: Larger clusters tend to have higher average ratings")
else:
    print("-> Weak correlation between cluster size and average ratings")