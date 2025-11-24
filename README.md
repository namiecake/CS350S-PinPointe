# CS350S-PinPointe

hi ok so i tried minibatchkmeans clustering which uses euclidean distance but the silhouette score (i.e. measure of clustering algorithm performance) was reallyyy bad, probably because the vectors are so sparse. the results are still there in cluster_embeddings_euclidean.py and user_clusters_euclidean.json but basically just ignore these lol.

also used truncatedSVD for dimensionality reduction to dimension 300 (instead of 124k) but the parameter of 300 can maybe be toyed with/improved.

so instead i use spherical kmeans clustering (i.e. kmeans on l2 normalized data) which uses cosine similarity because, as claude states:
- Cosine similarity is often more meaningful for collaborative filtering
- Works better with sparse matrices
- Better captures user preference patterns

results are in user_clusters_cosine.json! so tldr we have SOME CLUSTERS to experiment with! i'll first move onto the next step of getting recs per cluster.

TODO:
- BALANCE CLUSTERS - currently the largest cluster is 25% of the dataset! (7000 reviews). the smallest is 1 user. the tiptoe paper says that " To obtain
roughly balanced clusters, we recursively split large clusters into multiple smaller ones". I'm working on doing this too, and also merging super small clusters with other clusters nearby. currently thinking max cluster size = 400 and min cluster size = 40. I don't think it's super important to have like equal size clusters but at least maybe split up that huge cluster.
- anyway so i'm thinking we can run the experiment on both "not recursively splitting to balance" and with "balancing" and observe which one works better". because maybe like 25% of children just have the same mainstream tastes ya know.
- MULTI-CLUSTER ASSIGNMENTS - tiptoe also says: "A common technique to increase search quality in clusterbased nearest-neighbor-search is to assign a single document to multiple clusters [26,64]. Following prior work [26], Tiptoe assigns documents to multiple clusters if they are close to cluster boundaries. In particular, Tiptoe assigns 20% of the documents to two clusters and the remaining 80% only to a single cluster, resulting in a roughly 1.2× overhead in server computation and √ 1.2× overhead in communication. We show in §8 that this optimization improves search quality."
- so, we should also try multi-clustering, i.e. assign each user to the top 3 clusters.



