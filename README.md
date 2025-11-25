# CS350S-PinPointe

updates:

if changing the dataset, run:
- create_embeddings
- cluster_embeddings_cosine
- move the saved svd_model.pkl to data/server
- generate_cluster_recs

eval:
- filter_uses.py (get active users only)
- process_test_users (test/train split per user profile)
- query_for_recs on the train set of the evaluated users
- baseline_recommenders - all on the train set of the evaluated users
- evaluate_recall (not done yet)

COMPLETED
- embeddings
- clustering
- getting recs per cluster (used implicit library instead of lightFM bc lightFM is not compatible with python ver>12.0 and implicit uses a more straightforward algo)
- getting recs per user NON PRIVATELY
- filtered books dataset. lots of duplicate book titles listed under multiple book IDs so I just combined them all so we have 96k books now. lowkey there's still duplicate books e.g. "wonder (wonder #1)" vs. "wonder" but it's ok. i reran process_dataset.py on the new data and everything so recommend pulling from the repo!!!
- filtered test users to only have users with > 20 ratings for accurate evaluation
- script to run baseline rec algos (recommend most popular, itemknn, userknn, implicit library on all data) for evaluation

CURRENTLY WORKING ON:
- evaluation to make sure we actually have a good model lol
- evaluation method: take each user profile and make a train/test split out of it (e.g. for user with 20 ratings, 16 books are part of their profile, and we expect the rec algo to recommend the other 4). then see how well poprec, itemknn, userknn, implicit, and pinpointe all do.

STILL TODO:
- experiments (clustering, train/test split)
- simplePIR retrieval i.e. the whole point lol - should do after we have our best performing model
- local item re-ranking

**random notes on clustering:**
ok for clustering used truncatedSVD for dimensionality reduction to dimension 300 (instead of 124k) but the parameter of 300 can maybe be toyed with/improved.

i use spherical kmeans clustering (i.e. kmeans on l2 normalized data) which uses cosine similarity because, as claude states:
- Cosine similarity is often more meaningful for collaborative filtering
- Works better with sparse matrices
- Better captures user preference patterns

TODO:
- BALANCE CLUSTERS - currently the largest cluster is 25% of the dataset! (7000 reviews). the smallest is 1 user. the tiptoe paper says that " To obtain
roughly balanced clusters, we recursively split large clusters into multiple smaller ones". I'm working on doing this too, and also merging super small clusters with other clusters nearby. currently thinking max cluster size = 400 and min cluster size = 40. I don't think it's super important to have like equal size clusters but at least maybe split up that huge cluster.
- anyway so i'm thinking we can run the experiment on both "not recursively splitting to balance" and with "balancing" and observe which one works better". because maybe like 25% of children just have the same mainstream tastes ya know.
- MULTI-CLUSTER ASSIGNMENTS - tiptoe also says: "A common technique to increase search quality in clusterbased nearest-neighbor-search is to assign a single document to multiple clusters [26,64]. Following prior work [26], Tiptoe assigns documents to multiple clusters if they are close to cluster boundaries. In particular, Tiptoe assigns 20% of the documents to two clusters and the remaining 80% only to a single cluster, resulting in a roughly 1.2× overhead in server computation and √ 1.2× overhead in communication. We show in §8 that this optimization improves search quality."
- so, we should also try multi-clustering, i.e. assign each user to the top 3 clusters.










