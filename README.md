# CS350S-PinPointe

## jank but accurate setup guide: ##

### getting the right data:
- go to util and run `book_stats_dedup.py` -- this combines books with the exact same title into one ID. then take the two output files (ending in `_dedup`), rename them to take out the `_dedup`, and delete the old goodreads json files. use these ones instead!
 
### getting `book_id_to_index.json`:
 it's best to just pull the one from the github, that one has all books in the full dataset including books that were filtered out of ratings. to reproduce this version:
 - go to `process_dataset.py` and comment out the if-statement on line 27 (but still run the code in the if-statement), then run
 - run `create_embeddings.py --dataset books`, which gives you `book_id_to_index.json`
 - go back to `process_dataset.py` and restore the if-statement and run this again, then run `create_embeddings.py --dataset all`

### preparing for eval:
we evaluate by taking the active users in the test set (20+ ratings, with at least 10 being 4-5 stars), and splitting each user profile into a train/test set
- run `filter_users.py` _path-to-user_embeddings_test.json path-to-new-file-active_users_embeddings_test.json_ --min-total 20 --min-relevant 10
- put active_users_embeddings_test.json in data/server btw
- run `process_test_users.py` to get the test/train split per user profile -> `train_user_profiles.json` and `test_user_profiles.json`. only takes users who have at least five 4-5 star ratings in their test set

### getting results for the baseline algorithms (poprec, itemknn, userknn, mf):
can just see the numbers in the table below - these stay the same no matter what, but here are steps to reproduce:
- prereq: make sure you have generated the `train_user_profiles.json` file
- run `get_training_data_for_baselines.py` from util to do exactly that
- run `baseline_recommenders.py --algorithm all` -- note: itemknn and userknn take a while! each outputs a file called {algo}_train_recs.json
- run `evaluate_recall.py`

### getting recommendations and evaluating pinpointe:
*NOTE: was a small bug in filter users, so may want to rerun the `create_embeddings.py --dataset all` before running the filter users below
- `filter_users.py path-to-user_embeddings_train path-to-user_embeddings_train --min-total 5 --min-relevant 0` - performs better when we only train with users with 5+ ratings, other users are just noise
- `cluster_embeddings.py` -> `user_clusters.json`, `svd_model.pkl` (saved SVD model)
- `generate_cluster_recs.py` -> `recs_per_cluster.json`
- `setup_pir.py` (sets up the PIR server)
- `query_for_recs.py` -> `pinpointe_train_recs.json` (this automatically runs PIR by default, to test with no pir use --no-pir flag)
-  run `evaluate_recall.py`

yay!

### STILL TODO:
- simplePIR retrieval i.e. the whole point lol
- local item re-ranking
- evaluating NDCG metric (do it after item re-ranking is done)
- experiments (train/test split threshold)
- writeup!

--------
## older updates: ##

#### older setup guide: 

if you're getting 'file not found' errors, just update the paths lol
btw i did change the dataset. so rerun the process_dataset and everything lol

if changing the dataset, run:
- `process_dataset.py` -> `user_map.json`, `user_train.json`, `user_test.json
- `create_embeddings.py` -> `user_embeddings.json`, `user_embeddings_train.json`, `user_embeddings_test.json`
- `filter_users.py path-to-user_embeddings_train path-to-user_embeddings_train --min-total 5 --min-relevant 0` - performs better when we only train with users with 5+ ratings, other users are just noise
- `cluster_embeddings.py` -> `user_clusters.json`, `svd_model.pkl` (saved SVD model)
- `generate_cluster_recs.py` -> `recs_per_cluster.json`


if pulling from this repo, run `cluster_embeddings.py` and `generate_cluster_recs.py` to get the saved SVD model since it's too large to add to the repo


for eval (specify the input and output files when running)
- `filter_users.py` (get active users only)
- `process_test_users.py` (test/train split per user profile)
- **`query_for_recs.py`** (gets recommendations for the train set of the evaluated users): `python query_for_recs.py --embeddings-file train_user_profiles.json`
- `get_training_data_for_baselines.py`
- `baseline_recommenders.py` - (gets recommendations for the train set using traditional algorithms: poprec, itemknn, userknn, and matrix factorization)
- `evaluate_recall.py`

### preliminary results!
| Algorithm | Users | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|----------|-------|-----------|-----------|-----------|------------|
| mf       | 2485  | 0.0552    | 0.0881    | 0.1474    | 0.2069     |
| **pinpointe**| 2485  | 0.0303    | 0.0497    | 0.0888    | 0.1303     |
| userknn  | 2485  | 0.0188    | 0.0339    | 0.0659    | 0.0871     |
| itemknn  | 2485  | 0.0072    | 0.0121    | 0.0207    | 0.0347     |
| poprec   | 2485  | 0.0048    | 0.0065    | 0.0103    | 0.0216     |

(actual numbers don't matter - it's just about how well it performs relative to the baseline algos, especially matrix factorization)

### COMPLETED
- embeddings
- clustering
- getting recs per cluster (used implicit library instead of lightFM bc lightFM is not compatible with python ver>12.0 and implicit uses a more straightforward algo)
- getting recs per user NON PRIVATELY
- filtered books dataset. lots of duplicate book titles listed under multiple book IDs so I just combined them all so we have 96k books now. lowkey there's still duplicate books e.g. "wonder (wonder #1)" vs. "wonder" but it's ok. i reran process_dataset.py on the new data and everything so recommend pulling from the repo!!!
- filtered test users to only have users with > 20 ratings for accurate evaluation
- script to run baseline rec algos (recommend most popular, itemknn, userknn, implicit library on all data) for evaluation
- MULTI-CLUSTER ASSIGNMENTS - tiptoe also says: "A common technique to increase search quality in clusterbased nearest-neighbor-search is to assign a single document to multiple clusters [26,64]. Following prior work [26], Tiptoe assigns documents to multiple clusters if they are close to cluster boundaries. In particular, Tiptoe assigns 20% of the documents to two clusters and the remaining 80% only to a single cluster, resulting in a roughly 1.2× overhead in server computation and √ 1.2× overhead in communication. We show in §8 that this optimization improves search quality."
- evaluating recall
  evaluation method: take each user profile and make a train/test split out of it (e.g. for user with 20 ratings, 16 books are part of their profile, and we expect the rec algo to recommend the other 4). then see how well poprec, itemknn, userknn, implicit, and pinpointe all do.
- clustering experiments (almosttt done this was super ad hoc)


**random notes on clustering:**
ok for clustering used truncatedSVD for dimensionality reduction to dimension 300 (instead of 124k) but the parameter of 300 can maybe be toyed with/improved.

i use spherical kmeans clustering (i.e. kmeans on l2 normalized data) which uses cosine similarity because, as claude states:
- Cosine similarity is often more meaningful for collaborative filtering
- Works better with sparse matrices
- Better captures user preference patterns

older TODO:
- BALANCE CLUSTERS - the tiptoe paper says that " To obtain roughly balanced clusters, we recursively split large clusters into multiple smaller ones".  I don't think it's super important to have like equal size clusters but at least maybe split up that huge cluster.
- anyway so i'm thinking we can run the experiment on both "not recursively splitting to balance" and with "balancing" and observe which one works better.



























