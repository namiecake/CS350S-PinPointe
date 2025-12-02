# CS350S-PinPointe

 ## RUNNING EXPERIMENTS
 
 results here!: https://docs.google.com/spreadsheets/d/1VuNzrv8lEP68u9SJCrAJFcX423gDiUkjQUrhptcLOLU/edit?usp=sharing
 
 best experiment for 30/70 split: n_components 300, assign users to 5 clusters each, generate recs proportionally, retrieve 5 nearest clusters proportionally
 
best results:
pinpointe       |    2485 |     0.0323 |     0.0523 |     0.0953 |     0.1458
 
**To run experiments:**
(i'm done running them tho LOL)

Step 1: Cluster
> **python3 cluster_embeddings.py --multicluster [# clusters to assign each user to]**
> 
> optional: add  --n-components [int, default 100] if you want to change the number of components (not crazy important lol)

Step 2: Generate cluster recommendations (uses the new clusters)
> **python3 generate_cluster_recs.py**
> 
> optional: add --equal-weights if you want each user to contribute recs equally to the cluster. otherwise they will contribute based priority/cluster rank

Step 3: Query
> **python3 query_for_recs.py --multiassignment_prop [# clusters to query]**
> 
> switch between --multiassignment_prop and --multiassignment_equal to choose whether clusters contribute recs proportionally by user similarity to the cluster, or contribute recs equally

Step 4: Run Eval
> **python3 eval/evaluate_recall.py**


## setup guide: ##

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
- `filter_users.py path-to-user_embeddings_train path-to-user_embeddings_train --min-total 5 --min-relevant 0` - performs better when we only train with users with 5+ ratings, other users are just noise
- `cluster_embeddings.py` -> `user_clusters.json`, `svd_model.pkl` (saved SVD model)
- `generate_cluster_recs.py` -> `recs_per_cluster.json`
- `setup_pir.py` (sets up the PIR server)
- `query_for_recs.py` -> `pinpointe_train_recs.json` (this automatically runs PIR by default, to test with no pir use --no-pir flag)
-  run `evaluate_recall.py`

yay!

### baseline results!
| Algorithm | Users | Recall@10 | Recall@20 | Recall@50 | Recall@100 |
|----------|-------|-----------|-----------|-----------|------------|
| mf       | 2485  | 0.0552    | 0.0881    | 0.1474    | 0.2069     |
| **pinpointe**| 2485  | 0.0303    | 0.0497    | 0.0888    | 0.1303     |
| userknn  | 2485  | 0.0188    | 0.0339    | 0.0659    | 0.0871     |
| itemknn  | 2485  | 0.0072    | 0.0121    | 0.0207    | 0.0347     |
| poprec   | 2485  | 0.0048    | 0.0065    | 0.0103    | 0.0216     |

(actual numbers don't matter - it's just about how well it performs relative to the baseline algos, especially matrix factorization)































