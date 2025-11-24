import json
import random

# Input file
input_file = "goodreads_reviews_children.json"

# Output files
output_file_map = "user_map.json"
output_file_train = "user_train.json"
output_file_test = "user_test.json"

user_map = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        user_id = data["user_id"]
        book_id = data["book_id"]
        rating = data["rating"]

        if user_id not in user_map:
            user_map[user_id] = []

        user_map[user_id].append({"book_id": book_id, "rating": rating})

# Split into train and test sets, with roughly 70% of the data in test and 30% in train
all_users = list(user_map.keys())

# Randomly shuffle but with a set seed (so can reproduce)
random.seed(42)
random.shuffle(all_users)

user_split = int(0.3 * len(all_users))  # 30% train, 70% test
train_users = set(all_users[:user_split])
test_users = set(all_users[:user_split])

train_dict = {}
test_dict = {}

for user in all_users:
    if user in train_users:
        train_dict[user] = user_map[user]
    else:
        test_dict[user] = user_map[user]


# Write all to separate json files
with open(output_file_map, "w", encoding="utf-8") as f_map:
    json.dump(user_map, f_map, indent=2)

with open(output_file_train, "w", encoding="utf-8") as f_train:
    json.dump(train_dict, f_train, indent=2)

with open(output_file_test, "w", encoding="utf-8") as f_test:
    json.dump(test_dict, f_test, indent=2)

print(f"Done! {len(train_dict)} in train.json, {len(test_dict)} in test.json")