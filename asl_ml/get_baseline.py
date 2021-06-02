import os
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from asl_ml.preprocessing import preprocess_dataset
from sklearn.dummy import DummyClassifier


if not os.path.exists("valid_results"):
    os.makedirs("valid_results")

metrics = ["micro", "macro"]
test_size = 0.15

drop_features_lr = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"]
drop_features_center = ["Hip.Center"]

labels = ["Movement", "MajorLocation", "SignType"]

random_seeds = random.sample(range(0, 2**32), 10)
print("Number of different seeds:", len(random_seeds))

average_results = {}
for label in labels:
    print("Label {}".format(label))
    X, y, le = preprocess_dataset(label, drop_feat_lr=drop_features_lr,
                                  drop_feat_center=drop_features_center, different_length=True,
                                  trick_maj_loc=False)

    metric_results = {}
    for metric in metrics:
        results = []
        for random_seed in random_seeds:
            model = DummyClassifier(strategy="stratified" if metric == "macro" else "most_frequent", random_state=random_seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(f1_score(y_test, y_pred, average=metric))

        metric_results[metric] = (np.mean(results), np.std(results))

    average_results[label] = metric_results

print(average_results)