import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from machine_learning.preprocessing import preprocess_dataset
from sklearn.dummy import DummyClassifier
from deep_learning.train_main import load_npy_and_pkl
import json

metrics = [accuracy_score, balanced_accuracy_score]
test_size = 0.15

zero_shot = True
tracker = "27-frank-frank"
labels = ["MajorLocation", "SignType", "Movement", "SelectedFingers", "MinorLocation", "Flexion"]

# random_seeds = [1483533434, 3708593420, 1435909850, 1717893437, 2058363314, 375901956, 3122268818, 3001508778, 278900983, 4174692793]
random_seeds = [13]
print("Number of different seeds:", len(random_seeds))

average_results = {}
for label in labels:
    print("Label {}".format(label))
    X_train, y_train = load_npy_and_pkl(label, tracker, "train+val", zero_shot)
    X_test, y_test = load_npy_and_pkl(label, tracker, "test", zero_shot)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # X = np.apply_along_axis(scale_in_range, 0, X, -1, 1)
    # X_train, X_val, X_test = X[:len(X_train)], X[len(X_train):len(X_train) + len(X_val)], X[len(X_train) + len(X_val):]
    dataset = list(zip(X, y))
    with open("data/npy/{}/{}/label2id.json".format(label.lower(), tracker), "rb") as fp:
        label2id = json.load(fp)

    metric_results = {}
    for metric in metrics:
        results = []
        for random_seed in random_seeds:
            model = DummyClassifier(strategy="most_frequent", random_state=random_seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(metric(y_test, y_pred))

        metric_results["acc" if metric == accuracy_score else "bal_acc"] = round(np.mean(results) * 100, 2)

    average_results[label] = metric_results

print(sorted(average_results.items()))