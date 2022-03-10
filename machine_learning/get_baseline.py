import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, matthews_corrcoef, f1_score
from sklearn.model_selection import train_test_split
from machine_learning.preprocessing import preprocess_dataset
from sklearn.dummy import DummyClassifier
from deep_learning.train_main import load_npy_and_pkl
import json

metrics = {
    # "acc": accuracy_score,
    # "bacc": balanced_accuracy_score,
    "prec": precision_score,
    "rec": recall_score,
    "mcc": matthews_corrcoef,
    # "f1": f1_score
}
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
    X_train, y_train, _ = load_npy_and_pkl(label, tracker, "train+val", zero_shot)
    X_test, y_test, _ = load_npy_and_pkl(label, tracker, "test", zero_shot)
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train, y_test])
    # X = np.apply_along_axis(scale_in_range, 0, X, -1, 1)
    # X_train, X_val, X_test = X[:len(X_train)], X[len(X_train):len(X_train) + len(X_val)], X[len(X_train) + len(X_val):]
    dataset = list(zip(X, y))
    with open("data/npy/{}/{}/label2id.json".format(label.lower(), tracker), "rb") as fp:
        label2id = json.load(fp)

    metric_results = {}
    for k, metric in metrics.items():
        results = []
        for random_seed in random_seeds:
            model = DummyClassifier(strategy="most_frequent", random_state=random_seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if k in ["prec", "rec", "f1"]:
                results.append(metric(y_test, y_pred, average="micro"))
                results.append(metric(y_test, y_pred, average="macro"))
            else:
                results.append(metric(y_test, y_pred))

        if k in ["prec", "rec", "f1"]:
            metric_results["micro"+k] = round(np.mean(results[::2]) * 100, 2)
            metric_results["macro"+k] = round(np.mean(results[1::2]) * 100, 2)
        else:
            metric_results[k] = round(np.mean(results) * 100, 2)

    average_results[label] = metric_results

line = "& "
for k, v in sorted(average_results.items()):
    line += " & ".join(["$"+str(vv)+"$" for vv in v.values()])
    line += " & "
print(line[:-2]+" \\\\")