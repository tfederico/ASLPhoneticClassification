import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from asl_ml.preprocessing import preprocess_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

test_size = 0.15

drop_features_lr = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"]
drop_features_center = ["Hip.Center"]

random_seeds = [1483533434, 3708593420, 1435909850, 1717893437, 2058363314, 375901956, 3122268818, 3001508778, 278900983, 4174692793]
print("Number of different seeds:", len(random_seeds))

clf = LogisticRegression
params = {"clf__class_weight": "balanced", "clf__multi_class": "multinomial", "clf__penalty": "l2", "clf__solver": "newton-cg", "max_iter": 10}
params = {k.replace("clf__", ""): v for k, v in params.items()}
metrics = ["micro"]
labels = ["MajorLocation"]

average_results = {}
for label in labels:
    print("Label {}".format(label))
    X, y = preprocess_dataset(label, drop_feat_lr=drop_features_lr,
                                  drop_feat_center=drop_features_center, different_length=True,
                                  trick_maj_loc=False)

    metric_results = {}
    for metric in metrics:
        results = []
        for random_seed in random_seeds:
            model = clf(random_state=random_seed)
            model.set_params(**params)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=y)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append(f1_score(y_test, y_pred, average=metric))

        metric_results[metric] = (np.mean(results), np.std(results))

    average_results[label] = metric_results

print(average_results)