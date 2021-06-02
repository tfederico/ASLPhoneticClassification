from sklearn.model_selection import validation_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from asl_ml.preprocessing import preprocess_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import json
from asl_ml.model_selection import get_classifiers, get_classifiers_names, get_numerical_parameters

random_seed = 87342

test_size = 0.15

drop_features_lr = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"]
drop_features_center = ["Hip.Center"]

labels = ["Movement", "MajorLocation", "SignType"]
metrics = ["f1_micro", "f1_macro"]

models_dict = dict(zip(get_classifiers_names(), get_classifiers(random_seed)))
models_dict.pop("Dummy")

params_dict = dict(zip(get_classifiers_names(), get_numerical_parameters()))
params_dict.pop("Dummy")

for label in labels:
    print("Label {}".format(label))
    X, y, le = preprocess_dataset(label, drop_feat_lr=drop_features_lr,
                                    drop_feat_center=drop_features_center, different_length=True,
                                    trick_maj_loc=False)
    #print_labels_statistics(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=y)
    
    for model, clf in models_dict.items():
        for metric in metrics:
            with open("valid_results/{}_{}.json".format(label, metric), "r") as fp:
                best_params = {k.replace("clf__", ""): v for k, v in json.load(fp)[model].items()}
                clf.set_params(**best_params)
                for param_name, param_range in params_dict[model].items():
                        train_scores, valid_scores = validation_curve(clf, X_train, y_train, param_name=param_name,
                                                                      param_range=param_range, scoring=metric, n_jobs=-1,
                                                                      cv=5)

                        train_scores_mean = np.mean(train_scores, axis=1)
                        train_scores_std = np.std(train_scores, axis=1)
                        valid_scores_mean = np.mean(valid_scores, axis=1)
                        valid_scores_std = np.std(valid_scores, axis=1)

                        fig = plt.figure()
                        plt.title("Validation Curve")
                        plt.xlabel("Iters")
                        plt.ylabel("Score")
                        plt.ylim(0.0, 1.1)
                        lw = 2
                        plt.semilogx(param_range, train_scores_mean, label="Training score",
                                     color="darkorange", lw=lw)
                        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                                         train_scores_mean + train_scores_std, alpha=0.2,
                                         color="darkorange", lw=lw)
                        plt.semilogx(param_range, valid_scores_mean, label="Cross-validation score",
                                     color="navy", lw=lw)
                        plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                                         valid_scores_mean + valid_scores_std, alpha=0.2,
                                         color="navy", lw=lw)
                        plt.legend(loc="best")
                        plt.savefig("valid_results/{}/{}_{}_{}.pdf".format(model, label, metric, param_name))
                        plt.close()