from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from machine_learning.preprocessing import preprocess_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
from machine_learning.model_selection import get_classifiers, get_classifiers_names, get_numerical_parameters
from sklearn.model_selection import StratifiedKFold

random_seed = 87342
different_length = True

test_size = 0.15

drop_features_lr = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"]
drop_features_center = ["Hip.Center"]

labels = ["Movement", "MajorLocation", "SignType"]
metrics = ["f1_micro"]

models_dict = dict(zip(get_classifiers_names(), get_classifiers(random_seed)))

params_dict = dict(zip(get_classifiers_names(), get_numerical_parameters()))

for label in labels:
    print("Label {}".format(label))
    X, y = preprocess_dataset(label, drop_feat_lr=drop_features_lr,
                                    drop_feat_center=drop_features_center, different_length=different_length,
                                    trick_maj_loc=False)
    #print_labels_statistics(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed, shuffle=True, stratify=y)
    for metric in metrics:
        for model, clf in models_dict.items():
            for param_name, param_range in params_dict[model].items():
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
                train_scores, valid_scores = validation_curve(clf, X_train, y_train, param_name=param_name,
                                                              param_range=param_range, scoring=metric, n_jobs=-1,
                                                              cv=cv)

                train_scores_mean = np.mean(train_scores, axis=1)
                train_scores_std = np.std(train_scores, axis=1)
                valid_scores_mean = np.mean(valid_scores, axis=1)
                valid_scores_std = np.std(valid_scores, axis=1)

                fig = plt.figure()
                plt.title("Validation Curve")
                plt.xlabel("Parameter")
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