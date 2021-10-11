from sklearn.model_selection import validation_curve
from sklearn.model_selection import train_test_split
from machine_learning.preprocessing import preprocess_dataset
import matplotlib
matplotlib.rcParams.update({'font.size': 12.5})
import matplotlib.pyplot as plt

import numpy as np
import json
from machine_learning.model_selection import get_classifiers, get_classifiers_names, get_numerical_parameters
from sklearn.model_selection import StratifiedKFold

random_seed = 87342
different_length = True

test_size = 0.15

drop_features_lr = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"]
drop_features_center = ["Hip.Center"]

labels = ["Movement", "MajorLocation", "SignType"]
metrics = ["f1_micro"]

models_names = get_classifiers_names()
params_dict = get_numerical_parameters()

colors = [
    dict(train="navy", valid="cornflowerblue"),
    dict(train="darkgreen", valid="limegreen"),
    dict(train="darkred", valid="lightcoral")
]

colors_dict = dict(zip(labels, colors))

models_dict = dict(zip(models_names, params_dict))

for metric in metrics:
    for model, params in models_dict.items():
        for param_name, param_range in params.items():
            fig = plt.figure()
            plt.xlabel(param_name)
            plt.ylabel("Score")
            plt.ylim(0.0, 1.05)
            lw = 2
            for label in labels:
                with open("valid_results/{}/{}_{}_{}.json".format(model, label, metric, param_name), "r") as fp:

                    js = json.load(fp)
                    train_scores_mean = np.array(js["train_scores_mean"])
                    train_scores_std = np.array(js["train_scores_std"])
                    valid_scores_mean = np.array(js["valid_scores_mean"])
                    valid_scores_std = np.array(js["valid_scores_std"])
                    plt.semilogx(param_range, train_scores_mean, label="{} (train)".format(label),
                                 color=colors_dict[label]["train"], lw=lw)
                    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                                     train_scores_mean + train_scores_std, alpha=0.2,
                                     color=colors_dict[label]["train"], lw=lw)
                    plt.semilogx(param_range, valid_scores_mean, label="{} (val)".format(label),
                                 color=colors_dict[label]["valid"], lw=lw)
                    plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                                     valid_scores_mean + valid_scores_std, alpha=0.2,
                                     color=colors_dict[label]["valid"], lw=lw)
            plt.legend(loc="best", ncol=2)
            plt.tight_layout()
            plt.savefig("valid_results/{}/crammed_{}_{}.pdf".format(model, metric, param_name))
            plt.close()
