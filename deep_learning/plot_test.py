import matplotlib
matplotlib.rcParams.update({'font.size': 12.5})
import matplotlib.pyplot as plt
import json
import numpy as np

seeds = [1483533434, 3708593420, 1435909850, 1717893437, 2058363314, 375901956, 3122268818, 3001508778, 278900983,
         4174692793]

colors = [
    dict(train="navy", valid="cornflowerblue"),
    dict(train="darkgreen", valid="limegreen"),
    dict(train="darkred", valid="lightcoral")
]

line_styles = ['-', '--', ':']
markers = ["o", "x", "v"]

models = ["mlp", "lstm", "gru"]
labels = ["MajorLocation", "SignType"]

colors_dict = dict(zip(labels, colors))
line_styles_dict = dict(zip(models, line_styles))
markers_dict = dict(zip(models, markers))

for label in labels:
    fig = plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 2)
    lw = 2
    for model in models:
        train_losses = []
        valid_losses = []
        for seed in seeds:
            with open("test_results/{}_{}_{}.json".format(label, model, seed)) as fp:
                js = json.load(fp)
                train_losses.append(js["train"])
                valid_losses.append(js["valid"])


        train_scores_mean, train_scores_std = np.mean(train_losses, axis=0), np.std(valid_losses, axis=0)
        valid_scores_mean, valid_scores_std = np.mean(valid_losses, axis=0), np.std(valid_losses, axis=0)

        param_range = range(len(train_scores_mean))
        plt.plot(param_range, train_scores_mean, label="{} (train)".format(label),
                     color=colors_dict[label]["train"], lw=lw, linestyle=line_styles_dict[model], marker=markers_dict[model])
        plt.fill_between(param_range, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color=colors_dict[label]["train"], lw=lw, linestyle=line_styles_dict[model], marker=markers_dict[model])
        plt.plot(param_range, valid_scores_mean, label="{} (val)".format(label),
                     color=colors_dict[label]["valid"], lw=lw)
        plt.fill_between(param_range, valid_scores_mean - valid_scores_std,
                         valid_scores_mean + valid_scores_std, alpha=0.2,
                         color=colors_dict[label]["valid"], lw=lw, linestyle=line_styles_dict[model], marker=markers_dict[model])
    plt.legend(loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig("test_results/crammed_{}.pdf".format(label))
    plt.close()