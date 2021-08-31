import seaborn as sns
import matplotlib
matplotlib.rcParams.update({'font.size': 26})
import json
import matplotlib.pyplot as plt
import numpy as np

sign_type = ['AsymmetricalDifferentHandshape', 'AsymmetricalSameHandshape', 'OneHanded', 'Other', 'SymmetricalOrAlternating']
sign_type = ['ADH', 'ASH', 'OH', 'O', 'SOA']
movement = ['BackAndForth', 'Circular', 'Curved', 'None', 'Other', 'Straight']
movement = ['BAF', 'Cir', 'Curv', 'None', 'O', 'Str']
major_loc = ['Arm', 'Body', 'Hand', 'Head', 'Neutral']
major_loc = ['Arm', 'Body', 'Hand', 'Head', 'Neutr']
labels = ["MajorLocation", "Movement", "SignType"]

names = dict(zip(labels, [major_loc, movement, sign_type]))

i = 0
f, ax = plt.subplots(1, 3, figsize=(30, 8))
for label in labels:
    with open("test_results/log_file_{}_mlp.json".format(label), "r") as fp:
        cm = np.array(json.load(fp)["1483533434"]["confusion_matrix"])
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cmn, vmin=0, vmax=1, annot=True, fmt='.2f', cmap="Blues",
                square=True, ax=ax[i], xticklabels=names[label], yticklabels=names[label])
    ax[i].set_xlabel("Predicted")
    ax[i].set_ylabel("Actual")
    i += 1


plt.tight_layout()
plt.savefig("test_results/cf_mlp.pdf", format="pdf")