import numpy as np
from asl_data.asl_dataset import ASLDataset
import matplotlib.pyplot as plt
## Joints names
# Heel, Knee, Hip, Wrist, Elbow, Shoulder, Neck, Head, Nose,
# Eye, Ear, Toe, Pinkie, Ankle, Hip.Center
## w/0 ".L"/".R"
# Neck, Head, Nose, Hip.Center
## with ".L"/".R"
# Heel, Knee, Hip, Wrist, Elbow, Shoulder, Eye, Ear, Toe, Pinkie, Ankle
## all of them have "_x", "_y", "_z"

labels = ["Movement", "MajorLocation", "SignType"]

for label in labels:
    fig = plt.figure(figsize=(16, 36))
    fig2 = plt.figure(figsize=(16, 36))
    drop_features = ["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle", "Eye", "Ear", "Elbow", "Shoulder"]
    drop_features = [f+s for f in drop_features for s in [".L", ".R"]]
    drop_features += ["Neck", "Head", "Nose", "Hip.Center"]
    drop_features = [f+a for f in drop_features for a in ["_x", "_y", "_z"]]
    dataset = ASLDataset("/home/federico/Git/human_dynamics/interpolated_csvs",
                            "reduced_SignData.csv", sel_labels=[label], drop_features=drop_features)

    values = np.unique(dataset[:][1])
    n_values = len(values)
    max_val = -10
    min_val = 10
    for i in range(n_values):
        coords_ids = np.where(dataset[:][1] == values[i])[0]
        coords = dataset[coords_ids][0]
        ax_R = fig.add_subplot(2*int(np.ceil(n_values/2)), 2, (2*i)+2, projection='3d')
        ax_L = fig.add_subplot(2*int(np.ceil(n_values/2)), 2, (2*i)+1, projection='3d')
        ax_mR = fig2.add_subplot(2*int(np.ceil(n_values/2)), 2, (2*i)+2, projection='3d')
        ax_mL = fig2.add_subplot(2*int(np.ceil(n_values/2)), 2, (2*i)+1, projection='3d')
        ax_R.set_title("{} (right)".format(values[i]))
        ax_L.set_title("{} (left)".format(values[i]))
        ax_mR.set_title("{} (right)".format(values[i]))
        ax_mL.set_title("{} (left)".format(values[i]))
        xmax = np.max(coords[:, :, 0])
        xmax = max(np.max(coords[:, :, 3]), xmax)
        xmin = np.min(coords[:, :, 0])
        xmin = min(np.min(coords[:, :, 3]), xmin)
        ymax = np.max(coords[:, :, 1])
        ymax = max(np.max(coords[:, :, 4]), ymax)
        ymin = np.min(coords[:, :, 1])
        ymin = min(np.min(coords[:, :, 4]), ymin)
        zmax = np.max(coords[:, :, 2])
        zmax = max(np.max(coords[:, :, 5]), zmax)
        zmin = np.min(coords[:, :, 2])
        zmin = min(np.min(coords[:, :, 5]), zmin)
        xlim = [xmin, xmax]
        ylim = [ymin, ymax]
        zlim = [zmin, zmax]
        ax_R.set_xlim(xlim)
        ax_R.set_ylim(ylim)
        ax_R.set_zlim(zlim)
        ax_L.set_xlim(xlim)
        ax_L.set_ylim(ylim)
        ax_L.set_zlim(zlim)
        ax_mR.set_xlim(xlim)
        ax_mR.set_ylim(ylim)
        ax_mR.set_zlim(zlim)
        ax_mL.set_xlim(xlim)
        ax_mL.set_ylim(ylim)
        ax_mL.set_zlim(zlim)
        ax_mR.plot(np.mean(coords[:, :, 0], axis=0), np.mean(coords[:, :, 1], axis=0), np.mean(coords[:, :, 2], axis=0), linewidth=0.2, linestyle='dotted', marker='o', markersize=1)
        ax_mL.plot(np.mean(coords[:, :, 3], axis=0), np.mean(coords[:, :, 4], axis=0), np.mean(coords[:, :, 5], axis=0), linewidth=0.2, linestyle='dotted', marker='o', markersize=1)
        for j in range(coords.shape[0]):
            hand_R = coords[j, :, :3]
            hand_L = coords[j, :, 3:]
            ax_R.plot(hand_R[:, 0], hand_R[:, 1], hand_R[:, 2], linewidth=0.2, linestyle='dotted', marker='o', markersize=1)
            ax_L.plot(hand_L[:, 0], hand_L[:, 1], hand_L[:, 2], linewidth=0.2, linestyle='dotted', marker='o', markersize=1)

    fig.savefig("Trajectory_{}.pdf".format(label), format="pdf")
    fig2.savefig("MeanTrajectory_{}.pdf".format(label), format="pdf")
