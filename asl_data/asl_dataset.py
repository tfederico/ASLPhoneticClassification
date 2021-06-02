import torch
import numpy as np
from os import path
from os.path import join
from os import listdir
from torch.utils.data import Dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

class ASLDataset(Dataset):
    def __init__(self, motion_path, labels_path, sel_labels, drop_features=[], transform=None, different_length=False):
        dir_path = path.dirname(path.realpath(__file__))
        self.motion_path = path.join(dir_path, motion_path)
        self.labels_path = path.join(dir_path, labels_path)
        self.transform = transform
        self.different_length = different_length
        self.pad_end = False
        self.max_length = -1
        self.motions = []
        self.motions_keys = []
        self.labels = []
        self.sel_labels = [sel_labels] if not isinstance(sel_labels, list) else sel_labels
        self.drop_features = [drop_features] if not isinstance(drop_features, list) else drop_features
        self._expand_drop_features()
        self._load_labels()
        self._load_motions()
        self._join_and_remove()
        self._preprocessing()

    def _expand_drop_features(self):
        features_lr = ["Heel", "Knee", "Hip", "Eye", "Ear", "Toe", "Pinkie", "Ankle", "Elbow", "Shoulder", "Wrist"]
        # features_center = ["Neck", "Nose", "Hip.Center", "Head"]
        self.drop_features = [f + s for f in self.drop_features for s in [".L", ".R"] if f in features_lr]
        self.drop_features = [f + a for f in self.drop_features for a in ["_x", "_y", "_z"]]

    def _load_labels(self):
        ldf = read_csv(self.labels_path)
        ldf.sort_values(by=['EntryID'], inplace=True)
        self.labels = ldf

    def _load_motions(self):
        motion_files = sorted(listdir(self.motion_path))
        for motion_file in motion_files:
            df = read_csv(join(self.motion_path, motion_file))
            df.drop("frame", axis="columns", inplace=True)
            df.drop(self.drop_features, axis="columns", inplace=True)
            self.motions.append(df.to_numpy())
            self.max_length = max(self.max_length, df.shape[0])
            self.motions_keys.append(motion_file.split("_")[1].rstrip(".csv"))
        self.motions_keys = np.array(self.motions_keys)
        self.motions = np.array(self.motions)
        assert len(self.motions_keys) == len(np.unique(self.motions_keys)), "Some motion files are not unique {}".format(
            self.motions_keys[np.unique(self.motions_keys, return_inverse=True, return_counts=True)[1] > 1])

    def _join_and_remove(self):
        files = set(self.motions_keys)
        labels = set(self.labels["EntryID"].values)
        common = files.intersection(labels)
        positions = [np.where(self.motions_keys == c)[0][0] for c in sorted(common)]
        self.motions_keys = np.array(sorted(common))
        self.motions = self.motions[positions]
        self.labels = self.labels[self.labels["EntryID"].isin(self.motions_keys)]
        self.labels.sort_values(by=['EntryID'], inplace=True) # just to make sure
        drop_cols = [c for c in self.labels.columns if c not in self.sel_labels]
        self.labels.drop(drop_cols, axis="columns", inplace=True)

    def _preprocessing(self):
        self.labels = LabelEncoder().fit_transform(self.labels.to_numpy())
        if self.different_length:
            new_motions = []
            for i in range(len(self.motions)):
                compensate = self.max_length - self.motions[i].shape[0]
                pad_width = ((0, compensate), (0, 0)) if self.pad_end else ((compensate, 0), (0, 0))
                new_motions.append(np.pad(self.motions[i], pad_width=pad_width, mode="constant", constant_values=0))
            self.motions = np.array(new_motions)
        self.motions = np.apply_along_axis(scale_in_range, 1, self.motions, -1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        motion = self.motions[idx]
        labels = self.labels[idx]
        sample = motion, labels
        if self.transform:
            sample = self.transform(sample)
        return sample


def scale_in_range(X, a, b):
    assert a < b
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (b-a) * (X - X_min)/(X_max - X_min) + a