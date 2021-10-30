import torch
import torchvision
import numpy as np
from os import path
from os.path import join
from os import listdir
from torch.utils.data import Dataset
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
import json
from tqdm import tqdm
import pandas as pd
import cv2

from joblib import Parallel, delayed

class ASLDataset(Dataset):
    def __init__(self, motion_path, labels_path, sel_labels, drop_features=[], transform=None, different_length=False, do_preprocessing=True,
                 relabel_map={}):
        dir_path = path.dirname(path.realpath(__file__))
        self.motion_path = path.join(dir_path, motion_path)
        self.labels_path = path.join(dir_path, labels_path)
        self.transform = transform
        self.different_length = different_length
        self.relabel_map = relabel_map
        self.pad_end = False
        self.max_length = -1
        self.motions = []
        self.motions_keys = []
        self.labels = []
        self.sel_labels = [sel_labels] if not isinstance(sel_labels, list) else sel_labels
        self.drop_features = [drop_features] if not isinstance(drop_features, list) else drop_features
        self._load_labels()
        self._load_motions()
        self._join_and_remove()
        self.do_preprocessing = do_preprocessing
        self._preprocessing()

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
        le = LabelEncoder()
        old = self.labels.to_numpy()
        labels = old.copy()
        for ks, v in self.relabel_map.items():
            for k in ks:
                labels[old == k] = v

        self.labels = le.fit_transform(labels)
        self.label_to_id = dict(zip(le.classes_, le.transform(le.classes_)))
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        #self.labels = LabelEncoder().fit_transform(self.labels.to_numpy())
        if self.do_preprocessing:
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
        if self.transform:
            motion = self.transform(motion)
        sample = motion, labels
        return sample


def scale_in_range(X, a, b):
    assert a < b
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    return (b-a) * (X - X_min)/(X_max - X_min) + a

def load_motions_parallel(motion_path, motion_file, drop_features):
    df = read_csv(path.join(motion_path, motion_file))
    df.drop("frame", axis="columns", inplace=True)
    df.drop(df.columns[drop_features], axis="columns", inplace=True)
    df.fillna(0, inplace=True)
    return df.to_numpy(), df.shape[0], motion_file.replace(".csv", "")

class CompleteASLDataset(ASLDataset):
    def __init__(self, motion_path, labels_path, sel_labels, drop_features=[], transform=None, different_length=False, 
                 map_file="WLASL_v0.3.json", debug=False, *args, **kwargs):
        dir_path = path.dirname(path.realpath(__file__))
        self.map_file = path.join(dir_path, map_file)
        self.debug = debug or []
        super().__init__(motion_path, labels_path, sel_labels, drop_features, transform, different_length)

    def _load_motions(self):
        motion_files = sorted(listdir(self.motion_path))
        if self.debug:
            motion_files = [f"{x}.csv" for x in self.debug]
        res = Parallel(n_jobs=7)(delayed(load_motions_parallel)(self.motion_path, m, self.drop_features) for m in tqdm(motion_files))
        # res = [load_motions_parallel(self.motion_path, m, self.drop_features) for m in tqdm(motion_files)]
        for motion_df, max_len, motion_key in res:
            self.motions.append(motion_df)
            self.max_length = max(self.max_length, max_len)
            self.motions_keys.append(motion_key)
        self.motions_keys = np.array(self.motions_keys)
        self.motions = np.array(self.motions)
        assert len(self.motions_keys) == len(np.unique(self.motions_keys)), "Some motion files are not unique {}".format(
            self.motions_keys[np.unique(self.motions_keys, return_inverse=True, return_counts=True)[1] > 1])

    def _join_and_remove(self):
        with open(self.map_file, "r") as fp:
            wlasl_v03 = json.load(fp)

        gloss_id_dict = {}
        for gloss_dict in wlasl_v03:
            gloss = gloss_dict["gloss"]
            ids = []
            for instance in gloss_dict["instances"]:
                ids.append(instance["video_id"])
            gloss_id_dict[gloss] = ids

        rev_gloss_id_dict = {}

        for k, v in gloss_id_dict.items():
            if isinstance(v, list):
                for val in v:
                    rev_gloss_id_dict[val] = k
            else:
                rev_gloss_id_dict[v] = k

        files = set(self.motions_keys).intersection(set(rev_gloss_id_dict.keys())) # remove from all ids the one who don't have an associated video
        labels = set(self.labels["EntryID"].values)
        files = set([rev_gloss_id_dict[v].lower() for v in files])
        common = sorted(files.intersection(labels))
        file_list = [idx for gloss in common for idx in gloss_id_dict[gloss]]
        self.file_list = file_list
        self.gloss2file = gloss_id_dict
        self.file2gloss = rev_gloss_id_dict
        all_good_ids = []
        for c in common:
            videos_ids = gloss_id_dict[c]
            all_good_ids += videos_ids
        positions = [np.where(self.motions_keys == c)[0] for c in all_good_ids if np.isin(c, self.motions_keys)]
        positions = np.array(positions).flatten()
        self.motions_keys = self.motions_keys[positions]
        self.motions = self.motions[positions]
        self.labels = [self.labels[self.labels["EntryID"] == rev_gloss_id_dict[v]] for v in self.motions_keys]
        self.labels = pd.concat(self.labels)
        drop_cols = [c for c in self.labels.columns if c not in self.sel_labels]
        self.labels.drop(drop_cols, axis="columns", inplace=True)

    def _load_labels(self):
        ldf = read_csv(self.labels_path)
        ldf.sort_values(by=['EntryID'], inplace=True) # sort them so that when you remove the duplicates the first has _1
        ldf = ldf.groupby(by="LemmaID", as_index=False).first() # remove duplicate entries
        ldf["EntryID"] = ldf["EntryID"].str.replace("_1", "") # remove duplicate number from name
        ldf["EntryID"] = ldf["EntryID"].str.replace("_", " ") # remove underscore
        ldf["EntryID"] = ldf["EntryID"].str.lower()
        # some are easier to fix manually...
        ldf.loc[ldf['EntryID'] == "hotdog", ['EntryID']] = "hot dog"
        ldf.loc[ldf['EntryID'] == "frenchfries", ['EntryID']] = "french fries"
        ldf.loc[ldf['EntryID'] == "icecream", ['EntryID']] = "ice cream"
        self.labels = ldf


def load_npy_motions_parallel(motion_path, motion_file):
    skel = np.load(os.path.join(motion_path, motion_file))
    return skel, skel.shape[0], motion_file.replace(".mp4.npy", "")


class HRNetASLDataset(CompleteASLDataset):
    def _load_motions(self):
        motion_files = sorted(listdir(self.motion_path))
        if self.debug:
            motion_files = [f"{x}.npy" for x in self.debug]
        # res = Parallel(n_jobs=11)(delayed(load_npy_motions_parallel)(self.motion_path, m) for m in tqdm(motion_files))
        res = [load_npy_motions_parallel(self.motion_path, m) for m in tqdm(motion_files)]
        for skel, max_len, motion_key in res:
            self.motions.append(skel)
            self.max_length = max(self.max_length, max_len)
            self.motions_keys.append(motion_key)
        self.motions_keys = np.array(self.motions_keys)
        self.motions = np.array(self.motions)
        assert len(self.motions_keys) == len(np.unique(self.motions_keys)), "Some motion files are not unique {}".format(
            self.motions_keys[np.unique(self.motions_keys, return_inverse=True, return_counts=True)[1] > 1])

class CompleteVideoASLDataset(CompleteASLDataset):
    def __init__(self, motion_path, labels_path, sel_labels, drop_features=[], transform=None, different_length=False, map_file="WLASL_v0.3.json"):
        self.n_output_classes = -1
        super().__init__(motion_path, labels_path, sel_labels, drop_features, transform, different_length, map_file)

    def _load_motions(self):
        motion_files = sorted(listdir(self.motion_path))
        for motion_file in tqdm(motion_files):
            vframes, aframes, info = torchvision.io.read_video(join(self.motion_path, motion_file))
            self.max_length = max(self.max_length, len(vframes))
            self.motions_keys.append(motion_file.replace(".mp4", ""))
        self.motions_keys = np.array(self.motions_keys)
        assert len(self.motions_keys) == len(np.unique(self.motions_keys)), "Some motion files are not unique {}".format(
            self.motions_keys[np.unique(self.motions_keys, return_inverse=True, return_counts=True)[1] > 1])

    def _join_and_remove(self):
        with open(self.map_file, "r") as fp:
            wlasl_v03 = json.load(fp)

        gloss_id_dict = {}
        for gloss_dict in wlasl_v03:
            gloss = gloss_dict["gloss"]
            ids = []
            for instance in gloss_dict["instances"]:
                ids.append(instance["video_id"])
            gloss_id_dict[gloss] = ids

        rev_gloss_id_dict = {}

        for k, v in gloss_id_dict.items():
            if isinstance(v, list):
                for val in v:
                    rev_gloss_id_dict[val] = k
            else:
                rev_gloss_id_dict[v] = k

        files = set(self.motions_keys).intersection(set(rev_gloss_id_dict.keys())) # remove from all ids the one who don't have an associated video
        labels = set(self.labels["EntryID"].values)
        files = set([rev_gloss_id_dict[v].lower() for v in files])
        common = sorted(files.intersection(labels))
        all_good_ids = []
        for c in common:
            videos_ids = gloss_id_dict[c]
            all_good_ids += videos_ids
        positions = [np.where(self.motions_keys == c)[0] for c in all_good_ids if np.isin(c, self.motions_keys)]
        positions = np.array(positions).flatten()
        self.motions_keys = self.motions_keys[positions]
        self.labels = [self.labels[self.labels["EntryID"] == rev_gloss_id_dict[v]] for v in self.motions_keys]
        self.labels = pd.concat(self.labels)
        drop_cols = [c for c in self.labels.columns if c not in self.sel_labels]
        self.labels.drop(drop_cols, axis="columns", inplace=True)

    def _preprocessing(self):
        self.labels = LabelEncoder().fit_transform(self.labels.to_numpy())

    def _pad_sequence(self, sequence):
        diff = self.max_length - len(sequence)
        if diff == 0:
            return sequence
        else:
            pad_dim = (diff,) + sequence.shape[1:]
            pad = torch.zeros(pad_dim)
            seq = torch.cat((sequence, pad))
            return seq

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        labels = self.labels[idx]
        if not isinstance(labels, (list, np.ndarray)):
            labels = np.asarray(labels)
        motion_key = self.motions_keys[idx]
        if isinstance(motion_key, (np.ndarray, list)):
            data = []
            for i in range(len(labels)):
                cap = cv2.VideoCapture(join(self.motion_path, motion_key[i] + ".mp4"))
                motion = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not (frame is None):
                        if self.transform:
                            frame = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        motion.append(frame)
                    else:
                        break
                cap.release()
                motion = self._pad_sequence(torch.stack(motion))
                data.append(motion)
            data = torch.stack(data)
            data = data.permute(0, 2, 1, 3, 4)
        else:
            cap = cv2.VideoCapture(join(self.motion_path, motion_key+".mp4"))
            motion = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not (frame is None):
                    if self.transform:
                        frame = self.transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    motion.append(frame)
                else:
                    break
            cap.release()
            data = self._pad_sequence(torch.stack(motion))
            data = data.permute(1, 0, 2, 3)
        sample = data, torch.from_numpy(labels)
        return sample

    def get_max_length(self):
        return self.max_length

    def get_num_occurrences(self):
        return np.unique(self.labels, return_counts=True)

    def get_labels(self):
        return self.labels

    def set_transforms(self, transforms):
        self.transform = transforms
