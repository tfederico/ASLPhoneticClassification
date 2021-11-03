import torch
import torchvision
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from deep_learning.utils import init_seed
from utils.parser import get_parser
import json
from sklearn.model_selection import train_test_split
from deep_learning.train_on_keypoints import perform_validation as validation
from deep_learning.train_on_video import perform_validation as cnn_validation
import wandb
import os
from data.dataset import CompleteASLDataset, HRNetASLDataset, CompleteVideoASLDataset


def viktor_to_human(X):
    X = np.transpose(X, (0, 2, 1, 3))
    X = X.reshape((X.shape[0], X.shape[1], -1))
    return X


def load_npy_and_pkl(labels, annotator, split):
    assert split in ["train", "val", "test"]
    assert annotator in ["27-frank-frank", "27_2-hrt"]
    X = np.load("data/npy/{}/{}/{}_data_joint_frank.npy".format(labels.lower(), annotator, split)).squeeze()
    X = viktor_to_human(X)
    with open("data/npy/{}/{}/{}_label_frank.pkl".format(labels.lower(), annotator, split), "rb") as fp:
        y = np.array(pickle.load(fp)[1])
    return X, y


def main(args):

    wandb.init(config=args)
    args = wandb.config
    init_seed(args.seed)
    source = args.source

    body = list(range(31)) + list(range(37, 44)) + [47, 48]
    base = 49
    hand1 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    base = 49 + 21
    hand2 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    drop_features = body + hand1 + hand2
    sel_labels = [os.environ.get('LABEL', '')]
    print(sel_labels)
    exit()
    transforms = None
    if args.model == "3dcnn":
        folder_name = "WLASL2000"
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage("RGB"),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor()
            ]
        )
        if source == "raw":
            dataset = CompleteVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
                                              drop_features=drop_features,
                                              different_length=True, transform=transforms)
        elif source == "pkl":
            with open("data/pkls/{}_video_dataset.pkl".format(sel_labels[0].lower()), "rb") as fp:
                dataset = pickle.load(fp)
                dataset.set_transforms(transforms)
        else:
            raise ValueError("npy loading not implemented")
    else:
        folder_name = "csvs"
        if source == "load":
            dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
                                 sel_labels=sel_labels, drop_features=drop_features,
                                 different_length=True)
            X, y = dataset[:][0], dataset[:][1]
        elif source == "pkl":
            with open("data/pkls/{}_dataset.pkl".format(sel_labels[0].lower()), "rb") as fp:
                dataset = pickle.load(fp)
            X, y = dataset[:][0], dataset[:][1]
        else:
            X_train, y_train = load_npy_and_pkl(sel_labels[0], "27-frank-frank", "train")
            X_val, y_val = load_npy_and_pkl(sel_labels[0], "27-frank-frank", "val")
            X_test, y_test = load_npy_and_pkl(sel_labels[0], "27-frank-frank", "test")
            X = np.concatenate([X_train, X_val, X_test])
            y = np.concatenate([y_train, y_val, y_test])
            dataset = list(zip(X, y))


    if args.model == "3dcnn":
        input_dim = dataset[0][0].numpy().shape
        classes, occurrences = dataset.get_num_occurrences()
        output_dim = len(classes)
        X = list(range(len(dataset)))
        y = dataset.get_labels()
    else:
        output_dim = len(np.unique(y))
        if args.model != "mlp":
            input_dim = X[0].shape[1]
        else:
            input_dim = X[0].shape[0] * X[0].shape[1]

    if source != "npy":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed,
                                                            shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15 / 85, random_state=args.seed,
                                                          shuffle=True)


    if args.weighted_loss:
        classes, occurrences = np.unique(y, return_counts=True)
        weights = torch.FloatTensor(1. / occurrences).to(args.device)
    else:
        weights = None

    writer = None
    log_dir = writer.log_dir if writer else ""
    out_log = {"args": args.__dict__}

    if args.model == "3dcnn":
        min_train_loss, max_train_f1_score, min_val_loss, max_val_f1_score = cnn_validation(args, dataset,
                                                                                            X_train, X_val,
                                                                                            weights, input_dim,
                                                                                            output_dim, writer,
                                                                                            log_dir)

    else:
        min_train_loss, max_train_f1_score, min_val_loss, max_val_f1_score = validation(args, dataset, X_train, y_train,
                                                                                        X_val, y_val,
                                                                                        weights, input_dim,
                                                                                        output_dim, writer,
                                                                                        log_dir)


    # out_log["min_train_loss"] = min_train_loss
    # out_log["max_train_f1_score"] = max_train_f1_score
    # out_log["min_val_loss"] = min_val_loss
    # out_log["max_val_f1_score"] = max_val_f1_score
    #
    # with open("{}/log_file.json".format(log_dir), "w") as fp:
    #     json.dump(out_log, fp)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
