import torch
import random
import numpy as np
from data.dataset import ASLDataset, CompleteASLDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from deep_learning.parser import get_parser
import json
from deep_learning.train import seed_worker, train_n_epochs
import pickle

def perform_validation(args, X, y, weights, input_dim, output_dim, writer, log_dir):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=args.seed, shuffle=True, stratify=y)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                   torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                                 torch.from_numpy(y_val))

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, train_dataset, val_dataset,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")

    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.interpolated:
        folder_name = "interpolated_csvs"
    else:
        folder_name = "csvs"

    sel_labels = ["MajorLocation"]
    # dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
    #                      sel_labels=sel_labels, drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
    #                      different_length=not args.interpolated)
    with open("data/{}_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "rb") as fp:
        dataset = pickle.load(fp)

    # print_stats(dataset)
    X, y = dataset[:][0], dataset[:][1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed, shuffle=True, stratify=y)

    input_dim = X[0].shape[1] if args.model != "mlp" else X[0].shape[0] * X[0].shape[1]
    output_dim = len(np.unique(y))

    if args.weighted_loss:
        classes, occurrences = np.unique(dataset[:][1], return_counts=True)
        weights = torch.FloatTensor(1. / occurrences).to(args.device)
    else:
        weights = None

    writer = SummaryWriter()
    log_dir = writer.log_dir
    out_log = {}
    out_log["args"] = args.__dict__

    min_train_loss, max_train_f1_score, min_val_loss, max_val_f1_score = perform_validation(args, X_train, y_train,
                                                                                            weights, input_dim,
                                                                                            output_dim, writer,
                                                                                            log_dir)
    out_log["min_train_loss"] = min_train_loss
    out_log["max_train_f1_score"] = max_train_f1_score
    out_log["min_val_loss"] = min_val_loss
    out_log["max_val_f1_score"] = max_val_f1_score

    with open("{}/log_file.json".format(log_dir), "w") as fp:
        json.dump(out_log, fp)


if __name__ == '__main__':
    main()
