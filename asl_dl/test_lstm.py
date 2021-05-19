import torch
import random
import numpy as np
import pandas as pd
from asl_data.asl_dataset import ASLDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import json
from asl_dl.train_lstm import get_loss, get_model, train_n_epochs, run_once, seed_worker
from dotmap import DotMap

SEED = 13

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def retrain(args, X, y, weights, input_dim, output_dim, writer, log_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED) # todo: fix this?
    train_indeces, val_indeces = next(skf.split(X, y))
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[train_indeces]),
                                                   torch.from_numpy(y[train_indeces]))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[val_indeces]),
                                                 torch.from_numpy(y[val_indeces]))

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, train_dataset, val_dataset,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")
    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


def test(args, X_test, y_test, weights, input_dim, output_dim, metric, log_dir):
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=6, drop_last=False, worker_init_fn=seed_worker)
    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    model.load_state_dict(torch.load('{}/state_dict_{}.pt'.format(log_dir, metric)))
    model.eval()
    test_losses, test_outs, test_gt = run_once(args, model, test_loader, criterion, None)

    return test_gt, test_outs


def main():
    df = pd.read_csv("runs/summary.csv", header=0, index_col=None)
    df.sort_values('mean_val_loss', inplace=True)
    args = df.iloc[0].to_dict()
    args = {k: getattr(v, "tolist", lambda: v)() for k, v in args.items()}
    if "model" not in args.keys():
        args["model"] = "lstm"
    if "interpolated" not in args.keys():
        args["interpolated"] = True
    args = DotMap(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.interpolated:
        folder_name = "interpolated_csvs"
    else:
        folder_name = "csvs"

    log_dir = "test_results"
    dataset = ASLDataset(folder_name, "reduced_SignData.csv",
                         sel_labels=["MajorLocation"], drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
                         different_length=not args.interpolated)

    X, y = dataset[:][0], dataset[:][1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, shuffle=True, stratify=y)

    input_dim = X[0].shape[1]
    output_dim = len(np.unique(y))

    if args.weighted_loss:
        classes, occurrences = np.unique(dataset[:][1], return_counts=True)
        weights = torch.FloatTensor(1. / occurrences).to(args.device)
    else:
        weights = None
    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = retrain(args, X_train, y_train, weights, input_dim,
                                                                         output_dim, None, log_dir)
    out_log = {}
    out_log["min_train_loss"] = train_loss_min
    out_log["max_train_f1_score"] = train_f1_max
    out_log["min_val_loss"] = valid_loss_min
    out_log["max_val_f1_score"] = valid_f1_max

    test_gt, test_outs = test(args, X_test, y_test, weights, input_dim, output_dim, "min_loss", log_dir)

    out_log["f1_score_test"] = f1_score(test_gt, test_outs, average="micro")
    out_log["confusion_matrix"] = confusion_matrix(test_gt, test_outs).tolist()
    out_log["normalized_cf_matrix"] = confusion_matrix(test_gt, test_outs, normalize="true").tolist()

    with open("{}/log_file.json".format(log_dir), "w") as fp:
        json.dump(out_log, fp)


if __name__ == '__main__':
    main()