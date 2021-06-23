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
import json
from asl_dl.train_lstm import train_n_epochs, get_loss, get_model, run_once, seed_worker
from dotmap import DotMap


def retrain(args, X, y, weights, input_dim, output_dim, writer, log_dir,):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=args.seed, shuffle=True,
                                                        stratify=y)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                   torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                                 torch.from_numpy(y_val))

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, train_dataset, val_dataset,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")
    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


def test(args, X_test, y_test, weights, input_dim, output_dim, log_dir):
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=6, drop_last=False, worker_init_fn=seed_worker)
    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    model.load_state_dict(torch.load('{}/state_dict_final.pt'.format(log_dir)))
    model.eval()
    test_losses, test_outs, test_gt = run_once(args, model, test_loader, criterion, None)

    return test_gt, test_outs


def main():


    feature = "SignType"
    use_loss = True # true for loss, false for f1 score
    df = pd.read_csv("{}/gru_runs/summary.csv".format(feature), header=0, index_col=None)
    df.sort_values('mean_val_loss' if use_loss else "mean_val_f1_score", inplace=True, ascending=use_loss)
    args = df.iloc[0].to_dict()
    args = {k: getattr(v, "tolist", lambda: v)() for k, v in args.items()}
    args = DotMap(args)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.interpolated:
        folder_name = "interpolated_csvs"
    else:
        folder_name = "csvs"
    log_dir = "test_results"
    dataset = ASLDataset(folder_name, "reduced_SignData.csv",
                         sel_labels=[feature], drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
                         different_length=not args.interpolated)

    X, y = dataset[:][0], dataset[:][1]

    seeds = [1483533434, 3708593420, 1435909850, 1717893437, 2058363314, 375901956, 3122268818, 3001508778, 278900983, 4174692793]
    logs = {}
    for seed in seeds:
        args.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed, shuffle=True, stratify=y)

        input_dim = X[0].shape[1] if args.model != "mlp" else X[0].shape[0] * X[0].shape[1]
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

        test_gt, test_outs = test(args, X_test, y_test, weights, input_dim, output_dim, log_dir)

        out_log["f1_score_test"] = f1_score(test_gt, test_outs, average="micro")
        print("Test score for current seed:", out_log["f1_score_test"])
        out_log["confusion_matrix"] = confusion_matrix(test_gt, test_outs).tolist()
        out_log["normalized_cf_matrix"] = confusion_matrix(test_gt, test_outs, normalize="true").tolist()
        logs[seed] = out_log

    with open("{}/log_file.json".format(log_dir), "w") as fp:
        json.dump(logs, fp)

    tests = []
    for k, v in logs.items():
        tests.append(v["f1_score_test"])

    print(np.mean(tests), np.std(tests))

if __name__ == '__main__':
    main()