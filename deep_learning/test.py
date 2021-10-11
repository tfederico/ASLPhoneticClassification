import torch
import random
import numpy as np
import pandas as pd
from data.dataset import ASLDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import json
from deep_learning.train import get_loss, get_model, run_once, seed_worker, get_lr_optimizer, get_lr_scheduler
from dotmap import DotMap
from tqdm import tqdm

def train_n_epochs(args, train_dataset, weights, input_dim, output_dim, log_dir):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=0, drop_last=False, worker_init_fn=seed_worker)

    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    optimizer = get_lr_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    train_loss_min = 1000
    train_f1_max = -1
    train_history = []
    for i in tqdm(range(args.epochs)):
        train_losses, train_outs, train_gt = run_once(args, model, train_loader, criterion, optimizer, is_train=True)
        train_f1_score = f1_score(train_gt, train_outs, average="micro")
        train_history.append(np.mean(train_losses))
        scheduler.step()

        train_loss_min = min(train_loss_min, np.mean(train_losses))
        train_f1_max = max(train_f1_max, train_f1_score)

    torch.save(model.state_dict(), '{}/state_dict_final.pt'.format(log_dir))

    return train_loss_min, train_f1_max


def retrain(args, X, y, weights, input_dim, output_dim, writer, log_dir,):
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loss_min, train_f1_max = train_n_epochs(args, train_dataset, weights, input_dim, output_dim,
                                                                                log_dir)
    return train_loss_min, train_f1_max


def test(args, X_test, y_test, weights, input_dim, output_dim, log_dir):
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                             num_workers=0, drop_last=False, worker_init_fn=seed_worker)
    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    model.load_state_dict(torch.load('{}/state_dict_final.pt'.format(log_dir)))
    model.eval()
    test_losses, test_outs, test_gt = run_once(args, model, test_loader, criterion, None)

    return test_gt, test_outs


def main():
    use_loss = True # true for loss, false for f1 score

    labels = ["Movement", "SignType", "MajorLocation"]
    models = ["mlp", "gru", "lstm"]

    best_dict_movement = dict(mlp="Jun19_12-05-25_824c78007764", gru="Jun19_14-04-11_d11bbfeea33f", lstm="Jun19_21-51-37_d23428c5e3ac")
    best_dict_sign_type = dict(mlp="Jun18_11-10-34_d33faca24ba3", gru="Jun19_00-20-17_d6924f8950bf", lstm="Jun18_13-10-56_7d03cdf3b46f")
    best_dict_major_loc = dict(mlp="Jun14_10-46-21_c4326084d471", gru="Jun16_08-54-52_6dfcb86d951b", lstm="Jun14_13-18-46_eff1ca3fcc55")

    best_dicts = [best_dict_movement, best_dict_sign_type, best_dict_major_loc]

    best_folders = dict(zip(labels, best_dicts))

    for feature in labels:
        for model in models:
            df = pd.read_csv("{}/{}_runs/summary.csv".format(feature, model), header=0, index_col=None)
            df = df[df["folder"] == "{}/{}_runs/{}".format(feature, model, best_folders[feature][model])]
            df.sort_values('mean_val_loss' if use_loss else "mean_val_f1_score", inplace=True, ascending=use_loss)
            args = df.iloc[0].to_dict()
            args = {k: getattr(v, "tolist", lambda: v)() for k, v in args.items()}
            args = DotMap(args)
            args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            if args.interpolated:
                folder_name = "interpolated_csvs"
            else:
                folder_name = "csvs"
            args.epochs = 50
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
                train_loss_min, train_f1_max = retrain(args, X_train, y_train, weights, input_dim,
                                                                                     output_dim, None, log_dir)
                out_log = {}
                out_log["min_train_loss"] = train_loss_min
                out_log["max_train_f1_score"] = train_f1_max

                test_gt, test_outs = test(args, X_test, y_test, weights, input_dim, output_dim, log_dir)

                out_log["f1_score_test"] = f1_score(test_gt, test_outs, average="micro")
                out_log["macro_f1_score_test"] = f1_score(test_gt, test_outs, average="macro")
                out_log["confusion_matrix"] = confusion_matrix(test_gt, test_outs).tolist()
                out_log["normalized_cf_matrix"] = confusion_matrix(test_gt, test_outs, normalize="true").tolist()
                logs[seed] = out_log

            with open("{}/log_file_{}_{}.json".format(log_dir, feature, model), "w") as fp:
                json.dump(logs, fp)

            micro_tests = []
            macro_tests = []
            for k, v in logs.items():
                micro_tests.append(v["f1_score_test"])
                macro_tests.append(v["macro_f1_score_test"])

            print(feature, model)
            print(np.mean(micro_tests), np.std(micro_tests))
            print(np.mean(macro_tests), np.std(macro_tests))

if __name__ == '__main__':
    main()