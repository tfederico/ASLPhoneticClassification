import torch
import torchvision
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from deep_learning.utils import init_seed
from utils.parser import get_parser
import json
from sklearn.model_selection import train_test_split
from deep_learning.train_valid import run_once
from deep_learning.utils import get_loss, get_lr_optimizer, get_lr_scheduler, get_model, seed_worker
import wandb
from deep_learning.dataset import CompleteASLDataset, LoopedVideoASLDataset, NpyLoopedVideoASLDataset
from deep_learning.dataset import scale_in_range
from deep_learning.train_main import load_npy_and_pkl
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef


def adapt_shape(X):
    X = np.transpose(X, (0, 2, 1, 3))
    X = X.reshape((X.shape[0], X.shape[1], -1))
    return X


def train_and_test(args, label2id, train_dataset, test_dataset, weights, input_dim, output_dim, writer, log_dir, tag):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=8, drop_last=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size,
                            num_workers=8, drop_last=False, worker_init_fn=seed_worker)

    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    optimizer = get_lr_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    train_loss_min = 1000
    train_f1_max = -1

    for i in tqdm(range(args.epochs)):
        train_losses, train_outs, train_gt = run_once(args, model, train_loader, criterion, optimizer, is_train=True)
        train_f1_score = f1_score(train_gt, train_outs, average="micro")

        if writer:
            writer.add_scalar("Loss{}/train".format(tag), np.mean(train_losses), i)
            writer.add_scalar("F1{}/train".format(tag), train_f1_score, i)

        wdb_log = {
            "train/loss": np.mean(train_losses).item(),
            "train/micro_f1": train_f1_score,
            "train/macro_f1": f1_score(train_gt, train_outs, average="macro"),
            "train/accuracy": accuracy_score(train_gt, train_outs),
            "train/balanced_accuracy": balanced_accuracy_score(train_gt, train_outs),
            "train/mcc": matthews_corrcoef(train_gt, train_outs)
        }

        for k, v in label2id.items():
            indices = np.where(train_gt == v)
            wdb_log[f"train/acc_{k}"] = accuracy_score(train_gt[indices], train_outs[indices])

        scheduler.step()
        wandb.log(wdb_log, step=i)

    model.eval()
    test_losses, test_outs, test_gt = run_once(args, model, test_loader, criterion, None)

    wdb_log = {
        "test/loss": np.mean(test_losses).item(),
        "test/micro_f1": f1_score(test_gt, test_outs, average="micro"),
        "test/macro_f1": f1_score(test_gt, test_outs, average="macro"),
        "test/accuracy": accuracy_score(test_gt, test_outs),
        "test/balanced_accuracy": balanced_accuracy_score(test_gt, test_outs),
        "test/mcc": matthews_corrcoef(test_gt, test_outs)
    }
    wandb.log(wdb_log, step=i)

    return test_outs, test_gt


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
    sel_labels = [args.label]

    transforms = None
    if args.model == "3dcnn":
        folder_name = "../data/WLASL2000"
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage("RGB"),
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.RandomCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
        if source == "raw":
            dataset = LoopedVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
                                              window_size=args.window_size, drop_features=drop_features,
                                              different_length=True, transform=transforms)
        elif source == "pkl":
            with open("data/pkls/{}_video_dataset.pkl".format(sel_labels[0].lower()), "rb") as fp:
                dataset = pickle.load(fp)
                dataset.set_transforms(transforms)
        else:
            suffix = "frank" if "frank" in args.tracker else "hrt"
            if args.zero_shot:
                suffix += "-zs"
            train_dataset = NpyLoopedVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
                                                    window_size=args.window_size, tracker=args.tracker, set_type="train+val",
                                                    suffix=suffix, drop_features=drop_features, different_length=True,
                                                    transform=transforms)
            test_dataset = NpyLoopedVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
                                                    window_size=args.window_size, tracker=args.tracker, set_type="test",
                                                    suffix=suffix, drop_features=drop_features, different_length=True,
                                                    transform=transforms)
            ids_test = test_dataset.motions_keys
            with open("data/npy/{}/{}/label2id.json".format(sel_labels[0].lower(), args.tracker+"-zs" if args.zero_shot else args.tracker), "rb") as fp:
                label2id = json.load(fp)
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
            X_train, y_train, ids_train = load_npy_and_pkl(sel_labels[0], args.tracker, "train+val", zero_shot=args.zero_shot)
            X_test, y_test, ids_test = load_npy_and_pkl(sel_labels[0], args.tracker, "test", zero_shot=args.zero_shot)
            X = np.concatenate([X_train, X_test])
            y = np.concatenate([y_train, y_test])
            X = np.apply_along_axis(scale_in_range, 0, X, -1, 1)
            X_train, X_test = X[:len(X_train)], X[len(X_train):]
            dataset = list(zip(X, y))
            with open("data/npy/{}/{}/label2id.json".format(sel_labels[0].lower(), args.tracker), "rb") as fp:
                label2id = json.load(fp)


    if args.model == "3dcnn":
        input_dim = train_dataset[0][0].numpy().shape
        classes, occurrences = train_dataset.get_num_occurrences()
        output_dim = len(classes)
        y = np.concatenate((train_dataset.get_labels(), test_dataset.get_labels()))
    else:
        output_dim = len(np.unique(y))
        if args.model != "mlp":
            input_dim = X[0].shape[1]
        else:
            input_dim = X[0].shape[0] * X[0].shape[1]

    if source != "npy":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed,
                                                            shuffle=True)


    if args.weighted_loss:
        classes, occurrences = np.unique(y, return_counts=True)
        weights = torch.FloatTensor(1. / occurrences).to(args.device)
    else:
        weights = None

    writer = None
    log_dir = writer.log_dir if writer else ""
    out_log = {"args": args.__dict__}

    if args.model != "3dcnn":
        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                       torch.from_numpy(y_train))
        test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test),
                                                     torch.from_numpy(y_test))

    test_out, test_gt = train_and_test(args, label2id, train_dataset, test_dataset,
                                                    weights, input_dim,
                                                    output_dim, writer,
                                                    log_dir, "")
                                                    
    id2label = {v: k for k, v in label2id.items()}
    
    test_out = [id2label[t] for t in test_out]
    test_gt = [id2label[t] for t in test_gt]

    results = dict(zip([int(i) for i in ids_test], test_out))
    with open(f"temp_results_{args.model}_{args.label}_{args.tracker}_{args.zero_shot}.json", "w") as fp:
        json.dump(results, fp, sort_keys=True)
    wandb.save(f"temp_results_{args.model}_{args.label}_{args.tracker}_{args.zero_shot}.json")




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
