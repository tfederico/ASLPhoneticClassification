import torch
import random
import numpy as np
from tqdm import tqdm
from deep_learning.models import ASLModelLSTM, ASLModelGRU, ASLModelMLP, ASLModel3DCNN, ASLModelI3D
from deep_learning.i3d import InceptionI3d
from data.dataset import CompleteASLDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import adabound
from utils.parser import get_parser
import json


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# def print_stats(dataset):
#     labels, counts = np.unique(dataset[:][1], return_counts=True)
#     print("Dataset size: ", len(dataset))
#     for l, n in zip(labels, counts):
#         print("# entries with label {}: {}".format(l, n))


def get_lr_optimizer(args, model):
    assert args.optimizer in ["sgd", "adam", "adabound"]
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    elif args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "adabound":
        return adabound.AdaBound(model.parameters(), lr=args.lr, final_lr=args.final_lr)
    else:
        raise ValueError("Optimizer name not valid.")


def get_lr_scheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=args.gamma, milestones=range(args.step_size, args.epochs, args.step_size))
    return scheduler


def get_loss(weights):
    return CrossEntropyLoss(weight=weights)


def get_model(args, input_dim, output_dim):
    if args.model == "mlp":
        return ASLModelMLP(input_dim, args.hidden_dim, output_dim, n_lin_layers=args.n_lin_layers,
                           lin_dropout=args.lin_dropout, batch_norm=args.batch_norm)
    elif args.model == "3dcnn":
        return ASLModel3DCNN(d_in=input_dim[1], h_in=input_dim[2], w_in=input_dim[3], n_cnn_layers=args.n_layers,
                             in_channels=input_dim[0], out_channels=args.out_channels, kernel_size=args.kernel_size,
                             pool_size=args.pool_size, pool_freq=args.pool_freq, n_lin_layers=args.n_lin_layers,
                             hidden_dim=args.hidden_dim, out_dim=output_dim, c_stride=args.c_stride,
                             c_padding=args.c_padding, c_dilation=args.c_dilation, c_groups=args.c_groups,
                             p_stride=args.p_stride, p_padding=args.p_padding, p_dilation=args.p_dilation,
                             dropout=args.dropout, lin_dropout=args.lin_dropout, batch_norm=args.batch_norm)
    elif args.model == "i3d":
        model = ASLModelI3D(d_in=input_dim[1], h_in=input_dim[2], w_in=input_dim[3], in_channels=input_dim[0],
                            n_lin_layers=args.n_lin_layers, hidden_dim=args.hidden_dim, out_dim=output_dim,
                            dropout=args.dropout, lin_dropout=args.lin_dropout, batch_norm=args.batch_norm)
        return model
    elif args.model == "lstm":
        model = ASLModelLSTM
    elif args.model == "gru":
        model = ASLModelGRU
    else:
        raise ValueError("Invalid value for model, must be either \"mlp\", \"3dcnn\", \"lstm\" or \"gru\", got {}".format(args.model))
    return model(input_dim, args.hidden_dim, args.n_layers, output_dim, batch_first=True,
                 dropout=args.dropout, bidirectional=args.bidirectional,
                 n_lin_layers=args.n_lin_layers, lin_dropout=args.lin_dropout,
                 batch_norm=args.batch_norm)


def run_once(args, model, loader, criterion, optimizer, is_train=False):
    losses = []
    outs = []
    gt = []
    torch.set_grad_enabled(is_train)
    for idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        if is_train:
            optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), labels.squeeze())
        gt.append(labels.squeeze().detach().cpu().numpy())
        outs.append(torch.argmax(torch.nn.functional.softmax(output.detach().cpu(), dim=1), dim=1).numpy())
        losses.append(loss.item())
        if is_train:
            loss.backward()
            optimizer.step()
    torch.set_grad_enabled(not is_train)

    return losses, np.concatenate(outs), np.concatenate(gt)


def train_n_epochs(args, train_dataset, val_dataset, weights, input_dim, output_dim, writer, log_dir, tag):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=6, drop_last=False, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                            num_workers=2, drop_last=False, worker_init_fn=seed_worker)

    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    optimizer = get_lr_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    train_loss_min = 1000
    train_f1_max = -1
    valid_loss_min = 1000
    valid_f1_max = -1
    for i in tqdm(range(args.epochs)):
        train_losses, train_outs, train_gt = run_once(args, model, train_loader, criterion, optimizer, is_train=True)
        train_f1_score = f1_score(train_gt, train_outs, average="micro")
        model.eval()
        val_losses, val_outs, val_gt = run_once(args, model, val_loader, criterion, None)
        val_f1_score = f1_score(val_gt, val_outs, average="micro")
        if writer:
            writer.add_scalar("Loss{}/train".format(tag), np.mean(train_losses), i)
            writer.add_scalar("F1{}/train".format(tag), train_f1_score, i)
            writer.add_scalar("Loss{}/val".format(tag), np.mean(val_losses), i)
            writer.add_scalar("F1{}/val".format(tag), val_f1_score, i)
        scheduler.step()
        model.train()
        if np.mean(val_losses) < valid_loss_min:
            valid_loss_min = np.mean(val_losses)
        if val_f1_score > valid_f1_max:
            valid_f1_max = val_f1_score
        train_loss_min = min(train_loss_min, np.mean(train_losses))
        train_f1_max = max(train_f1_max, train_f1_score)
    if tag == "":
        torch.save(model.state_dict(), '{}/state_dict_final.pt'.format(log_dir))
    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


def perform_validation(args, X, y, weights, input_dim, output_dim, writer, log_dir):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    min_train_loss_per_fold = {}
    max_train_f1_score_per_fold = {}
    min_val_loss_per_fold = {}
    max_val_f1_score_per_fold = {}
    for fold_num, (train_indeces, val_indeces) in enumerate(skf.split(X, y)):

        train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[train_indeces]),
                                                       torch.from_numpy(y[train_indeces]))
        val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X[val_indeces]),
                                                     torch.from_numpy(y[val_indeces]))

        train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, train_dataset, val_dataset,
                                                                                    weights, input_dim, output_dim,
                                                                                    writer, log_dir, tag="-Fold{}".format(fold_num))
        min_train_loss_per_fold[fold_num] = train_loss_min
        max_train_f1_score_per_fold[fold_num] = train_f1_max
        min_val_loss_per_fold[fold_num] = valid_loss_min
        max_val_f1_score_per_fold[fold_num] = valid_f1_max

    return min_train_loss_per_fold, max_train_f1_score_per_fold, min_val_loss_per_fold, max_val_f1_score_per_fold


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
    dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
                         sel_labels=["SignType"], drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
                         different_length=not args.interpolated)
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

    min_train_loss_per_fold, max_train_f1_score_per_fold, min_val_loss_per_fold, max_val_f1_score_per_fold = perform_validation(args, X_train, y_train,                                                                                                      weights, input_dim,
                                                                                                                                output_dim, writer,
                                                                                                                                log_dir)

    out_log["min_train_loss_per_fold"] = min_train_loss_per_fold
    out_log["max_train_f1_score_per_fold"] = max_train_f1_score_per_fold
    out_log["min_val_loss_per_fold"] = min_val_loss_per_fold
    out_log["max_val_f1_score_per_fold"] = max_val_f1_score_per_fold

    with open("{}/log_file.json".format(log_dir), "w") as fp:
        json.dump(out_log, fp)


if __name__ == '__main__':
    main()
