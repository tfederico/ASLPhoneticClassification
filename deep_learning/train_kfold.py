import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from deep_learning.train_on_keypoints import train_n_epochs


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