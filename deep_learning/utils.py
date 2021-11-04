import random

import adabound
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from deep_learning.models import ASLModelMLP, ASLModelI3D, ASLModelLSTM, ASLModelGRU, InceptionI3d
import numpy


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


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
        model = InceptionI3d(157, in_channels=input_dim[0], dropout_keep_prob=args.dropout,
                                final_endpoint="Logits", spatial_squeeze=True)
        model.load_state_dict(torch.load('data/weights/rgb_charades.pt'))
        model.replace_logits(output_dim)
        # model = ASLModelI3D(d_in=input_dim[1], h_in=input_dim[2], w_in=input_dim[3], in_channels=input_dim[0],
        #                     n_lin_layers=args.n_lin_layers, hidden_dim=args.hidden_dim, out_dim=output_dim,
        #                     dropout=args.dropout, lin_dropout=args.lin_dropout, batch_norm=args.batch_norm)
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


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False