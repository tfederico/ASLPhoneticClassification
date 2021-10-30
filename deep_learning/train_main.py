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


def main(args):

    wandb.init(project=os.environ.get('WANDB_PROJECT', ''), entity="mrroboto", config=args)
    args = wandb.config
    init_seed(args.seed)

    body = list(range(31)) + list(range(37, 44)) + [47, 48]
    base = 49
    hand1 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    base = 49 + 21
    hand2 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    drop_features = body + hand1 + hand2
    sel_labels = ["MajorLocation"]
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
        # dataset = CompleteVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
        #                                   drop_features=drop_features,
        #                                   different_length=not args.interpolated, transform=transforms)
        with open("data/pkls/{}_video_dataset.pkl".format(
                "majloc" if ["MajorLocation"] == sel_labels else "signtype"), "rb") as fp:
            dataset = pickle.load(fp)
            dataset.set_transforms(transforms)
    else:
        folder_name = "csvs"
        # dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
        #                      sel_labels=sel_labels, drop_features=drop_features,
        #                      different_length=not args.interpolated)
        with open("data/pkls/{}_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "rb") as fp:
                dataset = pickle.load(fp)

        # print_stats(dataset)


    if args.model == "3dcnn":
        input_dim = dataset[0][0].numpy().shape
        classes, occurrences = dataset.get_num_occurrences()
        output_dim = len(classes)
        X = list(range(len(dataset)))
        y = dataset.get_labels()
    else:
        X, y = dataset[:][0], dataset[:][1]
        output_dim = len(np.unique(y))
        if args.model != "mlp":
            input_dim = X[0].shape[1]
        else:
            input_dim = X[0].shape[0] * X[0].shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=args.seed,
                                                        shuffle=True,
                                                        stratify=y)

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
                                                                                            X_train, y_train,
                                                                                            weights, input_dim,
                                                                                            output_dim, writer,
                                                                                            log_dir)

    else:
        min_train_loss, max_train_f1_score, min_val_loss, max_val_f1_score = validation(args, X_train, y_train,
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
