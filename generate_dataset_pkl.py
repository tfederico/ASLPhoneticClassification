import torch
import random
import numpy as np
from data.dataset import CompleteASLDataset, CompleteVideoASLDataset
from utils.parser import get_parser
import pickle


def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    folder_name = "csvs"

    body = list(range(31)) + list(range(37, 44)) + [47, 48]
    base = 49
    hand1 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    base = 49 + 21
    hand2 = [i + base for i in [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]]
    drop_features = body + hand1 + hand2

    sel_labels = ["SignType"]
    dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
                         sel_labels=sel_labels, drop_features=drop_features,
                         different_length=True)

    with open("data/pkls/{}_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "wb") as fp:
        pickle.dump(dataset, fp, protocol=4)

    dataset = CompleteVideoASLDataset("WLASL2000", "reduced_SignData.csv", sel_labels=sel_labels,
                                      drop_features=[],
                                      different_length=True, transform=None)

    with open("data/pkls/{}_video_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "wb") as fp:
        pickle.dump(dataset, fp, protocol=4)


if __name__ == '__main__':
    main()