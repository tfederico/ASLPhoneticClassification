import torch
import random
import numpy as np
from deep_learning.dataset import LoopedVideoASLDataset, CompleteASLDataset
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

    sel_labels = ["SignType.2.0"]
    d = CompleteASLDataset(folder_name, "reduced_SignData_v2.csv",
                         sel_labels=sel_labels, drop_features=drop_features,
                         different_length=True)

    with open("data/pkls/{}_dataset.pkl".format(sel_labels[0].lower()), "wb") as fp:
        pickle.dump(d, fp, protocol=4)

    # d = LoopedVideoASLDataset("WLASL2000", "reduced_SignData_v2.csv", sel_labels=sel_labels,
    #                                   drop_features=[],
    #                                   different_length=True, transform=None, window_size=50)

    print(len(d), len(d.motions), len(d.labels), len(d.motions_keys))
    assert len(d) == len(d.labels)
    assert len(d.motions) == len(d.labels)
    assert len(d.motions) == len(d)

    # with open("data/pkls/{}_video_dataset.pkl".format(sel_labels[0].lower()), "wb") as fp:
    #     pickle.dump(d, fp, protocol=4)


if __name__ == '__main__':
    main()