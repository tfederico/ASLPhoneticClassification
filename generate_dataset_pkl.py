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
    if args.interpolated:
        folder_name = "interpolated_csvs"
    else:
        folder_name = "csvs"

    sel_labels = ["MajorLocation"]
    dataset = CompleteASLDataset(folder_name, "reduced_SignData.csv",
                         sel_labels=sel_labels, drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
                         different_length=not args.interpolated)

    with open("data/pkls/{}_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "wb") as fp:
        pickle.dump(dataset, fp)

    dataset = CompleteVideoASLDataset("WLASL2000", "reduced_SignData.csv", sel_labels=sel_labels,
                                      drop_features=[],
                                      different_length=not args.interpolated, transform=None)

    with open("data/pkls/{}_video_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "wb") as fp:
        pickle.dump(dataset, fp)


if __name__ == '__main__':
    main()