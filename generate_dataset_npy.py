import torch
import random
import numpy as np
from data.dataset import CompleteVideoASLDataset, LoopedVideoASLDataset
from utils.parser import get_parser
import pickle
import torchvision
from tqdm import tqdm

def main():
    parser = get_parser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage("RGB"),
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor()
        ]
    )


    sel_labels = ["SignType"]

    dataset = LoopedVideoASLDataset("WLASL2000", "reduced_SignData.csv", sel_labels=sel_labels,
                                      drop_features=[],
                                      different_length=True, transform=None)
    dataset.set_transforms(transforms)

    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        video_name = dataset.motions_keys[i]
        np.save("data/npy/videos/{}.npy".format(video_name), sample[0].numpy())
        exit()


if __name__ == '__main__':
    main()