import pickle as pkl
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True, type=str, choices=["MajorLocation", "SignType", "Movement"])
parser.add_argument("--tracker", required=True, type=str, choices=["frank", "hrnet"])
parser.add_argument("--zero_shot", action='store_true')

args = parser.parse_args()
label = args.label.lower()
tracker = args.tracker.replace("hrnet", "hrt")
zero_shot = "-zs" if args.zero_shot else ""
folder = "27-frank-frank" if tracker == "frank" else "27_2-hrt"
folder += zero_shot
tracker += zero_shot

with open(f"data/npy/{label}/{folder}/test_label_{tracker}.pkl", "rb") as fp:
    test_label = pkl.load(fp)

test_label = zip(test_label[0], test_label[1])

with open(f"data/npy/{label}/{folder}/split.json", "r") as fp:
    split = json.load(fp)

test_split = split["test"]

with open(f"data/npy/{label}/{folder}//label2id.json", "r") as fp:
    label2class_id = json.load(fp)

class_id2label = {v: k for k, v in label2class_id.items()}

with open(f"data/npy/{label}/{folder}/file2gloss.json", "r") as fp:
    file2gloss = json.load(fp)

video_id2label = {}
for video_id, class_id in test_label:
    video_id2label[video_id] = class_id2label[class_id]

with open(f"data/npy/{label}/{folder}/gt.csv", "w") as fp:
    for video_id, label in sorted(video_id2label.items()):
        fp.write(','.join([video_id, file2gloss[video_id], label])+"\n")