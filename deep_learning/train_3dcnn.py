import torch
import torchvision
import random
import pickle
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from utils.parser import get_parser
import json
from deep_learning.train import get_loss, get_lr_scheduler, get_lr_optimizer
from deep_learning.models import get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import math
from sklearn.utils import shuffle


def run_once(args, model, dataset, ids, criterion, optimizer, is_train=False):
    losses = []
    outs = []
    gt = []
    num_batches = math.ceil(len(ids)/args.batch_size)
    ids_copy = ids.copy()
    if is_train:
        ids_copy = shuffle(ids_copy, random_state=args.seed)

    torch.set_grad_enabled(is_train)
    for i in tqdm(range(num_batches), leave=False):
        start_id = i*args.batch_size
        stop_id = (i+1)*args.batch_size if (i+1)*args.batch_size < len(ids) else len(ids)
        batch_ids = ids_copy[start_id:stop_id]
        inputs = dataset[batch_ids][0]
        labels = dataset[batch_ids][1]
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        if is_train:
            optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze() if labels.shape[0] != 1 else output, labels.squeeze() if labels.shape[0] != 1 else labels)
        gt.append(labels.squeeze().detach().cpu().numpy() if labels.shape[0] != 1 else labels.detach().cpu().numpy())
        outs.append(torch.argmax(torch.nn.functional.softmax(output.detach().cpu(), dim=1), dim=1).numpy())
        losses.append(loss.item())
        if is_train:
            loss.backward()
            optimizer.step()
    torch.set_grad_enabled(not is_train)

    outs = np.concatenate(outs)
    gt = np.concatenate(gt)

    return losses, outs, gt


def train_n_epochs(args, dataset, train_ids, val_ids, weights, input_dim, output_dim, writer, log_dir, tag):

    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    optimizer = get_lr_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    train_loss_min = 1000
    train_f1_max = -1
    valid_loss_min = 1000
    valid_f1_max = -1
    for i in tqdm(range(args.epochs)):
        train_losses, train_outs, train_gt = run_once(args, model, dataset, train_ids, criterion, optimizer, is_train=True)
        train_f1_score = f1_score(train_gt, train_outs, average="micro")
        model.eval()
        val_losses, val_outs, val_gt = run_once(args, model, dataset, val_ids, criterion, None)
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


def perform_validation(args, dataset, ids, y, weights, input_dim, output_dim, writer, log_dir):
    train_ids, val_ids, y_train, y_val = train_test_split(ids, y, test_size=0.15, random_state=args.seed, shuffle=True, stratify=y)

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, dataset, train_ids, val_ids,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")

    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


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
    sel_labels = ["MajorLocation"]
    # folder_name = "small_WLASL2000"
    # dataset = CompleteVideoASLDataset(folder_name, "reduced_SignData.csv", sel_labels=sel_labels,
    #                                   drop_features=["Heel", "Knee", "Hip", "Toe", "Pinkie", "Ankle"],
    #                                   different_length=not args.interpolated, transform=transforms)
    with open("data/pkls/small_{}_video_dataset.pkl".format("majloc" if ["MajorLocation"] == sel_labels else "signtype"), "rb") as fp:
        dataset = pickle.load(fp)
        dataset.set_transforms(transforms)

    # print_stats(dataset)

    input_dim = dataset[0][0].numpy().shape #X[0].shape[1] if args.model != "mlp" else X[0].shape[0] * X[0].shape[1]
    classes, occurrences = dataset.get_num_occurrences()
    output_dim = len(classes)

    data_ids = list(range(len(dataset)))
    y = dataset.get_labels()
    train_ids, test_ids, y_train, y_test = train_test_split(data_ids, y, test_size=0.15, random_state=args.seed, shuffle=True, stratify=y)

    if args.weighted_loss:
        weights = torch.FloatTensor(1. / occurrences).to(args.device)
    else:
        weights = None

    writer = SummaryWriter()
    log_dir = writer.log_dir
    out_log = {}
    out_log["args"] = args.__dict__

    min_train_loss, max_train_f1_score, min_val_loss, max_val_f1_score = perform_validation(args, dataset, train_ids, y_train,
                                                                                            weights, input_dim,
                                                                                            output_dim, writer,
                                                                                            log_dir)
    out_log["min_train_loss"] = min_train_loss
    out_log["max_train_f1_score"] = max_train_f1_score
    out_log["min_val_loss"] = min_val_loss
    out_log["max_val_f1_score"] = max_val_f1_score

    with open("{}/log_file.json".format(log_dir), "w") as fp:
        json.dump(out_log, fp)


if __name__ == '__main__':
    main()
