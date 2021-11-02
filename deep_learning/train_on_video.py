import torch
import numpy as np
from deep_learning.utils import get_lr_optimizer, get_lr_scheduler, get_loss, get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, matthews_corrcoef
from tqdm import tqdm
import math
from sklearn.utils import shuffle
import wandb


def run_once(args, model, dataset, ids, criterion, optimizer, is_train=False):
    losses = []
    outs = []
    gt = []
    num_batches = math.ceil(len(ids)/args.batch_size)
    ids_copy = ids.copy()
    if is_train:
        ids_copy = shuffle(ids_copy)

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

        wdb_log = {
            "train/loss": np.mean(train_losses),
            "train/micro_f1": train_f1_score,
            "train/macro_f1": f1_score(train_gt, train_outs, average="macro"),
            "train/accuracy": accuracy_score(train_gt, train_outs),
            "train/balanced_accuracy": balanced_accuracy_score(train_gt, train_outs),
            "train/mcc": matthews_corrcoef(train_gt, train_outs),
            "val/loss": np.mean(val_losses),
            "val/micro_f1": val_f1_score,
            "val/macro_f1": f1_score(val_gt, val_outs, average="macro"),
            "val/accuracy": accuracy_score(val_gt, val_outs),
            "val/balanced_accuracy": balanced_accuracy_score(val_gt, val_outs),
            "val/mcc": matthews_corrcoef(val_gt, val_outs)
        }

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


def perform_validation(args, dataset, train_ids, val_ids, weights, input_dim, output_dim, writer, log_dir):

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, dataset, train_ids, val_ids,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")

    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max