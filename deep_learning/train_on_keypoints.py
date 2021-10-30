import torch
import numpy as np
import wandb
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from deep_learning.utils import get_loss, get_lr_optimizer, get_lr_scheduler, get_model, seed_worker


def run_once(args, model, loader, criterion, optimizer, is_train=False):
    losses = []
    outs = []
    gt = []
    torch.set_grad_enabled(is_train)
    for idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        if is_train:
            optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output.squeeze(), labels.squeeze())
        gt.append(labels.squeeze().detach().cpu().numpy())
        outs.append(torch.argmax(torch.nn.functional.softmax(output.detach().cpu(), dim=1), dim=1).numpy())
        losses.append(loss.item())
        if is_train:
            loss.backward()
            optimizer.step()
    torch.set_grad_enabled(not is_train)

    return losses, np.concatenate(outs), np.concatenate(gt)


def train_n_epochs(args, train_dataset, val_dataset, weights, input_dim, output_dim, writer, log_dir, tag):
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                              num_workers=6, drop_last=True, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size,
                            num_workers=2, drop_last=False, worker_init_fn=seed_worker)

    criterion = get_loss(weights)
    model = get_model(args, input_dim, output_dim).to(args.device)
    optimizer = get_lr_optimizer(args, model)
    scheduler = get_lr_scheduler(args, optimizer)

    train_loss_min = 1000
    train_f1_max = -1
    valid_loss_min = 1000
    valid_f1_max = -1
    for i in tqdm(range(args.epochs)):
        train_losses, train_outs, train_gt = run_once(args, model, train_loader, criterion, optimizer, is_train=True)
        train_f1_score = f1_score(train_gt, train_outs, average="micro")
        model.eval()
        val_losses, val_outs, val_gt = run_once(args, model, val_loader, criterion, None)
        val_f1_score = f1_score(val_gt, val_outs, average="micro")
        if writer:
            writer.add_scalar("Loss{}/train".format(tag), np.mean(train_losses), i)
            writer.add_scalar("F1{}/train".format(tag), train_f1_score, i)
            writer.add_scalar("Loss{}/val".format(tag), np.mean(val_losses), i)
            writer.add_scalar("F1{}/val".format(tag), val_f1_score, i)


        wdb_log = {
                "train/loss": np.mean(train_losses).item(),
                "train/f1": train_f1_score,
                "val/loss": np.mean(val_losses).item(),
                "val/f1": val_f1_score
        }
        wandb.log(wdb_log, step=i)

        scheduler.step()
        model.train()
        if np.mean(val_losses) < valid_loss_min:
            valid_loss_min = np.mean(val_losses)
        if val_f1_score > valid_f1_max:
            valid_f1_max = val_f1_score
        train_loss_min = min(train_loss_min, np.mean(train_losses))
        train_f1_max = max(train_f1_max, train_f1_score)
    # if tag == "":
    #     torch.save(model.state_dict(), '{}/state_dict_final.pt'.format(log_dir))
    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max


def perform_validation(args, X, y, weights, input_dim, output_dim, writer, log_dir):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=15/85, random_state=args.seed, shuffle=True, stratify=y)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train),
                                                   torch.from_numpy(y_train))
    val_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_val),
                                                 torch.from_numpy(y_val))

    train_loss_min, train_f1_max, valid_loss_min, valid_f1_max = train_n_epochs(args, train_dataset, val_dataset,
                                                                                weights, input_dim, output_dim,
                                                                                writer, log_dir, tag="")

    return train_loss_min, train_f1_max, valid_loss_min, valid_f1_max
