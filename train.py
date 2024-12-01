import torch
from torch import nn

from time import time
import sys

from metrics import accuracy
from result_worker import write_details

from config import num_epochs, result_full_path


def train_epoch(model: nn.Module, criterion, optimizer: torch.optim.Optimizer, train_dataloader, device: str):
    model.train()

    start_time = time()
    epoch_acc, epoch_loss = 0, 0
    sm = 0
    for iter, (batch, batch_labels) in enumerate(train_dataloader):
        batch = batch.to(device)
        batch_labels = batch_labels.to(device)

        preds = model(batch)
        preds = torch.flatten(preds)

        batch_labels = batch_labels.float()

        loss: torch.Tensor = criterion(preds, batch_labels)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_acc = accuracy(preds, batch_labels)
        sm += len(batch_labels)
        epoch_acc += iter_acc*len(batch_labels)

        print(f'{iter_acc:.3f}', len(batch_labels))
        sys.stdout.write("\033[F")
        sys.stdout.flush()

    end_time = time()
    epoch_loss /= (iter + 1)
    epoch_acc /= sm
    return epoch_acc, epoch_loss, end_time-start_time


def eval_epoch(model: nn.Module, criterion, test_dataloader, device):
    model.eval()

    start_time = time()
    epoch_acc = 0
    sm = 0
    for iter, (batch, batch_labels) in enumerate(test_dataloader):
        batch = batch.to(device)
        batch_labels = batch_labels.to(device)

        preds = model(batch)
        preds = torch.flatten(preds)

        batch_labels = batch_labels.float()

        iter_acc = accuracy(preds, batch_labels)
        sm += len(batch_labels)
        epoch_acc += iter_acc*len(batch_labels)

    end_time = time()
    epoch_acc /= sm
    return epoch_acc,  end_time-start_time


def train(model: nn.Module, criterion, optimizer: torch.optim.Optimizer, train_dataloader, test_data_loader, device):

    print("             Train Accuracy\t\tTest Accuracy\t\tLoss\t\tTime")
    write_details(result_full_path, "Epoch", " Train Acc",
                  " Test Acc", " BCE", " Duration")

    for epoch in range(1, num_epochs+1):

        train_epoch_acc, epoch_loss, train_duration = train_epoch(
            model, criterion, optimizer, train_dataloader, device)

        eval_epoch_acc, eval_duration = eval_epoch(
            model, criterion, test_data_loader, device)

        dur = train_duration+eval_duration

        print(
            f"Epoch {epoch} :\t{train_epoch_acc*100:.2f}\t\t\t{eval_epoch_acc*100:.2f}\t\t\t{epoch_loss:.4f}\t\t{dur:.1f}")

        write_details(result_full_path, epoch, f'{train_epoch_acc*100: .2f}',
                      f'{eval_epoch_acc*100: .2f}', f'{epoch_loss: .4f}', f'{dur: .1f}')
