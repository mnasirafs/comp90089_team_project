import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def compute_batch_accuracy(output, target):
    """Computes the accuracy for a batch"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum()

        return correct * 100.0 / batch_size


def train(model, device, data_loader, criterion, optimizer, epoch, print_freq=10):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()

    model.train()

    end = time.time()
    for i, (data, target) in enumerate(data_loader):
        # Unpack data into sequences and lengths
        sequences, lengths = data

        # Move tensors to the appropriate device
        sequences = sequences.to(device)
        lengths = lengths.to(device)
        target = target.to(device)

        data_time.update(time.time() - end)

        optimizer.zero_grad()
        output = model(sequences, lengths)  # Pass both sequences and lengths
        loss = criterion(output, target)

        assert not np.isnan(loss.item()), 'Model diverged with loss = NaN'

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses.update(loss.item(), target.size(0))
        accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

        if i % print_freq == 0:
            print(f'Epoch: [{epoch}][{i}/{len(data_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})')

    return losses.avg, accuracy.avg

def evaluate(model, device, data_loader, criterion, print_freq=10):
    batch_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    results = []

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (data, target) in enumerate(data_loader):
            sequences, lengths = data  # Unpack data

            sequences = sequences.to(device)
            lengths = lengths.to(device)
            target = target.to(device)

            output = model(sequences, lengths)  # Pass both sequences and lengths
            loss = criterion(output, target)

            batch_time.update(time.time() - end)
            end = time.time()

            losses.update(loss.item(), target.size(0))
            accuracy.update(compute_batch_accuracy(output, target).item(), target.size(0))

            y_true = target.detach().cpu().numpy().tolist()
            y_pred = output.detach().cpu().max(1)[1].numpy().tolist()
            results.extend(list(zip(y_true, y_pred)))

            if i % print_freq == 0:
                print(f'Test: [{i}/{len(data_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})')

    return losses.avg, accuracy.avg, results

def best_evaluate(model, device, data_loader):
    y_true, y_pred, y_prob = [], [], []

    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            sequences, lengths = data  # Unpack data

            sequences = sequences.to(device)
            lengths = lengths.to(device)
            target = target.to(device)

            output = model(sequences, lengths)
            output_score = torch.softmax(output, dim=1)

            y_true.extend(target.detach().cpu().numpy().tolist())
            y_pred.extend(output.detach().cpu().max(1)[1].numpy().tolist())
            y_prob.extend(output_score.detach().cpu().select(1, 1).numpy().tolist())

    print(f'Test Accuracy: {accuracy_score(y_true, y_pred):.4f}')
    print(f'Test Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Test Recall: {recall_score(y_true, y_pred):.4f}')
    print(f'Test F1-score: {f1_score(y_true, y_pred):.4f}')
    print(f'Test ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}')
    print(f'Test MCC: {matthews_corrcoef(y_true, y_pred):.4f}')

    y_true = []
    y_pred = []
    y_prob = []
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(data_loader):

            if isinstance(input, tuple):
                input = tuple([e.to(device) if type(e) == torch.Tensor else e for e in input])
            else:
                input = input.to(device)

            output = model(input)
            output_score = torch.softmax(model(input), 1)

            y_true.extend(target.detach().to('cpu').numpy().tolist())
            y_pred.extend(output.detach().to('cpu').max(1)[1].numpy().tolist())
            y_prob.extend(output_score.detach().to('cpu').select(1, 1).numpy().tolist())

    print('Test Accuracy: ' + str(accuracy_score(y_true, y_pred)) + '\t')
    print('Test Precision: ' + str(precision_score(y_true, y_pred)) + '\t')
    print('Test Recall: ' + str(recall_score(y_true, y_pred)) + '\t')
    print('Test F1-score: ' + str(f1_score(y_true, y_pred)) + '\t')
    print('Test ROC-AUC: ' + str(roc_auc_score(y_true, y_prob)) + '\t')
    print('Test MCC: ' + str(matthews_corrcoef(y_true, y_pred)) + '\t')