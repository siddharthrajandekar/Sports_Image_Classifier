import torch

def accuracy(y_pred, y_true):
    _, predicted = torch.max(y_pred.data, 1)
    correct = (predicted == y_true).sum().item()
    total = y_true.size(0)
    return correct / total
