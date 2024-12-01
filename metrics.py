import torch 
from actiavtion_functions import CustomSigmoid

def accuracy(preds:torch.Tensor, labels:torch.Tensor):
    preds = 1/(1 + torch.exp(-preds)) #torch.sigmoid(preds)  # Convert logits to probabilities
    preds = (preds >= 0.5).float()  # Apply threshold
    correct = (preds == labels).sum().item()
    return correct / len(labels)

    