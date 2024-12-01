import torch
from torch import nn


class CustomBCEWithLogitsLoss(nn.Module):
    def __init__(self,):
        super(CustomBCEWithLogitsLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid function to the input logits
        sigmoid_input = torch.sigmoid(input)

        loss = target * torch.log(sigmoid_input)
        loss += (1-target) * torch.log(1-sigmoid_input)
        loss = -loss.mean()
        return loss
