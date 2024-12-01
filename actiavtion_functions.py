import torch
from math import e


class CustomReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):

        ctx.save_for_backward(input)

        return input.clamp(min=0)  # This replaces ReLU for simplicity

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):

        input, = ctx.saved_tensors

        grad_input = grad_output.clone()
        grad_input[input < 0] = 0  # Grad is 0 for all input <= 0
        return grad_input


class CustomSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Save the input for the backward pass
        ctx.save_for_backward(input)
        # Apply Sigmoid: 1 / (1 + e^(-x))
        return 1 / (1 + torch.exp(-input))
        # return torch.sigmoid(input)  # This replaces ReLU for simplicity

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve the saved input
        input, = ctx.saved_tensors
        # Compute the gradient of Sigmoid = sigmoid(x)*(1-sigmoid(x))

        grad_input = grad_output.clone()
        sigmoid_output = 1 / (1 + torch.exp(-input))
        grad_input = grad_input * sigmoid_output * (1 - sigmoid_output)

        return grad_input
    
    
class CustomSoftMax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        # Apply softmax directly without any numerical stability adjustment
        exp_input = torch.exp(input)  # Exponentiate the input values
        softmax_output = exp_input / exp_input.sum(dim=-1, keepdim=True)  # Normalize by sum of exponentials
        
        # Save the softmax output for backward pass
        ctx.save_for_backward(softmax_output, input)
        
        return softmax_output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Retrieve saved tensors from the forward pass
        softmax_output, input = ctx.saved_tensors
        
        # Compute the Jacobian matrix of the softmax function
        grad_input = grad_output * softmax_output * (1 - softmax_output)
        
        if input.dim() > 1:
            batch_size = input.size(0)
            grad_input = grad_input.view(batch_size, -1)
        
        return grad_input
