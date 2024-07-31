import math
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # trainable parameter, Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True) # Usually mean cancels the batch dimension, but we want to keep it
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean) / (std+self.eps) + self.bias