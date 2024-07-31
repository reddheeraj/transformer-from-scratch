import math
import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) # W2 and b2
    
    def forward(self, x):
        # goes from (Batch, seq_len_d_model) to (Batch, seq_len, d_ff) and back to (Batch, seq_len, d_model)
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))