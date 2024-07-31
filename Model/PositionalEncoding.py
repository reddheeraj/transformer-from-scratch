import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len

        # Creating a matrix of shape (seq_len, d_model); basically for positional encoding
        pe = torch.zeros(seq_len, d_model)
        # Creating a vector of shape (seq_len)
        position = torch.arange(0, seq_len).unsqueeze(1) # tensor of shape (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Sin is applied to even positions
        pe[:, 0::2] = torch.sin(position * div_term)

        # Cos is applied to odd positions
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # tensor of shape (1, seq_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # Makes this particular tensor not needed for learning
        return self.dropout(x)