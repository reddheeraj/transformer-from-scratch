import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    """
    Class to create input embeddings for the transformer model
    Returns the embedding value multiplied by sqrt(d_model) as mentioned in the paper "Attention is All You Need"
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)