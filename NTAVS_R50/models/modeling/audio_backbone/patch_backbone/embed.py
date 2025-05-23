import torch
import torch.nn as nn

import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, start_pos=0, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.start_pos = start_pos

    def forward(self, x):
        pos = self.pe[:, self.start_pos:x.size(1) + self.start_pos]
        return pos


class TokenEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding, self).__init__()
        # self.tokenConv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, padding=0)
        self.tokenConv = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.tokenConv(x)
        return x
    
class TokenEmbedding_2(nn.Module):
    def __init__(self, d_model):
        super(TokenEmbedding_2, self).__init__()
        # self.tokenConv = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, padding=0)
        self.tokenConv = nn.Linear(d_model, 2 * d_model)

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, d_model, start_pos=0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(d_model=d_model//4)
        self.position_embedding = PositionalEmbedding(d_model=d_model//4, start_pos=start_pos)

    def forward(self, x, pos_only=False):
        pos = self.position_embedding(x)
        if pos_only:
            return pos
        else:
            x = self.value_embedding(x) + pos
            return x


class DataEmbedding_2(nn.Module):
    def __init__(self, d_model, start_pos=0):
        super(DataEmbedding_2, self).__init__()

        self.value_embedding = TokenEmbedding_2(d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=2*d_model, start_pos=start_pos)

    def forward(self, x, pos_only=False):
        pos = self.position_embedding(x)
        if pos_only:
            return pos
        else:
            x = self.value_embedding(x) + pos
            return x