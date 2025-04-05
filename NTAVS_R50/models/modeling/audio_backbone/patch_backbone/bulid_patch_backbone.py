import numpy as np
import torch
import torch.nn as nn
import math
from .embed import DataEmbedding, DataEmbedding_2

class Atten(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Atten, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, V, D = queries.shape
        
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bvd,bsd->bvs", queries, keys)  

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  
        out = torch.einsum("bvs,bsd->bvd", attn, values)  

        return self.out_projection(out)  
    
class Atten_L(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Atten_L, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, L, C, D = queries.shape
        
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("blcd,blsd->blcs", queries, keys)  

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  
        out = torch.einsum("blcs,blsd->blcd", attn, values)  

        return self.out_projection(out)  

class Atten_C(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Atten_C, self).__init__()
        self.query_projection = nn.Linear(d_model, d_model)
        self.kv_projection = nn.Linear(d_model, d_model)

        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        B, V, D = queries.shape
        
        scale = 1. / math.sqrt(D)

        queries = self.query_projection(queries)
        keys = self.kv_projection(keys)
        values = self.kv_projection(values)

        scores = torch.einsum("bvd,bsd->bvs", queries, keys)  

        attn = self.dropout(torch.softmax(scale * scores, dim=-1))  
        out = torch.einsum("bvs,bsd->bvd", attn, values)  

        return self.out_projection(out)  
    
class Patch_Audio(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        # self.level = 3
        self.d_model = cfg.MODEL.AUDIO.DMODEL # 256
        self.dropout_rate = cfg.MODEL.AUDIO.DROPOUT # 0.1
        self.embedding = DataEmbedding(self.d_model)

        self.attention_1 = Atten(self.d_model//4, self.dropout_rate)

        self.attention_2 = Atten(self.d_model//2, self.dropout_rate)

        self.attention_3 = Atten(self.d_model, self.dropout_rate)

        self.activation = nn.GELU()
        self.norm1_1 = nn.LayerNorm(self.d_model//4)
        self.norm1_2 = nn.LayerNorm(self.d_model//4)

        self.norm2_1 = nn.LayerNorm(self.d_model//2)
        self.norm2_2 = nn.LayerNorm(self.d_model//2)

        self.norm3_1 = nn.LayerNorm(self.d_model)
        self.norm3_2 = nn.LayerNorm(self.d_model)
        
        self.dropout = nn.Dropout(self.dropout_rate)

        self.linear1_1 = nn.Linear(self.d_model//4, self.d_model)
        self.linear1_2 = nn.Linear(self.d_model, self.d_model//4)

        self.linear2_1 = nn.Linear(self.d_model//2, self.d_model*2)
        self.linear2_2 = nn.Linear(self.d_model*2, self.d_model//2)

        self.linear3_1 = nn.Linear(self.d_model, self.d_model*4)
        self.linear3_2 = nn.Linear(self.d_model*4, self.d_model)

        self.linear_sec = nn.Linear(self.d_model//2, self.d_model//2)
        self.linear_thir = nn.Linear(self.d_model, self.d_model)

    # def attention(self, x):

    def forward(self, x):
        b, l, c = x.shape
        # [bs*5, 96, 64]
        x = self.embedding(x)

        attn_1 = self.attention_1(x, x, x)
        y = x = self.norm1_1(x + self.dropout(attn_1))
        x = self.activation(self.linear1_1(x))
        x = self.norm1_2(y + self.dropout(self.linear1_2(x)))

        x = x.contiguous().view(b, l//2, 2, c).view(b, l//2, 2*c)
        x = self.linear_sec(x)
        attn_2 = self.attention_2(x, x, x)
        y = x = self.norm2_1(x + self.dropout(attn_2))
        x = self.activation(self.linear2_1(x))
        x = self.norm2_2(y + self.dropout(self.linear2_2(x)))

        x = x.contiguous().view(b, l//4, 2, 2*c).view(b, l//4, 4*c)
        x = self.linear_thir(x)
        attn_3 = self.attention_3(x, x, x)
        y = x = self.norm3_1(x + self.dropout(attn_3))
        x = self.activation(self.linear3_1(x))
        x = self.norm3_2(y + self.dropout(self.linear3_2(x)))
        # [bs*5, 24, 4 * 64]
        return x

class Add_and_FFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.dropout = nn.Dropout(0.1)   
        self.activation = nn.GELU()
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)

        self.linear1_1 = nn.Linear(d_model, d_model * 4)
        self.linear1_2 = nn.Linear(d_model * 4, d_model)

    def forward(self, attn, x):
        y = x = self.norm1_1(x + self.dropout(attn))
        x = self.activation(self.linear1_1(x))
        x = self.norm1_2(y + self.dropout(self.linear1_2(x)))

        return x


class Patch_Audio_2_dimensional(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        # self.level = 3
        self.d_model = cfg.MODEL.AUDIO.DMODEL # 256
        self.dropout_rate = cfg.MODEL.AUDIO.DROPOUT # 0.1
        dim1, dim2 = 96, 64
        embed_dim = dim1 // 16 * dim2 // 16
        self.embedding = DataEmbedding_2(embed_dim)
        
        self.attention_1_1 = Atten_L(embed_dim * 2, self.dropout_rate)
        self.attention_1_2 = Atten_L(embed_dim * 2, self.dropout_rate)

        self.ffn_1_1 = Add_and_FFN(embed_dim * 2)
        self.ffn_1_2 = Add_and_FFN(embed_dim * 2)

        # self.down_linear_1 = nn.Linear(embed_dim * 4, embed_dim * 2)

        self.attention_2_1 = Atten_L(embed_dim*4 * 2, self.dropout_rate)
        self.attention_2_2 = Atten_L(embed_dim*4 * 2, self.dropout_rate)

        self.ffn_2_1 = Add_and_FFN(embed_dim*4 * 2)
        self.ffn_2_2 = Add_and_FFN(embed_dim*4 * 2)

        # self.down_linear_2 = nn.Linear(embed_dim * 8, embed_dim * 4)

        self.attention_3_1 = Atten_L(embed_dim*16 * 2, self.dropout_rate)
        self.attention_3_2 = Atten_L(embed_dim*16 * 2, self.dropout_rate)

        self.ffn_3_1 = Add_and_FFN(embed_dim*16 * 2)
        self.ffn_3_2 = Add_and_FFN(embed_dim*16 * 2)

        self.post_linear = nn.Sequential(
            nn.Linear(384 * 2 * 4 * 4, 4096), nn.GELU(), nn.Linear(4096, 4096), nn.GELU(), nn.Linear(4096, 128), nn.GELU())

    def forward(self, x):
        b, l, c = x.shape
        # [bs*5, 96, 64]
        x = x.contiguous().view(b, l//6, 6, c//4, 4).permute(0, 1, 3, 2, 4).contiguous().view(b, l//6, c//4, 24) #b, 16, 16, 48
        x = x.contiguous().view(b, 256, 24)
        x = self.embedding(x) #b, 16, 16, 24
        x = x.contiguous().view(b, 16, 16, 48)

        attn_1_1 = self.attention_1_1(x, x, x)
        x = self.ffn_1_1(attn_1_1, x)
        x = x.transpose(1, 2)

        attn_1_2 = self.attention_1_2(x, x, x)
        x = self.ffn_1_2(attn_1_2, x)
        x = x.transpose(1, 2)

        x = x.contiguous().view(b, l//12, 2, c//8, 2, 48).permute(0, 1, 3, 2, 4, 5).contiguous().view(b, l//12, c//8, 192) #b, 8, 8, 96
        # x = self.down_linear_1(x) #b, 8, 8, 48 

        attn_2_1 = self.attention_2_1(x, x, x)
        x = self.ffn_2_1(attn_2_1, x)
        x = x.transpose(1, 2)

        attn_2_2 = self.attention_2_2(x, x, x)
        x = self.ffn_2_2(attn_2_2, x)
        x = x.transpose(1, 2)

        x = x.contiguous().view(b, l//24, 2, c//16, 2, 192).permute(0, 1, 3, 2, 4, 5).contiguous().view(b, l//24, c//16, 768) #b, 4, 4, 384
        # x = self.down_linear_2(x) #b, 4, 4, 96

        attn_3_1 = self.attention_3_1(x, x, x)
        x = self.ffn_3_1(attn_3_1, x)
        x = x.transpose(1, 2)

        attn_3_2 = self.attention_3_2(x, x, x)
        x = self.ffn_3_2(attn_3_2, x)
        x = x.transpose(1, 2)

        x = x.contiguous().view(x.size(0), -1)

        x = self.post_linear(x)

        # print('x', x.shape)

        return x
  