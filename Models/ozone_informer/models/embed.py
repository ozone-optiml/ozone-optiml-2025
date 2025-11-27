import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Temporal embedding (sin/cos PE broadcast)
class TemporalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        tem_d_model = d_model // 2
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, tem_d_model, 2) * -(math.log(10000.0) / tem_d_model))
        pe[:, 0:tem_d_model:2] = torch.sin(position * div_term)
        pe[:, 1:tem_d_model:2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1,max_len,d]

    def forward(self, B, C, R, T):
        pe = self.pe[:, :T, :]             # [1,T,D]
        pe = pe.unsqueeze(1).unsqueeze(2)  # [1,1,1,T,D]
        return pe.expand(B, C, R, T, -1)   # [B,C,R,T,D]


class SpatialEmbedding(nn.Module):
    def __init__(self, d_model, c_max=100, r_max=100):
        super().__init__()
        assert d_model % 4 == 0, "d_model must be divisible by 4"
        spa_d_model = d_model // 4

        pe_c = torch.zeros(c_max, d_model)
        pe_r = torch.zeros(r_max, d_model)

        col_pos = torch.arange(0, c_max).unsqueeze(1)  # [c_max,1]
        row_pos = torch.arange(0, r_max).unsqueeze(1)  # [r_max,1]

        div_term = torch.exp(torch.arange(0, spa_d_model, 2) * -(math.log(10000.0) / spa_d_model))
        assert len(div_term) * 2 == spa_d_model, "div_term length mismatch"
        pe_c[:, spa_d_model*2:spa_d_model*3:2] = torch.sin(col_pos * div_term)
        pe_c[:, spa_d_model*2+1:spa_d_model*3:2] = torch.cos(col_pos * div_term)

        pe_r[:, spa_d_model*3::2] = torch.sin(row_pos * div_term) # [r_max, spa_d_model]/2
        pe_r[:, spa_d_model*3+1::2] = torch.cos(row_pos * div_term)

        self.register_buffer("pe_c", pe_c.unsqueeze(0))  # [1,c_max,D]
        self.register_buffer("pe_r", pe_r.unsqueeze(0))  # [1,r_max,D]

    def forward(self, B, C, R, T):
        col_emb = self.pe_c[:, :C, :]  # [1,C,D]
        row_emb = self.pe_r[:, :R, :]  # [1,R,D]

        col_emb = col_emb.unsqueeze(2).unsqueeze(3).expand(B, C, R, T, -1)  # [B,C,R,T,D]
        row_emb = row_emb.unsqueeze(1).unsqueeze(3).expand(B, C, R, T, -1)  # [B,C,R,T,D]

        return col_emb + row_emb


# Variable embedding (project V→D)
class VariableEmbedding(nn.Module):
    def __init__(self, in_features, d_model):
        super().__init__()
        self.proj = nn.Linear(in_features, d_model)

    def forward(self, x):
        # x: [B, C, R, T, V]
        B, C, R, T, V = x.shape
        # x = x.view(B * C * R * T, V)
        x = x.reshape(B * C * R * T, V)
        x = self.proj(x)  # [B*C*R*T, D]
        x = x.view(B, C, R, T, -1)
        return x

# Variable + Temporal + Spatial Embedding
class GridDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super().__init__()
        self.var_emb = VariableEmbedding(c_in, d_model)
        self.temp_emb = TemporalPositionalEmbedding(d_model)
        self.spat_emb = SpatialEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark=None):
        B, C, R, T, V = x.shape
        var = self.var_emb(x)  # [B, C, R, T, D]
        temp = self.temp_emb(B, C, R, T)
        spat = self.spat_emb(B, C, R, T)
        return self.dropout(var + temp + spat)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                    kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1,2)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super().__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super().__init__()

        minute_size = 4; hour_size = 24
        weekday_size = 7; day_size = 32; month_size = 13

        Embed = FixedEmbedding if embed_type=='fixed' else nn.Embedding
        if freq=='t':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)
    
    def forward(self, x):
        x = x.long()
        
        minute_x = self.minute_embed(x[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:,:,3])
        weekday_x = self.weekday_embed(x[:,:,2])
        day_x = self.day_embed(x[:,:,1])
        month_x = self.month_embed(x[:,:,0])
        
        return hour_x + weekday_x + day_x + month_x + minute_x

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super().__init__()

        freq_map = {'h':4, 't':5, 's':6, 'm':1, 'a':1, 'w':2, 'd':3, 'b':3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)
    
    def forward(self, x):
        return self.embed(x)

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super().__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) if embed_type!='timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)