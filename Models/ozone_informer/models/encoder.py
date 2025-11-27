import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None):
        # x: [B, C, R, T, D]
        B, C, R, T, D = x.shape

        # Flatten for temporal attention: (B*CR, T, D)
        x = x.view(B * C * R, T, D)

        # AttentionLayer expects (queries, keys, values)
        out, attn = self.attn_layer(x, x, x, attn_mask=attn_mask)  # [B*CR, T, D]

        # Restore shape
        out = out.view(B, C, R, T, D)
        return out, attn


class SpatialAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None):
        # x: [B, C, R, T, D]
        B, C, R, T, D = x.shape

        # Flatten for spatial attention: (B*T, C*R, D)
        x = x.permute(0, 3, 1, 2, 4).contiguous()  # [B, T, C, R, D]
        x = x.view(B * T, C * R, D)

        # AttentionLayer expects (queries, keys, values)
        out, attn = self.attn_layer(x, x, x, attn_mask=attn_mask)  # [B*T, C*R, D]

        # Restore shape
        out = out.view(B, T, C, R, D)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        return out, attn


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super().__init__()
        padding = 1 if torch.__version__>='1.5.0' else 2
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=padding,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1,2)
        return x


class TemporalDistill(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = ConvLayer(d_model)

    def forward(self, x):
        # x: [B, C, R, T, D]
        B, C, R, T, D = x.shape
        x = x.view(B * C * R, T, D)
        x = self.conv(x)   # ConvLayer expects (B,L,D)
        T_new = x.shape[1]
        x = x.view(B, C, R, T_new, D)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, temporal_attn, spatial_attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4*d_model
        # temporal and spatial attention
        self.temporal_attn = temporal_attn
        self.spatial_attn = spatial_attn

        # fusion after concat
        self.fusion = nn.Linear(2 * d_model, d_model)

        # feed-forward network
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # normalization, dropout, activation
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # x: [B, C, R, T, D]
        # temporal + spatial attention
        x_t, attn_t = self.temporal_attn(x, attn_mask=attn_mask)   # [B, C, R, T, D]
        x_s, attn_s = self.spatial_attn(x, attn_mask=attn_mask)    # [B, C, R, T, D]

        # concat + fusion
        new_x = torch.cat([x_t, x_s], dim=-1)   # [B, C, R, T, 2D]
        new_x = self.fusion(new_x)              # [B, C, R, T, D]

        # residual + norm
        x = x + self.dropout(new_x)             # [B, C, R, T, D]
        y = self.norm1(x)

        # feed-forward (applied per gridpoint/time)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x + y), (attn_t, attn_s)


class Encoder(nn.Module):
    def __init__(self, attn_layers, distill_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.distill_layers = nn.ModuleList(distill_layers) if distill_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: [B, C, R, T, D]
        attns = []
        if self.distill_layers is not None:
            for attn_layer, distill_layer in zip(self.attn_layers, self.distill_layers):
                # attention
                x, attn = attn_layer(x, attn_mask=attn_mask)   # [B, C, R, T, D]
                # temporal distill
                x = distill_layer(x)                           # [B, C, R, T', D]
                attns.append(attn)
            # last attention without distill
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            # apply norm to the last dimension D
            B, C, R, T, D = x.shape
            x = self.norm(x.view(B * C * R * T, D)).view(B, C, R, T, D)

        return x, attns



class EncoderStack(nn.Module):
    def __init__(self, encoders, inp_lens):
        super().__init__()
        self.encoders = nn.ModuleList(encoders)
        self.inp_lens = inp_lens

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns
