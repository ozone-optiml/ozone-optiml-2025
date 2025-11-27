import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCrossAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, memory, attn_mask=None):
        # x: [B, C, R, T_dec, D]
        # memory: [B, C, R, T_enc, D]
        B, C, R, T_dec, D = x.shape
        _, _, _, T_enc, _ = memory.shape

        # Flatten for temporal cross-attention
        q = x.view(B * C * R, T_dec, D)
        kv = memory.view(B * C * R, T_enc, D)

        # Attention: query from decoder, key/value from encoder
        out, attn = self.attn_layer(q, kv, kv, attn_mask=attn_mask)

        # Restore shape
        out = out.view(B, C, R, T_dec, D)
        return out, attn


class SpatialCrossAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, memory, attn_mask=None):
        # x: [B, C, R, T, D]
        # memory: [B, C, R, T_enc, D]
        B, C, R, T_dec, D = x.shape
        _, _, _, T_enc, _ = memory.shape

        # Flatten decoder: (B*T_dec, C*R, D)
        q = x.permute(0, 3, 1, 2, 4).contiguous()
        q = q.view(B * T_dec, C * R, D)

        # Flatten memory: (B*T_enc, C*R, D)
        kv = memory.permute(0, 3, 1, 2, 4).contiguous()
        kv = kv.view(B * T_enc, C * R, D)

        # Attention
        out, attn = self.attn_layer(q, kv, kv, attn_mask=attn_mask)

        # Restore shape
        out = out.view(B, T_dec, C, R, D)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        return out, attn


class DecoderLayer(nn.Module):
    def __init__(self, temporal_self_attn, spatial_self_attn,
                 temporal_cross_attn, spatial_cross_attn,
                 d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # self-attn (temporal + spatial)
        self.temporal_self_attn = temporal_self_attn
        self.spatial_self_attn = spatial_self_attn
        self.self_fusion = nn.Linear(2 * d_model, d_model)

        # cross-attn (temporal + spatial)
        self.temporal_cross_attn = temporal_cross_attn
        self.spatial_cross_attn = spatial_cross_attn
        self.cross_fusion = nn.Linear(2 * d_model, d_model)

        # feed-forward
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        # norms
        self.norm1 = nn.LayerNorm(d_model)  # after self-attn
        self.norm2 = nn.LayerNorm(d_model)  # after cross-attn
        self.norm3 = nn.LayerNorm(d_model)  # after FFN
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, memory, x_mask=None, cross_mask=None):
        # x: [B, C, R, T, D], memory: [B, C, R, T_enc, D]

        # 1. self-attn
        x_t, attn_t_self = self.temporal_self_attn(x, attn_mask=x_mask)
        x_s, attn_s_self = self.spatial_self_attn(x, attn_mask=x_mask)

        new_x = torch.cat([x_t, x_s], dim=-1)
        new_x = self.self_fusion(new_x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 2. cross-attn
        x_t, attn_t_cross = self.temporal_cross_attn(x, memory, attn_mask=cross_mask)
        x_s, attn_s_cross = self.spatial_cross_attn(x, memory, attn_mask=cross_mask)

        new_x = torch.cat([x_t, x_s], dim=-1)
        new_x = self.cross_fusion(new_x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # 3. FFN
        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))
        x = self.norm3(x + y)

        return x, (attn_t_self, attn_s_self, attn_t_cross, attn_s_cross)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        attns = []

        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns