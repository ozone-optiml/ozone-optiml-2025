import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class TemporalCrossAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, memory, attn_mask=None, rope_q=None, rope_k=None):
        # x: [B, C, R, T_dec, D]
        # memory: [B, C, R, T_enc, D]
        B, C, R, T_dec, D = x.shape
        _, _, _, T_enc, _ = memory.shape

        # Flatten for temporal cross-attention
        q = x.view(B * C * R, T_dec, D)
        kv = memory.view(B * C * R, T_enc, D)
        
        rope_tuple = None
        if rope_q is not None and rope_k is not None:
             cos_q, sin_q = rope_q
             cos_k, sin_k = rope_k
             
             cos_q = cos_q.view(B * C * R, T_dec, -1)
             sin_q = sin_q.view(B * C * R, T_dec, -1)
             cos_k = cos_k.view(B * C * R, T_enc, -1)
             sin_k = sin_k.view(B * C * R, T_enc, -1)
             
             rope_tuple = (cos_q, sin_q, cos_k, sin_k)

        # Attention: query from decoder, key/value from encoder
        out, attn = self.attn_layer(q, kv, kv, attn_mask=attn_mask, rotary_pos_emb=rope_tuple)

        # Restore shape
        out = out.view(B, C, R, T_dec, D)
        return out, attn


class SpatialCrossAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, memory, attn_mask=None, rope_q=None, rope_k=None):
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
        
        rope_tuple = None
        if rope_q is not None and rope_k is not None:
             cos_q, sin_q = rope_q
             cos_k, sin_k = rope_k
             
             # Permute and flatten q rope
             cos_q = cos_q.permute(0, 3, 1, 2, 4).contiguous().view(B * T_dec, C * R, -1)
             sin_q = sin_q.permute(0, 3, 1, 2, 4).contiguous().view(B * T_dec, C * R, -1)
             
             # Permute and flatten k rope
             cos_k = cos_k.permute(0, 3, 1, 2, 4).contiguous().view(B * T_enc, C * R, -1)
             sin_k = sin_k.permute(0, 3, 1, 2, 4).contiguous().view(B * T_enc, C * R, -1)
             
             rope_tuple = (cos_q, sin_q, cos_k, sin_k)

        # Attention
        out, attn = self.attn_layer(q, kv, kv, attn_mask=attn_mask, rotary_pos_emb=rope_tuple)

        # Restore shape
        out = out.view(B, T_dec, C, R, D)
        out = out.permute(0, 2, 3, 1, 4).contiguous()
        return out, attn


# class DecoderLayer(nn.Module):
#     def __init__(self, temporal_self_attn, spatial_self_attn,
#                  temporal_cross_attn, spatial_cross_attn,
#                  d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
#         d_ff = d_ff or 4 * d_model

#         # self-attn (temporal + spatial)
#         self.temporal_self_attn = temporal_self_attn
#         self.spatial_self_attn = spatial_self_attn
#         self.self_fusion = nn.Linear(2 * d_model, d_model)

#         # cross-attn (temporal + spatial)
#         self.temporal_cross_attn = temporal_cross_attn
#         self.spatial_cross_attn = spatial_cross_attn
#         self.cross_fusion = nn.Linear(2 * d_model, d_model)

#         # feed-forward
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)

#         # norms
#         self.norm1 = nn.LayerNorm(d_model)  # after self-attn
#         self.norm2 = nn.LayerNorm(d_model)  # after cross-attn
#         self.norm3 = nn.LayerNorm(d_model)  # after FFN
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, memory, x_mask=None, cross_mask=None, x_rope=None, cross_rope=None):
#         # x: [B, C, R, T, D], memory: [B, C, R, T_enc, D]

#         # 1. self-attn
#         # Pass x_rope for both Q and K in self attention
#         x_t, attn_t_self = self.temporal_self_attn(x, attn_mask=x_mask, rope=x_rope)
#         x_s, attn_s_self = self.spatial_self_attn(x, attn_mask=x_mask, rope=x_rope)

#         new_x = torch.cat([x_t, x_s], dim=-1)
#         new_x = self.self_fusion(new_x)
#         x = x + self.dropout(new_x)
#         x = self.norm1(x)

#         # 2. cross-attn
#         # Pass x_rope for Q, cross_rope (from encoder) for K
#         x_t, attn_t_cross = self.temporal_cross_attn(x, memory, attn_mask=cross_mask, rope_q=x_rope, rope_k=cross_rope)
#         x_s, attn_s_cross = self.spatial_cross_attn(x, memory, attn_mask=cross_mask, rope_q=x_rope, rope_k=cross_rope)

#         new_x = torch.cat([x_t, x_s], dim=-1)
#         new_x = self.cross_fusion(new_x)
#         x = x + self.dropout(new_x)
#         x = self.norm2(x)

#         # 3. FFN
#         y = self.dropout(self.activation(self.linear1(x)))
#         y = self.dropout(self.linear2(y))
#         x = self.norm3(x + y)

#         return x, (attn_t_self, attn_s_self, attn_t_cross, attn_s_cross)

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
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def _forward_impl(self, x, memory, x_mask, cross_mask, x_rope, cross_rope):
        # 1. self-attn
        x_t, attn_t_self = self.temporal_self_attn(x, attn_mask=x_mask, rope=x_rope)
        x_s, attn_s_self = self.spatial_self_attn(x, attn_mask=x_mask, rope=x_rope)

        new_x = torch.cat([x_t, x_s], dim=-1)
        new_x = self.self_fusion(new_x)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # 2. cross-attn
        x_t, attn_t_cross = self.temporal_cross_attn(x, memory, attn_mask=cross_mask, rope_q=x_rope, rope_k=cross_rope)
        x_s, attn_s_cross = self.spatial_cross_attn(x, memory, attn_mask=cross_mask, rope_q=x_rope, rope_k=cross_rope)

        new_x = torch.cat([x_t, x_s], dim=-1)
        new_x = self.cross_fusion(new_x)
        x = x + self.dropout(new_x)
        x = self.norm2(x)

        # 3. FFN
        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))
        x = self.norm3(x + y)

        # MEMORY SAVER: Return None for attentions during training
        # to prevent storing massive tensors for backprop
        if self.training:
             return x, None, None, None, None
        
        return x, attn_t_self, attn_s_self, attn_t_cross, attn_s_cross

    def forward(self, x, memory, x_mask=None, cross_mask=None, x_rope=None, cross_rope=None):
        # Checkpoint condition: Training mode AND inputs require gradients
        if self.training and x.requires_grad:
            # We must pass all arguments positionally to checkpoint
            out = checkpoint(
                self._forward_impl, 
                x, 
                memory, 
                x_mask, 
                cross_mask, 
                x_rope, 
                cross_rope,
                use_reentrant=False
            )
            # Unpack the flat return structure
            x, attn_t_self, attn_s_self, attn_t_cross, attn_s_cross = out
        else:
            # Standard forward pass
            x, attn_t_self, attn_s_self, attn_t_cross, attn_s_cross = self._forward_impl(
                x, memory, x_mask, cross_mask, x_rope, cross_rope
            )

        # Repack into the structure your model expects: (output, (tuple of attns))
        return x, (attn_t_self, attn_s_self, attn_t_cross, attn_s_cross)
    
class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None, x_rope=None, cross_rope=None):
        attns = []

        for layer in self.layers:
            x, attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, x_rope=x_rope, cross_rope=cross_rope)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns