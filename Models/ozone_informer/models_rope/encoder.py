import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class TemporalAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None, rope=None):
        # x: [B, C, R, T, D]
        # rope: (cos, sin) [B, C, R, T, d_head]
        B, C, R, T, D = x.shape

        # Flatten for temporal attention: (B*CR, T, D)
        x_flat = x.view(B * C * R, T, D)
        
        rope_flat = None
        if rope is not None:
            cos, sin = rope
            # Reshape rope to match x layout: [B*CR, T, d_head]
            cos_flat = cos.view(B * C * R, T, -1)
            sin_flat = sin.view(B * C * R, T, -1)
            rope_flat = (cos_flat, sin_flat)

        # AttentionLayer expects (queries, keys, values)
        out, attn = self.attn_layer(x_flat, x_flat, x_flat, attn_mask=attn_mask, rotary_pos_emb=rope_flat)

        # Restore shape
        out = out.view(B, C, R, T, D)
        return out, attn


class SpatialAttentionWrapper(nn.Module):
    def __init__(self, attn_layer):
        super().__init__()
        self.attn_layer = attn_layer

    def forward(self, x, attn_mask=None, rope=None):
        # x: [B, C, R, T, D]
        B, C, R, T, D = x.shape

        # Flatten for spatial attention: (B*T, C*R, D)
        x_flat = x.permute(0, 3, 1, 2, 4).contiguous()  # [B, T, C, R, D]
        x_flat = x_flat.view(B * T, C * R, D)
        
        rope_flat = None
        if rope is not None:
            cos, sin = rope
            # Permute and flatten rope: [B, T, C, R, d_head] -> [B*T, C*R, d_head]
            cos_flat = cos.permute(0, 3, 1, 2, 4).contiguous().view(B * T, C * R, -1)
            sin_flat = sin.permute(0, 3, 1, 2, 4).contiguous().view(B * T, C * R, -1)
            rope_flat = (cos_flat, sin_flat)

        # AttentionLayer expects (queries, keys, values)
        out, attn = self.attn_layer(x_flat, x_flat, x_flat, attn_mask=attn_mask, rotary_pos_emb=rope_flat)

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


# class EncoderLayer(nn.Module):
#     def __init__(self, temporal_attn, spatial_attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
#         super().__init__()
#         d_ff = d_ff or 4*d_model
#         # temporal and spatial attention
#         self.temporal_attn = temporal_attn
#         self.spatial_attn = spatial_attn

#         # fusion after concat
#         self.fusion = nn.Linear(2 * d_model, d_model)

#         # feed-forward network
#         self.linear1 = nn.Linear(d_model, d_ff)
#         self.linear2 = nn.Linear(d_ff, d_model)

#         # normalization, dropout, activation
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu

#     def forward(self, x, attn_mask=None, rope=None):
#         # x: [B, C, R, T, D]
#         # temporal + spatial attention
#         x_t, attn_t = self.temporal_attn(x, attn_mask=attn_mask, rope=rope)
#         x_s, attn_s = self.spatial_attn(x, attn_mask=attn_mask, rope=rope)

#         # concat + fusion
#         new_x = torch.cat([x_t, x_s], dim=-1)   # [B, C, R, T, 2D]
#         new_x = self.fusion(new_x)              # [B, C, R, T, D]

#         # residual + norm
#         x = x + self.dropout(new_x)             # [B, C, R, T, D]
#         y = self.norm1(x)

#         # feed-forward (applied per gridpoint/time)
#         y = self.dropout(self.activation(self.linear1(y)))
#         y = self.dropout(self.linear2(y))

#         return self.norm2(x + y), (attn_t, attn_s)

# Uses gradient checkpointing to reduce memory usage
class EncoderLayer(nn.Module):
    def __init__(self, temporal_attn, spatial_attn, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4*d_model
        self.temporal_attn = temporal_attn
        self.spatial_attn = spatial_attn
        self.fusion = nn.Linear(2 * d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    # 1. Isolate the heavy computation in a helper method
    def _forward_impl(self, x, attn_mask, rope):
        # x: [B, C, R, T, D]
        x_t, attn_t = self.temporal_attn(x, attn_mask=attn_mask, rope=rope)
        x_s, attn_s = self.spatial_attn(x, attn_mask=attn_mask, rope=rope)

        new_x = torch.cat([x_t, x_s], dim=-1)
        new_x = self.fusion(new_x)

        x = x + self.dropout(new_x)
        y = self.norm1(x)

        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))
        
        return self.norm2(x + y), attn_t, attn_s

    def forward(self, x, attn_mask=None, rope=None):
        # 2. Check if we should use checkpointing
        # Condition: Training mode AND input requires grad
        if self.training and x.requires_grad:
            # 3. Call checkpoint
            # Note: We flatten the return values because checkpoint expects tensors
            out, attn_t, attn_s = checkpoint(
                self._forward_impl, 
                x, 
                attn_mask, 
                rope, 
                use_reentrant=False # Recommended for modern PyTorch (less overhead)
            )
            return out, (attn_t, attn_s)
        else:
            # Standard forward pass for validation/inference
            out, attn_t, attn_s = self._forward_impl(x, attn_mask, rope)
            return out, (attn_t, attn_s)

class Encoder(nn.Module):
    def __init__(self, attn_layers, distill_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.distill_layers = nn.ModuleList(distill_layers) if distill_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, rope=None):
        # x: [B, C, R, T, D]
        attns = []
        if self.distill_layers is not None:
            # Note: Distillation changes time dimension T -> T/2
            # The 'rope' embedding calculated at the start is for original T.
            # If we distill, we break the alignment of the pre-calculated RoPE.
            # RoPE must be recalculated or sliced if we distill.
            # However, RoPE implementation here is passed from outside (embed.py).
            # If T changes, the passed 'rope' is invalid for the next layer.
            # RePE generally isn't compatible with pooling unless we pool the frequencies/positions too.
            # For this implementation, we assume if Distill is ON, we might need to handle this.
            # But the user only asked to replace PE. 
            # We will slice the RoPE simply by striding if T reduces.
            
            curr_rope = rope
            
            for attn_layer, distill_layer in zip(self.attn_layers, self.distill_layers):
                # attention
                x, attn = attn_layer(x, attn_mask=attn_mask, rope=curr_rope)
                # temporal distill
                x = distill_layer(x)
                attns.append(attn)
                
                # Update Rope for next layer if T changed
                # Distill is MaxPool1d kernel 3 stride 2.
                # We can simulate this on RoPE cos/sin by striding?
                # cos: [B, C, R, T, D]
                # We need [B, C, R, T_new, D]
                # Roughly T_new = (T-1)/2. 
                # This is tricky without generating fresh RoPE.
                # Assuming Standard Informer without Distill for RePE correctness or user accepts approximation.
                # We will skip RoPE update logic here to keep it simple, but warn.
                # Ideally, Encoder should re-query RePE with new T. But RePE is in Embed layer.
                # Workaround: stride rope.
                if curr_rope is not None:
                     c, s = curr_rope
                     # stride 2 on T dimension (dim 3)
                     # Note: MaxPool selects max, but position is roughly preserved.
                     # Taking every 2nd element is a crude approx.
                     c = c[:, :, :, ::2, :]
                     s = s[:, :, :, ::2, :]
                     # Trim to match x
                     T_new = x.shape[3]
                     c = c[:, :, :, :T_new, :]
                     s = s[:, :, :, :T_new, :]
                     curr_rope = (c, s)

            # last attention without distill
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask, rope=curr_rope)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, rope=rope)
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
        # Legacy support, not updated for RoPE as it uses 1D inputs
        x_stack = []; attns = []
        for i_len, encoder in zip(self.inp_lens, self.encoders):
            inp_len = x.shape[1]//(2**i_len)
            x_s, attn = encoder(x[:, -inp_len:, :])
            x_stack.append(x_s); attns.append(attn)
        x_stack = torch.cat(x_stack, -2)
        
        return x_stack, attns