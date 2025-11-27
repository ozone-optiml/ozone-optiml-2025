import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as sdpa

import numpy as np
from math import sqrt
from ..utils.masking import TriangularCausalMask, ProbMask
from .rope_module import apply_rotary_pos_emb

class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None,
                 attention_dropout=0.0, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = float(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # queries, keys, values: [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # rearrange to [B, H, L, E] for SDPA
        q = queries.transpose(1, 2).contiguous() # [B, H, L, E]
        k = keys.transpose(1, 2).contiguous()    # [B, H, S, E]
        v = values.transpose(1, 2).contiguous()  # [B, H, S, D]

        # Note: RoPE is applied in AttentionLayer BEFORE calling this, so q, k are already rotated.
        
        # Cast to float16 for Flash Attention if supported/desired, though here we stick to input dtype unless specified
        # or force float16 as per original code
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        out = sdpa(q, k, v,
                   dropout_p=self.dropout,
                   is_causal=False)  # [B, H, L, D]

        # back to [B, L, H, D]
        out = out.transpose(1, 2).contiguous().to(torch.float32)

        return (out, None)

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1./sqrt(E)

        # q, k already rotated
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top): 
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k)) 
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :] 
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            assert(L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        # queries, keys, values: [B, L, H, D] coming in
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        # Transpose for internal logic [B, H, L, D]
        queries = queries.transpose(2,1)
        keys = keys.transpose(2,1)
        values = values.transpose(2,1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = U_part if U_part<L_K else L_K
        u = u if u<L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u) 

        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2,1).contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super().__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, rotary_pos_emb=None):
        # rotary_pos_emb: tuple of (cos, sin) OR (cos_q, sin_q, cos_k, sin_k)
        
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Apply RoPE if provided
        if rotary_pos_emb is not None:
            if len(rotary_pos_emb) == 4:
                # Cross Attention case: explicit Q and K embeddings
                cos_q, sin_q, cos_k, sin_k = rotary_pos_emb
                # Apply to Queries (ignoring 2nd return value)
                queries, _ = apply_rotary_pos_emb(queries, queries, cos_q, sin_q)
                # Apply to Keys (ignoring 1st return value)
                _, keys = apply_rotary_pos_emb(keys, keys, cos_k, sin_k)
            else:
                # Self Attention case: shared embeddings
                cos, sin = rotary_pos_emb
                queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn