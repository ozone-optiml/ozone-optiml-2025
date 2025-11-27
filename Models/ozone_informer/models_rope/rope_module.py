import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, base=10000):
        super().__init__()
        self.d_model = d_model
        
        # Create the frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache to store sin/cos so we don't recompute every forward pass
        self.max_seq_len = max_seq_len
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
        
        # Initialize cache
        self._update_cache(max_seq_len)

    def _update_cache(self, seq_len):
        """Generates the cos/sin cache for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Create [seq_len, d_model] tensor
        emb = torch.cat((freqs, freqs), dim=-1)
        
        self.cos_cached = emb.cos()[None, None, :, :] # [1, 1, Seq, D]
        self.sin_cached = emb.sin()[None, None, :, :] # [1, 1, Seq, D]
        self.max_seq_len = seq_len

    def forward(self, x, seq_dim=2):
        """
        Args:
            x: Query or Key tensor [Batch, Heads, SeqLen, HeadDim]
            seq_dim: The dimension corresponding to sequence length (usually 2)
        Returns:
            cos, sin tensors ready for broadcasting
        """
        seq_len = x.shape[seq_dim]
        
        # Dynamically update cache if sequence length exceeds current cache
        if seq_len > self.max_seq_len:
            self._update_cache(seq_len)
            
        # Slice the cache to the current sequence length
        return (
            self.cos_cached[:, :, :seq_len, ...], 
            self.sin_cached[:, :, :seq_len, ...]
        )

class RoPE(nn.Module):
    def __init__(self, d_model, max_len=500, base=10000, 
                 time_ratio=0.5, h_ratio=0.25, w_ratio=0.25):
        """
        RoPE (Rotary Position Embedding) for Spatiotemporal Grid Data.
        Splits d_model (per head) into chunks for Time, Height (Row), and Width (Col).
        
        Default splits:
        - Time: 50%
        - Height: 25%
        - Width: 25%
        """
        super().__init__()
        assert time_ratio + h_ratio + w_ratio == 1.0, "Ratios must sum to 1"
        
        # Ensure d_model is even and splits align (roughly)
        self.d_time = int(d_model * time_ratio)
        self.d_h = int(d_model * h_ratio)
        self.d_w = d_model - self.d_time - self.d_h 

        # Helper to create freq bands
        def get_inv_freq(dim):
            return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq_t", get_inv_freq(self.d_time))
        self.register_buffer("inv_freq_h", get_inv_freq(self.d_h))
        self.register_buffer("inv_freq_w", get_inv_freq(self.d_w))
        
        # Caches for 1D components
        self.max_len = max_len
        self.register_buffer("cos_t", None, persistent=False)
        self.register_buffer("sin_t", None, persistent=False)
        self.register_buffer("cos_h", None, persistent=False)
        self.register_buffer("sin_h", None, persistent=False)
        self.register_buffer("cos_w", None, persistent=False)
        self.register_buffer("sin_w", None, persistent=False)
        
        # Initial Cache Population
        self._update_cache(max_len)

    def _update_cache(self, length):
        # We update all caches to 'length' for simplicity, 
        # though time/width/height might have different max constraints.
        t = torch.arange(length, device=self.inv_freq_t.device, dtype=self.inv_freq_t.dtype)
        
        # Time
        freqs_t = torch.einsum("i,j->ij", t, self.inv_freq_t)
        emb_t = torch.cat((freqs_t, freqs_t), dim=-1)
        self.cos_t = emb_t.cos() # [L, d_t]
        self.sin_t = emb_t.sin()

        # Height
        freqs_h = torch.einsum("i,j->ij", t, self.inv_freq_h)
        emb_h = torch.cat((freqs_h, freqs_h), dim=-1)
        self.cos_h = emb_h.cos() # [L, d_h]
        self.sin_h = emb_h.sin()

        # Width
        freqs_w = torch.einsum("i,j->ij", t, self.inv_freq_w)
        emb_w = torch.cat((freqs_w, freqs_w), dim=-1)
        self.cos_w = emb_w.cos() # [L, d_w]
        self.sin_w = emb_w.sin()
        
        self.max_len = length

    def forward(self, B, C, R, T):
        """
        Constructs the 3D grid RoPE on the fly using cached 1D components.
        C: Width (Cols)
        R: Height (Rows)
        T: Time
        Returns: cos, sin of shape [B, C, R, T, d_model]
        """
        max_req = max(C, R, T)
        if max_req > self.max_len:
            self._update_cache(max_req)
            
        # Select slices based on dimensions
        # Time [T, d_t] -> [1, 1, 1, T, d_t]
        c_t = self.cos_t[:T, :].view(1, 1, 1, T, -1)
        s_t = self.sin_t[:T, :].view(1, 1, 1, T, -1)
        
        # Height [R, d_h] -> [1, 1, R, 1, d_h]
        c_h = self.cos_h[:R, :].view(1, 1, R, 1, -1)
        s_h = self.sin_h[:R, :].view(1, 1, R, 1, -1)
        
        # Width [C, d_w] -> [1, C, 1, 1, d_w]
        c_w = self.cos_w[:C, :].view(1, C, 1, 1, -1)
        s_w = self.sin_w[:C, :].view(1, C, 1, 1, -1)
        
        # Broadcast to [1, C, R, T, d_sub]
        c_t = c_t.expand(1, C, R, T, -1)
        s_t = s_t.expand(1, C, R, T, -1)
        
        c_h = c_h.expand(1, C, R, T, -1)
        s_h = s_h.expand(1, C, R, T, -1)
        
        c_w = c_w.expand(1, C, R, T, -1)
        s_w = s_w.expand(1, C, R, T, -1)
        
        # Concatenate in feature dimension: [Time, Width, Height]
        # Order must match the split logic. 
        # Here: T(1/2), H(1/4), W(1/4) -> Concat(T, H, W)
        cos = torch.cat([c_t, c_h, c_w], dim=-1)
        sin = torch.cat([s_t, s_h, s_w], dim=-1)
        
        # Expand batch dimension
        cos = cos.expand(B, -1, -1, -1, -1)
        sin = sin.expand(B, -1, -1, -1, -1)
        
        return cos, sin
    
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies RoPE to query and key states.
    Args:
        q: Query states [Batch, Heads, SeqLen, HeadDim]
        k: Key states   [Batch, Heads, SeqLen, HeadDim]
        cos: Cosine part of embedding [Batch, SeqLen, 1, HeadDim] or broadcastable
        sin: Sine part of embedding
    """
    # Note: The cos/sin passed here might be [Batch, SeqLen, HeadDim] (missing Head dim)
    # or [Batch, SeqLen, Heads, HeadDim].
    # We need to ensure shapes align for broadcasting.
    
    # If cos/sin is [Batch, SeqLen, HeadDim], unsqueeze Head dim for broadcasting
    # Assuming standard RoPE where rotation is same across heads.
    if cos.ndim == 3: # [B, L, D]
        cos = cos.unsqueeze(2) # [B, L, 1, D]
        sin = sin.unsqueeze(2)
        
    # Transpose q, k from [B, H, L, D] to [B, L, H, D] for easier multiplication
    # Or Transpose cos/sin to [B, 1, L, D] if q is [B, H, L, D]
    
    # Current standard q shape in AttentionLayer is [B, L, H, D] (before transpose for Flash)
    # If q is [B, L, H, D]:
    if q.shape[1] == cos.shape[1]: # Lengths match on dim 1
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        # Fallback/Safety: Try to align dimensions assuming q is [B, H, L, D]
        # and cos is [B, L, 1, D]
        # Permute cos to [B, 1, L, D]
        cos = cos.permute(0, 2, 1, 3)
        sin = sin.permute(0, 2, 1, 3)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed