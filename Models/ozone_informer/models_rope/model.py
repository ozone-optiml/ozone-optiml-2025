import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.masking import TriangularCausalMask, ProbMask
from .encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack, TemporalAttentionWrapper, SpatialAttentionWrapper
from .decoder import Decoder, DecoderLayer, TemporalCrossAttentionWrapper, SpatialCrossAttentionWrapper
from .attn import FullAttention, ProbAttention, FlashAttention, AttentionLayer
from .embed import DataEmbedding, GridDataEmbedding
import time

class Informer_RoPE(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention=False, distil=False, mix=False, 
                device=None):
        super().__init__()
        # auto device select
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        # Pass n_heads to Embeddings to ensure RoPE dims match head dims
        self.enc_embedding = GridDataEmbedding(enc_in, d_model, dropout=dropout, n_heads=n_heads)
        self.dec_embedding = GridDataEmbedding(dec_in, d_model, dropout=dropout, n_heads=n_heads)
        
        # Attention
        Attn = ProbAttention if attn=='prob' else FlashAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    TemporalAttentionWrapper(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False
                        )
                    ),
                    SpatialAttentionWrapper(
                        AttentionLayer(
                            Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                            d_model, n_heads, mix=False
                        )
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            distill_layers=None if not distil else [
                ConvLayer(d_model) for _ in range(e_layers-1)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )
        
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    TemporalAttentionWrapper(
                        AttentionLayer(
                            Attn(True, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads, mix=mix
                        )
                    ),
                    SpatialAttentionWrapper(
                        AttentionLayer(
                            Attn(True, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads, mix=mix
                        )
                    ),
                    TemporalCrossAttentionWrapper(
                        AttentionLayer(
                            FlashAttention(False, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads, mix=False
                        )
                    ),
                    SpatialCrossAttentionWrapper(
                        AttentionLayer(
                            FlashAttention(False, factor, attention_dropout=dropout, output_attention=False),
                            d_model, n_heads, mix=False
                        )
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(d_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        # Projection
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
        self.to(self.device)

    def forward(self, x_enc, x_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        # Embeddings return (x, rope)
        enc_out, enc_rope = self.enc_embedding(x_enc)
        
        # Pass rope to encoder
        enc_out, enc_attns = self.encoder(
            enc_out,
            attn_mask=enc_self_mask,
            rope=enc_rope
        )
        
        dec_out, dec_rope = self.dec_embedding(x_dec)
        
        # Pass ropes to decoder (dec_rope for self, enc_rope for cross)
        dec_out, dec_attns = self.decoder(
            dec_out,
            enc_out,
            x_mask=dec_self_mask,
            cross_mask=dec_enc_mask,
            x_rope=dec_rope,
            cross_rope=enc_rope
        )

        # projection: flatten last dim only
        B, C, R, T, D = dec_out.shape
        dec_out = self.projection(dec_out)  # [B, C, R, T, c_out]

        if self.output_attention:
            return dec_out[:, :, :, -self.pred_len:, :], (enc_attns, dec_attns)
        else:
            out = dec_out[:, :, :, -self.pred_len:, 0]
            # print("Output shape:", out.shape)  # Debug print    
            return out


class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=False, mix=True,
                device=torch.device('cuda:0')):
        super().__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FlashAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FlashAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]