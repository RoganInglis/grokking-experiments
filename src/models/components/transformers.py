import math
import torch
import numpy as np
from torch import nn

import einops


def create_attn_mask(seq_len):
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


class Softmax1(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # subtract the max for stability
        x = x - x.max(dim=self.dim, keepdim=True).values

        return torch.exp(x) / 1 + (torch.exp(x).sum(dim=self.dim, keepdim=True))


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.d_model = dim_model

        period = 1 / 10000.0 ** (torch.arange(0, dim_model, 2).float() / dim_model)
        self.register_buffer('period', period)

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device).type_as(self.period)
        pos = pos.unsqueeze(1) * self.period.unsqueeze(0)
        pos = torch.stack((pos.sin(), pos.cos()), dim=-1).flatten(-2)
        return x + pos.unsqueeze(0)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=512):
        super().__init__()
        self.d_model = dim_model
        self.pe = nn.Embedding(max_len, dim_model, _weight=torch.randn(max_len, dim_model) / np.sqrt(dim_model))

    def forward(self, x):
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0)
        return x + self.pe(pos)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        Computes a linear layer with sqrt initialization
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_features, out_features) / np.sqrt(out_features))
        self.bias = bias
        if self.bias:
            self.b = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = x @ self.W
        if self.bias:
            x += self.b
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dropout=0., use_softmax1: bool = False):
        """
        Computes the multi-head attention from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dropout: The dropout probability
        """
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_heads = dim_heads
        self.dim_inner = num_heads * dim_heads

        self.scale = dim_heads ** -0.5

        self.w_q = Linear(in_features=dim_model, out_features=self.dim_inner, bias=False)
        self.w_k = Linear(in_features=dim_model, out_features=self.dim_inner, bias=False)
        self.w_v = Linear(in_features=dim_model, out_features=self.dim_inner, bias=False)
        self.w_out = Linear(in_features=self.dim_inner, out_features=dim_model, bias=False)
        if use_softmax1:
            self.softmax = Softmax1(dim=-1)
        else:
            self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_k, x_v, mask=None):
        """
        :param x_q: Tensor of shape (batch_size, seq_len_q, dim_model) containing the query vectors
        :param x_k: Tensor of shape (batch_size, seq_len_k, dim_model) containing the key vectors
        :param x_v: Tensor of shape (batch_size, seq_len_v, dim_model) containing the value vectors
        :param mask: Tensor of shape (seq_len_q, seq_len_k) containing the mask for the attention
        :return:
        """
        # Project to query, key, value and split into heads
        q = einops.rearrange(self.w_q(x_q), 'b s (h d) -> b h s d', h=self.num_heads)
        k = einops.rearrange(self.w_k(x_k), 'b s (h d) -> b h s d', h=self.num_heads)
        v = einops.rearrange(self.w_v(x_v), 'b s (h d) -> b h s d', h=self.num_heads)

        # Calculate attention weights
        attn = torch.einsum('b h q d, b h k d -> b h q k', q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        w_attn = self.softmax(attn)
        w_attn = self.dropout(w_attn)

        # Calculate output
        out = torch.einsum('b h q k, b h k d -> b h q d', w_attn, v)
        out = einops.rearrange(out, 'b h s d -> b s (h d)')
        return self.w_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner=2048, dropout=0.):
        """
        Computes the feed-forward layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param dim_inner: The dimensionality of the inner layer
        :param dropout:
        """
        super().__init__()

        # TODO - it is possible to optimize this by removing the padding before and adding it after the linear layers
        # TODO - allow for different activation functions

        self.net = nn.Sequential(
            Linear(dim_model, dim_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(dim_inner, dim_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ParallelFeedForward(nn.Module):
    def __init__(self, dim_model=512, dim_inner=2048, dropout=0., n_parallel=1):
        super().__init__()
        self.layers = nn.ModuleList([FeedForward(dim_model, dim_inner, dropout) for _ in range(n_parallel)])

    def forward(self, x):
        return torch.sum(torch.stack([layer(x) for layer in self.layers]), dim=0)


class Residual(nn.Module):
    def __init__(self, fn):
        """
        Computes the residual connection from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param fn: The function to apply the residual connection to
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    def __init__(self, dim_model, eps=1e-5):
        """
        Computes the layer normalization from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param eps: The epsilon value for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim_model))
        self.beta = nn.Parameter(torch.zeros(dim_model))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class EncoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, dropout=0., layer_norm=True,
                 use_softmax1: bool = False):
        """
        Computes the encoder layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dim_inner: The dimensionality of the inner feedforward layer
        :param dropout: The dropout probability
        """
        super().__init__()
        self.layer_norm = layer_norm

        self.attn = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout, use_softmax1=use_softmax1))
        self.ff = Residual(FeedForward(dim_model, dim_inner, dropout))

        if self.layer_norm:
            self.norm_1 = LayerNorm(dim_model)
            self.norm_2 = LayerNorm(dim_model)

    def forward(self, x, mask=None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model). Input to the encoder layer.
        :param mask: Tensor of shape (seq_len, seq_len). Mask to be applied to the attention weights.
        :return:
        """
        x = self.attn(x, x, x, mask)
        if self.layer_norm:
            x = self.norm_1(x)
        x = self.ff(x)
        if self.layer_norm:
            return self.norm_2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, dropout=0., layer_norm=True,
                 use_softmax1: bool = False):
        """
        Computes the decoder layer from the paper Attention is all you need (https://arxiv.org/abs/1706.03762)
        :param dim_model: The dimensionality of the input and output vectors
        :param num_heads: The number of attention heads
        :param dim_heads: The dimensionality of the attention heads
        :param dim_inner: The dimensionality of the inner feedforward layer
        :param dropout: The dropout probability
        :param layer_norm: Whether to apply layer normalization
        """
        super().__init__()
        self.layer_norm = layer_norm

        self.attn_1 = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout, use_softmax1=use_softmax1))
        self.attn_2 = Residual(MultiHeadAttention(dim_model, num_heads, dim_heads, dropout, use_softmax1=use_softmax1))
        self.ff = Residual(FeedForward(dim_model, dim_inner, dropout))
        if self.layer_norm:
            self.norm_1 = LayerNorm(dim_model)
            self.norm_2 = LayerNorm(dim_model)
            self.norm_3 = LayerNorm(dim_model)

    def forward(self, x, mem, src_mask=None, tgt_mask=None):
        """
        :param x: Tensor of shape (batch_size, seq_len, dim_model). The input to the decoder.
        :param mem: Tensor of shape (batch_size, seq_len, dim_model). The memory from the encoder.
        :param src_mask: Tensor of shape (1, seq_len). The mask for the encoder.
        :param tgt_mask: Tensor of shape (seq_len, seq_len). The mask for the decoder.
        :return:
        """
        x = self.attn_1(x, x, x, tgt_mask)
        if self.layer_norm:
            x = self.norm_1(x)
        x = self.attn_2(x, mem, mem, src_mask)
        if self.layer_norm:
            x = self.norm_2(x)
        x = self.ff(x)
        if self.layer_norm:
            return self.norm_3(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.,
                 layer_norm=True, use_softmax1: bool = False):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(
                dim_model,
                num_heads,
                dim_heads,
                dim_inner,
                dropout,
                layer_norm=layer_norm,
                use_softmax1=use_softmax1
            )
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerWrapper(nn.Module):
    def __init__(self, vocab_size, dim_model=512, num_heads=8, dim_heads=64, dim_inner=2048, num_layers=6, dropout=0.,
                 tie_emb_weights=True, max_len=1000, layer_norm=True, use_softmax1: bool = False,
                 num_registers: int = 0):
        super().__init__()

        self.vocab_size = vocab_size
        self.emb = nn.Parameter(torch.randn(vocab_size, dim_model) / np.sqrt(dim_model))
        self.pos_enc = LearnedPositionalEncoding(dim_model, max_len=max_len)

        self.num_registers = num_registers
        if num_registers > 0:
            self.register = nn.Parameter(torch.randn(num_registers, dim_model) / np.sqrt(dim_model))

        self.dropout = nn.Dropout(dropout)
        self.transformer = Transformer(
            dim_model=dim_model,
            num_heads=num_heads,
            dim_heads=dim_heads,
            dim_inner=dim_inner,
            num_layers=num_layers,
            dropout=dropout,
            layer_norm=layer_norm,
            use_softmax1=use_softmax1
        )
        self.tie_emb_weights = tie_emb_weights
        if not self.tie_emb_weights:
            self.out = nn.Parameter(torch.randn(dim_model, vocab_size) / np.sqrt(vocab_size))

        self.register_buffer('attn_mask', create_attn_mask(max_len))

    def forward(self, x):
        x = self.emb[x]
        x = self.pos_enc(x)  # / math.sqrt(self.emb.embedding_dim)))

        if self.num_registers > 0:
            x = torch.cat([x, self.register.unsqueeze(0).repeat(x.shape[0], 1, 1)], dim=1)

        x = self.dropout(x)

        mask = self.attn_mask[:x.size(1), :x.size(1)]

        if self.num_registers > 0:
            mask = torch.cat([mask, torch.zeros(mask.shape[0], self.num_registers, dtype=mask.dtype, device=mask.device)], dim=1)
            mask = torch.cat([mask, torch.zeros(self.num_registers, mask.shape[1], dtype=mask.dtype, device=mask.device)], dim=0)

        x = self.transformer(x, mask=mask)

        if self.num_registers > 0:
            x = x[:, :-self.num_registers, :]

        if self.tie_emb_weights:
            return x @ self.emb.T
        else:
            return x @ self.out
