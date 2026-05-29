# -*- coding: utf-8 -*-
"""Multi-Head Self-Attention 과제 템플릿."""

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
    GPT의 causal self-attention을 구현합니다.

    구현할 핵심:
    - Q/K/V projection
    - head 분리: (B, T, C) -> (B, n_heads, T, head_dim)
    - attention score = QK^T / sqrt(head_dim)
    - causal mask로 미래 토큰 가리기
    - attention weight와 V를 곱한 뒤 head를 다시 합치기
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout = nn.Dropout(drop_rate)
        self.W_q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: bool = True,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: multi-head attention forward를 구현합니다.

        Args:
            x: (batch_size, seq_len, d_model)
            causal_mask: True이면 미래 위치를 볼 수 없게 mask 처리
            return_attention_weights: True이면 attention weight도 함께 반환
        """
        #nn.Linear의 경우 x @ W_q가 아니라 w_q(x)로 나타내야함
        batch_size, token_size, dimension_size = x.shape
        if dimension_size != self.d_model:
            raise ValueError("last dimension of x must match d_model")

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch_size, token_size, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, token_size, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, token_size, self.n_heads, self.head_dim).transpose(1, 2)

        attn_score = q @ k.transpose(-2, -1)
        attn_score = attn_score / (self.head_dim ** 0.5)

        if causal_mask:
            mask = torch.triu(
                torch.ones(token_size, token_size, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_score = attn_score.masked_fill(mask.bool(), float("-inf"))

        attn_weight = torch.softmax(attn_score, dim=-1)
        attn_weight = self.dropout(attn_weight)

        context = attn_weight @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            token_size,
            self.d_model,
        )
        context = self.out_proj(context)

        if return_attention_weights:
            return (context, attn_weight)
        else:
            return context
