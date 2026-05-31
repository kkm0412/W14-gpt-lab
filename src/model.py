# -*- coding: utf-8 -*-
"""GPT 모델 구성 요소 과제 템플릿."""

import torch
import torch.nn as nn

try:
    from .attention import MultiHeadAttention
    from .embeddings import InputEmbedding
except ImportError:
    from attention import MultiHeadAttention
    from embeddings import InputEmbedding


class LayerNorm(nn.Module):
    """마지막 차원 기준 Layer Normalization."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: 마지막 차원의 평균과 분산으로 정규화한 뒤 gamma/beta를 적용합니다."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta
        raise NotImplementedError("LayerNorm.forward를 구현하세요.")


class GELU(nn.Module):
    """GPT FeedForward에서 사용하는 GELU 활성화 함수."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: tanh 근사식 또는 torch 연산으로 GELU를 구현합니다."""
        return 0.5 * x * (
            1.0
            + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Transformer FFN: Linear -> GELU -> Linear -> Dropout."""

    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        # TODO: d_model -> mult*d_model -> d_model 구조의 작은 MLP를 정의하세요.
        self.input_layer = nn.Linear(d_model, mult * d_model)
        self.output_layer = nn.Linear(mult * d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.input_layer,
            GELU(),
            self.output_layer,
            self.dropout,
        )
        # raise NotImplementedError("FeedForward.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO: FeedForward 네트워크를 통과시킵니다."""
        return self.net(x)
        # raise NotImplementedError("FeedForward.forward를 구현하세요.")


class TransformerBlock(nn.Module):
    """
    GPT block: LayerNorm -> Causal Self-Attention -> residual,
    LayerNorm -> FeedForward -> residual.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
    ):
        super().__init__()

        # self.net1 = nn.Sequential(
        #     LayerNorm(d_model),
        #     MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias),
        #     nn.Dropout(drop_rate),
        # )
        # self.net2 = nn.Sequential(
        #     LayerNorm(d_model),
        #     FeedForward(d_model, drop_rate, 4),
        #     nn.Dropout(drop_rate),
        # )
        self.attention = MultiHeadAttention(d_model, n_heads, drop_rate, qkv_bias)
        self.ffn = FeedForward(d_model, drop_rate, 4)
        self.layernorm1 = LayerNorm(d_model)
        self.layernorm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)
        # TODO: attention, ffn, layernorm, dropout을 정의하세요.
        # raise NotImplementedError("TransformerBlock.__init__을 구현하세요.")

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        """TODO: attention과 ffn을 residual connection으로 연결합니다."""
        # 전체 트랜스포머 모델 작동 순서(gpt모델 안에서 작동)
        shortcut = x
        x = self.layernorm1(x)
        x = self.attention(x, causal_mask)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layernorm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + shortcut
        return x
        raise NotImplementedError("TransformerBlock.forward를 구현하세요.")


class GPTModel(nn.Module):
    """InputEmbedding -> TransformerBlock N개 -> LayerNorm -> LM head."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        # TODO: embedding, blocks, final layernorm, lm_head를 정의하세요.
        self.embedding = InputEmbedding(
            config["vocab_size"],
            config["emb_dim"],
            config["context_length"],
            config["drop_rate"]
            )
        self.blocks = nn.ModuleList(
            [
            TransformerBlock(
                config["emb_dim"],
                config["n_heads"],
                config["drop_rate"],
                config["qkv_bias"]
                )
            for _ in range(config["n_layers"])
            ]
            )
        self.finalnorm = LayerNorm(config["emb_dim"])
        self.lm_head = nn.Linear(config["emb_dim"], config["vocab_size"])
        # raise NotImplementedError("GPTModel.__init__을 구현하세요.")

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: logits를 만들고, targets가 있으면 cross entropy loss도 함께 반환합니다.
        Returns:
            targets가 None이면 logits
            targets가 있으면 (loss, logits)
        """
        x = self.embedding(idx)
        for block in self.blocks:
            x = block(x)
        x = self.finalnorm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            return loss, logits
        return logits


def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
) -> torch.Tensor:
    """TODO: greedy 방식으로 max_new_tokens만큼 다음 토큰을 이어 붙입니다."""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size: ]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[: , -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
    raise NotImplementedError("generate_text_simple을 구현하세요.")
