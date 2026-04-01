from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torch import Tensor

import modern_lm.utils as utils

USE_FLASH_ATTN = os.getenv("USE_FLASH_ATTN") == "1"

if USE_FLASH_ATTN:
    try:
        from kernels import get_kernel
    except ImportError as err:
        raise ImportError(
            "USE_FLASH_ATTN is set to 1 but `kernels` module is not available."
        ) from err
    fa3 = get_kernel("kernels-community/flash-attn3")
    flash_attn_func: Callable[..., Tensor] | None = fa3.flash_attn_func
else:
    flash_attn_func = None


@dataclass
class ModernLMConfig:
    vocab_size: int = 50257
    seq_length: int = 1024
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: int | None = None
    d_ff: int = 2048  # use 8/3 * d_model to achive the same number of parameters compare to FFN when switching to SwiGLU
    dropout: float = 0.0
    eps: float = 1e-7
    tie_weights: bool = True
    rope_theta: float = 10000.0
    attn_logit_softcapping: float | None = None
    final_logit_softcapping: float | None = None
    rms_norm_eps: float = 1e-5


def norm(x: Tensor, eps: float = 1e-5) -> Tensor:
    return Fun.rms_norm(x, (x.shape[-1],), eps=eps)


def softcap(x: Tensor, cap: float) -> Tensor:
    return torch.tanh(x / cap) * cap


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
    dropout: float | nn.Dropout | None = None,
    softcapping: float | None = None,
) -> Tensor:
    head_dim = query.size(-1)
    attention_probs = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
    if softcapping is not None and softcapping > 0.0:
        attention_probs = attention_probs.float()
        attention_probs = softcap(attention_probs, softcapping)
    if mask is not None:
        attention_probs.masked_fill_(mask == False, float("-inf"))  # noqa: E712

    attention_probs = Fun.softmax(attention_probs, dim=-1)
    if dropout is not None:
        if isinstance(dropout, float):
            dropout = nn.Dropout(dropout)
        attention_probs = dropout(attention_probs)

    output = attention_probs @ value
    return output


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch, seq_length, num_heads, head_dim).
        cos (torch.Tensor): Precomputed cosine values of shape (seq_length, head_dim/2).
        sin (torch.Tensor): Precomputed sine values of shape (seq_length, head_dim/2).

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied, shape (batch, seq_length, num_heads, head_dim).
    """
    assert x.ndim == 4
    d = x.size(-1) // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat([y1, y2], dim=-1).type_as(x)


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(
        self,
        dim: int,
        base_theta: float = 10000.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.dim = dim
        self.base = base_theta
        self.max_seq_len = max_seq_len

        # Precompute frequencies
        # This matches the "1/10000^(2i/d)" formula
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

        # Register buffer so it saves with state_dict but isn't a parameter
        self.register_buffer("inv_freq", inv_freq)

        # Cache for the cosine and sine values
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # Outer product to generate frequencies for all positions
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)

        # (seq_len, dim/2) -> (1, seq_len, 1, dim/2) for broadcasting
        cos = self.cos_cache[:seq_len, ...].view(1, -1, 1, self.dim // 2)  # pyright: ignore[reportIndexIssue]
        sin = self.sin_cache[:seq_len, ...].view(1, -1, 1, self.dim // 2)  # pyright: ignore[reportIndexIssue]
        return cos, sin


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        max_seq_length: int,
        num_kv_heads: int | None = None,
        rope_theta: float = 10000.0,
        softcapping: float | None = None,
        rms_norm_eps: float = 1e-5,
    ):
        super().__init__()
        if num_kv_heads is None:
            num_kv_heads = num_heads

        if not d_model % num_heads == 0:
            raise ValueError("d_model must be divisible by num_heads")
        if num_kv_heads > num_heads:
            raise ValueError("num_kv_heads must be less than or equal to num_heads")
        if not num_heads % num_kv_heads == 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = self.d_model // self.num_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)
        self.use_flash_attn = USE_FLASH_ATTN
        self.rms_norm_eps = rms_norm_eps

        self.softcapping = softcapping if softcapping is not None else 0.0
        self.flash_attn_func = flash_attn_func
        if self.use_flash_attn and self.flash_attn_func is None:
            raise RuntimeError("USE_FLASH_ATTN=1 but flash-attn3 kernel is not available")

        self.w_q = nn.Linear(d_model, self.num_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)

        self.rl_projection = nn.Linear(self.num_heads * self.head_dim, d_model, bias=False)

        self.rope = RotaryPositionalEmbeddings(
            self.head_dim, base_theta=rope_theta, max_seq_len=max_seq_length
        )

        if not self.use_flash_attn:
            self.register_buffer(
                "causal_mask",
                torch.tril(
                    torch.ones(max_seq_length, max_seq_length).unsqueeze_(0).unsqueeze_(0)
                ).bool(),
            )

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_length, _ = x.size()

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Reshape to (batch_size, seq_length, num_heads/num_kv_heads, head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        # QK-Norm
        q = norm(q, eps=self.rms_norm_eps)
        k = norm(k, eps=self.rms_norm_eps)

        # Follow Qwen3/Gemma3 stype: applying norm and then RoPE
        cos, sin = self.rope(seq_length)
        q = apply_rotary_emb(q, cos=cos, sin=sin)
        k = apply_rotary_emb(k, cos=cos, sin=sin)

        if self.use_flash_attn and self.flash_attn_func is not None:
            # FA3 natively supports GQA if shapes match (B, S, num_heads, D) and (B, S, num_kv_heads, D)
            # FA3 does not support dropout (https://github.com/Dao-AILab/flash-attention/issues/1377#issuecomment-2529622590)
            y = self.flash_attn_func(
                q,
                k,
                v,
                causal=True,
                softcap=self.softcapping,
            )
            # returned shape is (batch_size, seq_length, num_heads, head_dim), reshape back to (batch_size, seq_length, num_heads * head_dim)
            y = y.reshape(batch_size, -1, self.num_heads * self.head_dim)
        else:
            # For standard PyTorch SDPA, we must manually repeat KV heads if num_kv_heads < num_heads
            if self.num_kv_heads != self.num_heads:
                k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
                v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

            mask = self.get_buffer("causal_mask")[..., :seq_length, :seq_length]

            y = scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                mask=mask,
                dropout=self.attention_dropout,
                softcapping=self.softcapping,
            )
            # returned shape is (batch_size, num_heads, seq_length, head_dim), reshape back to (batch_size, seq_length, num_heads * head_dim)
            y = y.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        y = self.residual_dropout(self.rl_projection(y))
        return y


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(d_model, d_ff, bias=False)
        self.rl_projection = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        x = Fun.relu(x)
        x = x * x  # calling `Fun.relu(x).square()` does not work with torch.compile()
        x = self.dropout(x)
        x = self.rl_projection(x)
        return x


class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.rl_projection = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = Fun.silu(self.w_gate(x)) * self.w_up(x)
        x = self.dropout(x)
        x = self.rl_projection(x)
        return x


class ModernLMDecoderBlock(nn.Module):
    def __init__(self, config: ModernLMConfig):
        super().__init__()
        self.rms_norm_eps = config.rms_norm_eps
        self.causal_self_attention = CausalMultiHeadSelfAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            max_seq_length=config.seq_length,
            num_kv_heads=config.num_kv_heads,
            rope_theta=config.rope_theta,
            softcapping=config.attn_logit_softcapping,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = SwiGLUFeedForward(
            d_model=config.d_model,
            d_ff=config.d_ff,
            dropout=config.dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """see: https://github.com/openai/gpt-2/blob/master/src/model.py#L123"""
        x = x + self.causal_self_attention(norm(x, eps=self.rms_norm_eps))
        x = x + self.mlp(norm(x, eps=self.rms_norm_eps))
        return x


class ModernLM(nn.Module):
    def __init__(self, config: ModernLMConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(self.config.vocab_size, self.config.d_model)
        self.decoder_blocks = nn.Sequential(*[
            ModernLMDecoderBlock(self.config) for _ in range(self.config.num_layers)
        ])
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)

        self.post_init()

    def forward(self, ids: Tensor) -> Tensor:
        token_embeddings = self.token_embedding(ids)
        x = self.decoder_blocks(token_embeddings)
        x = norm(x, eps=self.config.rms_norm_eps)
        logits = self.lm_head(x)  # (batch_size, seq_length, vocab_size)
        if (
            self.config.final_logit_softcapping is not None
            and self.config.final_logit_softcapping > 0.0
        ):
            logits = logits.float()
            logits = softcap(logits, self.config.final_logit_softcapping)
        return logits

    def post_init(self) -> None:
        self._init_model_weights()

    def tie_weights(self) -> None:
        if self.lm_head.weight.shape != self.token_embedding.weight.shape:
            raise RuntimeError(
                "When using tied weights, the weight of the last linear layer "
                "and the token embedding layer must be the same shape, "
                f"but found {self.lm_head.weight.shape} and {self.token_embedding.weight.shape}"
            )
        self.lm_head.weight = self.token_embedding.weight

    def _init_model_weights(self, std: float = 0.02) -> None:
        self.apply(lambda module: self._init_weights(module, std=std))

        # as in GPT-2 paper, weights of residual layers at initialization are scaled
        # by a factor of 1/sqrt(N) where N is the number of residual layers,
        # in this case N is equal to 2 * num_layers
        scaling_factor = 1 / math.sqrt(2 * self.config.num_layers)
        for param_name, param in self.named_parameters():
            if param_name.endswith("rl_projection.weight"):
                torch.nn.init.normal_(param, mean=0.0, std=std * scaling_factor)

    def _init_weights(self, module, std: float = 0.02):
        """ref: https://github.com/openai/gpt-2/blob/master/src/model.py#L50"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def estimate_flops_per_token(self) -> int:
        """Estimate FLOPs for a single forward + backward pass per token.

        Based on the standard approximation from the OpenAI scaling law paper
        (Kaplan et al., 2020) and PaLM (Chowdhery et al., 2022):
          - Each linear layer parameter contributes ~6 FLOPs per token
            (2 for forward matmul, 4 for backward: data grad + weight grad).
          - Attention dot-products (QK^T and softmax@V) are not captured by
            the parameter count and must be added separately.

        Ref: https://arxiv.org/abs/2001.08361 (Appendix B),
             https://www.adamcasson.com/posts/transformer-flops
        """
        config = self.config
        head_dim = config.d_model // config.num_heads
        kv_dim = (config.num_kv_heads or config.num_heads) * head_dim

        # matmul FLOPs (6 * param_count for fwd+bwd)
        # Per layer:
        #   Attention projections: Q(d_model*d_model) + K(d_model*kv_dim)
        #                        + V(d_model*kv_dim) + Out(d_model*d_model)
        #   SwiGLU FFN: gate(d_model*d_ff) + up(d_model*d_ff) + down(d_ff*d_model)
        attn_proj_params = config.d_model * (config.d_model + 2 * kv_dim + config.d_model)
        ffn_params = 3 * config.d_model * config.d_ff
        per_layer_params = attn_proj_params + ffn_params

        # lm_head is always a matmul even when weights are tied with embedding
        lm_head_params = config.d_model * config.vocab_size

        flops = 6 * (config.num_layers * per_layer_params + lm_head_params)

        # attention dot-product FLOPs
        # Per layer per token:
        #   Forward:  2 * d_model * seq_length  (QK^T)
        #           + 2 * d_model * seq_length  (attn @ V)
        #   Backward: 2x forward
        #   Total:  12 * d_model * seq_length per layer
        flops += config.num_layers * 12 * config.num_heads * head_dim * config.seq_length

        return flops

    @torch.no_grad()
    def generate(
        self,
        ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> Tensor:
        # ids has shape (batch_size, seq_length)
        ids = ids.detach().clone()

        # set model in evaluation mode
        is_training = self.training
        self.eval()

        for _ in range(max_new_tokens):
            input_ids = ids
            if ids.size(1) > self.config.seq_length:
                input_ids = ids[:, -self.config.seq_length :]

            # feed ids to the model to generate logits
            logits = self(input_ids)  # (batch_size, seq_length, vocab_size)
            logits = logits[:, -1, :]  # (batch_size, vocab_size)
            try:
                logits /= temperature
            except ZeroDivisionError:
                pass
            logits = utils.top_k_logits(logits, top_k=top_k)
            logits = utils.top_p_logits(logits, top_p=top_p)
            probs = Fun.softmax(logits, dim=-1)

            # next predicted token
            # take this
            # next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (batch_size, 1)
            # or this
            next_token = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            ids = torch.cat((ids, next_token), dim=-1)

        # set model back to training mode
        if is_training:
            self.train()

        return ids
